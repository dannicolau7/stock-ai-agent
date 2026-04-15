"""
performance_tracker.py — Signal outcome tracking and win-rate analytics.

Every time the pipeline fires a BUY or SELL alert, record_signal() is called
to log it. A background loop then checks the price 1d / 3d / 7d later and
computes actual returns, storing everything in SQLite.

Tables:
  signals  — one row per alert (BUY/SELL), with entry conditions
  outcomes — price checkpoints at 1d / 3d / 7d, return %, win/loss

Exposes:
  record_signal(state)        — call from alert_node after a real alert fires
  get_stats(lookback=30)      — win rate, avg return, best setups (for reflection agent)
  get_open_signals()          — signals without full outcome yet (for portfolio agent)
  performance_tracker_loop()  — async background loop that fills in outcomes
"""

import asyncio
import json
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path

import yfinance as yf

DB_PATH = Path(__file__).parent / "data" / "performance.db"
CHECK_INTERVAL = 60 * 60   # check outcomes every hour

# ── DB setup ──────────────────────────────────────────────────────────────────

def _get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with _get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS signals (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker          TEXT    NOT NULL,
                signal          TEXT    NOT NULL,   -- BUY / SELL
                price           REAL    NOT NULL,
                confidence      INTEGER NOT NULL,
                stop_loss       REAL,
                target_1        REAL,
                target_2        REAL,
                trade_horizon   TEXT,
                news_triggered  INTEGER DEFAULT 0,
                macro_regime    TEXT,
                macro_bias      TEXT,
                geo_bias        TEXT,
                market_health   TEXT,
                rsi             REAL,
                volume_spike    INTEGER DEFAULT 0,
                sentiment       TEXT,
                reasoning       TEXT,
                fired_at        TEXT    NOT NULL,
                paper           INTEGER DEFAULT 0,
                sector          TEXT,
                avg_volume      REAL    DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS outcomes (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_id   INTEGER NOT NULL REFERENCES signals(id),
                checkpoint  TEXT    NOT NULL,   -- '1d' / '3d' / '7d'
                price       REAL,
                return_pct  REAL,
                win         INTEGER,            -- 1=win, 0=loss, NULL=pending
                checked_at  TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_signals_ticker ON signals(ticker);
            CREATE INDEX IF NOT EXISTS idx_signals_fired  ON signals(fired_at);
            CREATE INDEX IF NOT EXISTS idx_outcomes_sig   ON outcomes(signal_id);
        """)
        # Migrate existing databases — ADD COLUMN is idempotent via try/except
        for ddl in (
            "ALTER TABLE signals ADD COLUMN sector TEXT",
            "ALTER TABLE signals ADD COLUMN avg_volume REAL DEFAULT 0",
        ):
            try:
                conn.execute(ddl)
            except Exception:
                pass   # column already exists


# ── Record a new signal ───────────────────────────────────────────────────────

def record_signal(state: dict) -> int | None:
    """
    Call this after a real (non-paper) alert fires.
    Returns the new signal row id, or None if skipped.
    """
    signal = state.get("signal", "HOLD")
    if signal not in ("BUY", "SELL"):
        return None

    import world_context as wctx
    ctx   = wctx.get()
    macro = ctx["macro"]
    geo   = ctx["geo"]
    breadth = ctx["breadth"]

    targets  = state.get("targets", [])
    t1 = targets[0] if len(targets) > 0 else None
    t2 = targets[1] if len(targets) > 1 else None

    row = {
        "ticker":         state["ticker"],
        "signal":         signal,
        "price":          state.get("current_price", 0.0),
        "confidence":     int(state.get("confidence", 0)),
        "stop_loss":      state.get("stop_loss", 0.0),
        "target_1":       t1,
        "target_2":       t2,
        "trade_horizon":  state.get("trade_horizon", "swing"),
        "news_triggered": int(state.get("news_triggered", False)),
        "macro_regime":   macro.get("regime", ""),
        "macro_bias":     macro.get("bias", ""),
        "geo_bias":       geo.get("overall_bias", ""),
        "market_health":  breadth.get("health", ""),
        "rsi":            state.get("rsi", 0.0),
        "volume_spike":   int(state.get("volume_spike", False)),
        "sentiment":      state.get("news_sentiment", "NEUTRAL"),
        "reasoning":      state.get("reasoning", "")[:300],
        "fired_at":       datetime.now().isoformat(),
        "paper":          int(state.get("paper_trading", False)),
        "sector":         state.get("sector", ""),
        "avg_volume":     state.get("avg_volume", 0.0),
    }

    with _get_conn() as conn:
        cur = conn.execute("""
            INSERT INTO signals
              (ticker, signal, price, confidence, stop_loss, target_1, target_2,
               trade_horizon, news_triggered, macro_regime, macro_bias, geo_bias,
               market_health, rsi, volume_spike, sentiment, reasoning, fired_at, paper,
               sector, avg_volume)
            VALUES
              (:ticker, :signal, :price, :confidence, :stop_loss, :target_1, :target_2,
               :trade_horizon, :news_triggered, :macro_regime, :macro_bias, :geo_bias,
               :market_health, :rsi, :volume_spike, :sentiment, :reasoning, :fired_at, :paper,
               :sector, :avg_volume)
        """, row)
        sig_id = cur.lastrowid

        # Pre-create pending outcome rows for each checkpoint
        for cp in ("1d", "3d", "7d"):
            conn.execute("""
                INSERT INTO outcomes (signal_id, checkpoint)
                VALUES (?, ?)
            """, (sig_id, cp))

    print(f"📊 [Tracker] Logged {signal} #{sig_id} — {row['ticker']} @ ${row['price']:.4f}  conf={row['confidence']}")
    return sig_id


# ── Fill in outcomes ──────────────────────────────────────────────────────────

_CP_DAYS = {"1d": 1, "3d": 3, "7d": 7}


def _fill_outcomes():
    """Check all pending outcome rows and fill them if enough time has passed."""
    with _get_conn() as conn:
        pending = conn.execute("""
            SELECT o.id, o.signal_id, o.checkpoint,
                   s.ticker, s.signal, s.price, s.fired_at
            FROM outcomes o
            JOIN signals s ON s.id = o.signal_id
            WHERE o.win IS NULL
            ORDER BY s.fired_at
        """).fetchall()

    if not pending:
        return

    # Group by ticker to batch yfinance calls
    by_ticker: dict[str, list] = {}
    for row in pending:
        by_ticker.setdefault(row["ticker"], []).append(row)

    for ticker, rows in by_ticker.items():
        try:
            prices = _fetch_close_prices(ticker)
        except Exception as e:
            print(f"⚠️  [Tracker] Price fetch failed for {ticker}: {e}")
            continue

        for row in rows:
            fired_dt  = datetime.fromisoformat(row["fired_at"])
            target_dt = fired_dt + timedelta(days=_CP_DAYS[row["checkpoint"]])

            if datetime.now() < target_dt:
                continue   # not ready yet

            # Find closest price to target date
            price_at = _price_at(prices, target_dt)
            if price_at is None:
                continue

            entry    = row["price"]
            ret_pct  = (price_at - entry) / entry * 100
            # For BUY: positive return = win. For SELL: negative return = win.
            win = int(ret_pct > 0) if row["signal"] == "BUY" else int(ret_pct < 0)

            with _get_conn() as conn:
                conn.execute("""
                    UPDATE outcomes
                    SET price=?, return_pct=?, win=?, checked_at=?
                    WHERE id=?
                """, (price_at, round(ret_pct, 3), win, datetime.now().isoformat(), row["id"]))

            icon = "✅" if win else "❌"
            print(f"📊 [Tracker] {icon} {ticker} {row['checkpoint']} "
                  f"entry=${entry:.2f} → ${price_at:.2f} ({ret_pct:+.1f}%)  [{row['signal']}]")


def _fetch_close_prices(ticker: str) -> dict:
    """Returns {date_str: close_price} for last 15 trading days."""
    t = yf.Ticker(ticker)
    df = t.history(period="15d", interval="1d", auto_adjust=True)
    result = {}
    if df is None or df.empty:
        return result
    for dt, row in df.iterrows():
        try:
            date_key = dt.strftime("%Y-%m-%d")
            result[date_key] = float(row["Close"])
        except Exception:
            continue
    return result


def _price_at(prices: dict, target_dt: datetime) -> float | None:
    """Find the closest trading day price on or after target_dt."""
    for offset in range(5):   # look up to 5 days forward (weekends, holidays)
        key = (target_dt + timedelta(days=offset)).strftime("%Y-%m-%d")
        if key in prices:
            return prices[key]
    return None


# ── Statistics for reflection agent ──────────────────────────────────────────

def get_stats(lookback_days: int = 30, paper: bool = False) -> dict:
    """
    Returns win rates, average returns, and best/worst setup conditions
    for signals fired in the last `lookback_days`.
    """
    since = (datetime.now() - timedelta(days=lookback_days)).isoformat()

    with _get_conn() as conn:
        rows = conn.execute("""
            SELECT s.ticker, s.signal, s.confidence, s.macro_regime, s.macro_bias,
                   s.geo_bias, s.market_health, s.rsi, s.volume_spike, s.sentiment,
                   s.trade_horizon, s.news_triggered, s.fired_at,
                   o.checkpoint, o.return_pct, o.win
            FROM signals s
            JOIN outcomes o ON o.signal_id = s.id
            WHERE s.fired_at >= ?
              AND s.paper = ?
              AND o.win IS NOT NULL
            ORDER BY s.fired_at DESC
        """, (since, int(paper))).fetchall()

    if not rows:
        return {"total": 0, "message": "No completed signals yet"}

    total    = len(rows)
    wins     = sum(1 for r in rows if r["win"])
    win_rate = round(wins / total * 100, 1)
    avg_ret  = round(sum(r["return_pct"] for r in rows) / total, 2)

    # Break down by checkpoint
    cp_stats = {}
    for cp in ("1d", "3d", "7d"):
        cp_rows = [r for r in rows if r["checkpoint"] == cp]
        if cp_rows:
            cp_wins = sum(1 for r in cp_rows if r["win"])
            cp_stats[cp] = {
                "n":        len(cp_rows),
                "win_rate": round(cp_wins / len(cp_rows) * 100, 1),
                "avg_ret":  round(sum(r["return_pct"] for r in cp_rows) / len(cp_rows), 2),
            }

    # Best conditions (win rate by macro regime)
    regime_stats: dict = {}
    for r in rows:
        regime = r["macro_regime"] or "UNKNOWN"
        regime_stats.setdefault(regime, {"wins": 0, "total": 0})
        regime_stats[regime]["total"] += 1
        if r["win"]:
            regime_stats[regime]["wins"] += 1
    for k in regime_stats:
        d = regime_stats[k]
        d["win_rate"] = round(d["wins"] / d["total"] * 100, 1)

    # Worst losses / best wins
    sorted_ret = sorted(rows, key=lambda r: r["return_pct"])
    worst = [{"ticker": r["ticker"], "ret": r["return_pct"], "cp": r["checkpoint"]}
             for r in sorted_ret[:3]]
    best  = [{"ticker": r["ticker"], "ret": r["return_pct"], "cp": r["checkpoint"]}
             for r in reversed(sorted_ret[-3:])]

    return {
        "total":        total,
        "wins":         wins,
        "win_rate":     win_rate,
        "avg_return":   avg_ret,
        "by_checkpoint": cp_stats,
        "by_regime":    regime_stats,
        "best_trades":  best,
        "worst_trades": worst,
        "lookback_days": lookback_days,
    }


# ── Open signals (for portfolio agent) ───────────────────────────────────────

def get_open_signals(paper: bool = False) -> list[dict]:
    """
    Returns BUY signals fired in the last 14 days that have no 7d outcome yet
    — i.e. positions that are still potentially open.
    """
    since = (datetime.now() - timedelta(days=14)).isoformat()
    with _get_conn() as conn:
        rows = conn.execute("""
            SELECT s.id, s.ticker, s.signal, s.price, s.confidence,
                   s.stop_loss, s.target_1, s.target_2, s.fired_at, s.trade_horizon,
                   s.macro_regime, s.reasoning
            FROM signals s
            WHERE s.signal = 'BUY'
              AND s.fired_at >= ?
              AND s.paper = ?
              AND s.id NOT IN (
                  SELECT signal_id FROM outcomes WHERE checkpoint='7d' AND win IS NOT NULL
              )
            ORDER BY s.fired_at DESC
        """, (since, int(paper))).fetchall()
    return [dict(r) for r in rows]


# ── Background loop ───────────────────────────────────────────────────────────

async def performance_tracker_loop():
    """Async loop: fills in price outcomes every hour."""
    print(f"📊 [Tracker] Started — checking outcomes every {CHECK_INTERVAL//60}min")
    init_db()

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _fill_outcomes)

    while True:
        try:
            await asyncio.sleep(CHECK_INTERVAL)
            await loop.run_in_executor(None, _fill_outcomes)
        except asyncio.CancelledError:
            print("📊 [Tracker] Stopped.")
            break
        except Exception as e:
            print(f"❌ [Tracker] Loop error: {e}")
            await asyncio.sleep(300)


# ── Standalone test / CLI ─────────────────────────────────────────────────────

if __name__ == "__main__":
    init_db()
    print("=== Performance Tracker ===\n")

    # Insert a fake historical signal to test outcome filling
    import world_context as wctx
    fake_state = {
        "ticker":        "AAPL",
        "signal":        "BUY",
        "current_price": 170.0,
        "confidence":    75,
        "stop_loss":     163.0,
        "targets":       [180.0, 195.0],
        "trade_horizon": "swing",
        "news_triggered": False,
        "rsi":           52.0,
        "volume_spike":  True,
        "news_sentiment": "BULLISH",
        "reasoning":     "Test signal",
        "paper_trading": False,
    }
    sig_id = record_signal(fake_state)
    print(f"Inserted test signal id={sig_id}")

    print("\nFilling outcomes (will skip if not enough time has passed)...")
    _fill_outcomes()

    print("\nStats:")
    stats = get_stats(lookback_days=90)
    import json
    print(json.dumps(stats, indent=2))

    print("\nOpen signals:")
    open_sigs = get_open_signals()
    for s in open_sigs:
        print(f"  {s['ticker']} BUY @ ${s['price']:.2f}  stop=${s['stop_loss']:.2f}  {s['fired_at'][:10]}")
