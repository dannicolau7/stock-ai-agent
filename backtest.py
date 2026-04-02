#!/usr/bin/env python3
"""
backtest.py — Rule-based backtest using the same RSI / MACD / Bollinger / volume
logic as the live agent. No Claude API calls — pure indicator scoring.

Usage:
  python3 backtest.py --ticker BZAI --days 30
  python3 backtest.py --ticker AAPL --days 90 --forward 5
"""

import argparse
from datetime import datetime

import numpy as np

from polygon_feed import get_daily_bars


# ── Indicator helpers (mirrors tech_agent.py exactly) ─────────────────────────

def _ema(arr: np.ndarray, period: int) -> np.ndarray:
    k = 2 / (period + 1)
    out = [float(arr[0])]
    for v in arr[1:]:
        out.append(float(v) * k + out[-1] * (1 - k))
    return np.array(out)


def _rsi(closes: np.ndarray, period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    deltas = np.diff(closes)
    gains  = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    ag = float(np.mean(gains[:period]))
    al = float(np.mean(losses[:period]))
    for i in range(period, len(gains)):
        ag = (ag * (period - 1) + gains[i]) / period
        al = (al * (period - 1) + losses[i]) / period
    if al == 0:
        return 100.0
    return round(100 - (100 / (1 + ag / al)), 2)


def _macd_hist(closes: np.ndarray, fast: int = 12, slow: int = 26, sig: int = 9) -> float:
    if len(closes) < slow + sig:
        return 0.0
    macd_line   = _ema(closes, fast) - _ema(closes, slow)
    signal_line = _ema(macd_line, sig)
    return float((macd_line - signal_line)[-1])


def _bollinger(closes: np.ndarray, period: int = 20) -> dict:
    if len(closes) < period:
        last = float(closes[-1])
        return {"upper": last, "middle": last, "lower": last}
    recent = closes[-period:]
    mid    = float(np.mean(recent))
    std    = float(np.std(recent, ddof=0))
    return {"upper": mid + 2 * std, "middle": mid, "lower": mid - 2 * std}


def _avg_vol(bars: list, period: int = 20) -> float:
    vols = [float(b.get("v", 0)) for b in bars[-period:] if b.get("v")]
    return sum(vols) / len(vols) if vols else 0.0


def _atr(bars: list, period: int = 14) -> float:
    if len(bars) < 2:
        return 0.0
    trs = [
        max(
            bars[i]["h"] - bars[i]["l"],
            abs(bars[i]["h"] - bars[i - 1]["c"]),
            abs(bars[i]["l"] - bars[i - 1]["c"]),
        )
        for i in range(1, len(bars))
    ]
    recent = trs[-period:]
    return sum(recent) / len(recent) if recent else 0.0


# ── Signal scoring ─────────────────────────────────────────────────────────────

def _score(bars_history: list) -> tuple[str, int]:
    """
    Returns (signal, score) using the same confluence approach as the live agent.
    BUY / SELL fires when score >= 45.
    """
    closes     = np.array([float(b["c"]) for b in bars_history], dtype=float)
    price      = closes[-1]
    prev_close = closes[-2] if len(closes) > 1 else price

    rsi_val    = _rsi(closes)
    hist_curr  = _macd_hist(closes)
    hist_prev  = _macd_hist(closes[:-1]) if len(closes) > 1 else 0.0
    bb         = _bollinger(closes)
    avg_v      = _avg_vol(bars_history)
    curr_v     = float(bars_history[-1].get("v", 0))
    vol_spike  = (curr_v > 2 * avg_v) if avg_v > 0 else False

    # ── Bull signals ──────────────────────────────────────────────────────────
    bull = 0
    if rsi_val < 30:                              bull += 30   # oversold
    elif rsi_val < 40:                            bull += 15
    if hist_curr > 0 and hist_prev <= 0:          bull += 25   # MACD bullish cross
    elif hist_curr > 0:                           bull += 10   # MACD bullish drift
    if price < bb["lower"]:                       bull += 20   # below BB lower
    if vol_spike and price >= prev_close:         bull += 15   # vol spike + up move

    # ── Bear signals ──────────────────────────────────────────────────────────
    bear = 0
    if rsi_val > 70:                              bear += 30   # overbought
    elif rsi_val > 60:                            bear += 15
    if hist_curr < 0 and hist_prev >= 0:          bear += 25   # MACD bearish cross
    elif hist_curr < 0:                           bear += 10   # MACD bearish drift
    if price > bb["upper"]:                       bear += 20   # above BB upper
    if vol_spike and price < prev_close:          bear += 15   # vol spike + down move

    THRESHOLD = 45
    if bull >= THRESHOLD and bull > bear:
        return "BUY",  bull
    if bear >= THRESHOLD and bear > bull:
        return "SELL", bear
    return "HOLD", max(bull, bear)


# ── Outcome evaluation ─────────────────────────────────────────────────────────

def _evaluate(
    signal: str,
    entry: float,
    future_bars: list,
    atr_val: float,
) -> tuple[bool | None, float, str]:
    """
    Simulates trade outcome using stop = 1.5×ATR, target = 2×ATR.
    Returns (win, pct_return, exit_reason).
    """
    if not future_bars:
        return None, 0.0, "NO_DATA"

    stop_dist   = atr_val * 1.5 if atr_val else entry * 0.05
    target_dist = atr_val * 2.0 if atr_val else entry * 0.05

    for bar in future_bars:
        hi = float(bar["h"])
        lo = float(bar["l"])

        if signal == "BUY":
            if lo <= entry - stop_dist:
                return False, -(stop_dist / entry) * 100, "STOP_LOSS"
            if hi >= entry + target_dist:
                return True,  (target_dist / entry) * 100, "TARGET_HIT"
        else:  # SELL
            if hi >= entry + stop_dist:
                return False, -(stop_dist / entry) * 100, "STOP_LOSS"
            if lo <= entry - target_dist:
                return True,  (target_dist / entry) * 100, "TARGET_HIT"

    # Neither stop nor target hit — use final close
    final = float(future_bars[-1]["c"])
    if signal == "BUY":
        pct = (final - entry) / entry * 100
    else:
        pct = (entry - final) / entry * 100
    return pct > 0, pct, "TIMEOUT"


# ── Main backtest runner ───────────────────────────────────────────────────────

def run_backtest(ticker: str, days: int, forward: int = 5) -> None:
    WARMUP = 65   # bars needed before indicators are reliable

    print(f"\n📈  Backtesting {ticker}  |  {days}-day window  |  {forward}-bar forward look")
    print("─" * 62)
    print(f"🔄  Fetching {days + WARMUP + forward} days of historical data from Polygon...")

    try:
        all_bars = get_daily_bars(ticker, days=days + WARMUP + forward)
    except Exception as e:
        print(f"❌  Data fetch failed: {e}")
        return

    if len(all_bars) < WARMUP + forward + 5:
        print(f"❌  Not enough data ({len(all_bars)} bars). Try fewer --days.")
        return

    print(f"✅  {len(all_bars)} bars fetched\n")

    # Index range: test on the last `days` bars, leaving `forward` bars for outcome eval
    test_start = max(WARMUP, len(all_bars) - days - forward)
    test_end   = len(all_bars) - forward

    signals: list[dict] = []
    last_signal_idx = -99

    for i in range(test_start, test_end):
        history = all_bars[: i + 1]
        signal, score = _score(history)

        if signal == "HOLD":
            continue
        if i - last_signal_idx < 3:        # 3-bar cooldown to avoid clusters
            continue

        entry        = float(all_bars[i]["c"])
        future_bars  = all_bars[i + 1 : i + 1 + forward]
        atr_val      = _atr(history)
        win, pct, reason = _evaluate(signal, entry, future_bars, atr_val)

        t     = all_bars[i].get("t", 0)
        date  = datetime.fromtimestamp(t / 1000).strftime("%Y-%m-%d") if t else f"idx[{i}]"

        signals.append({
            "date":   date,
            "signal": signal,
            "score":  score,
            "entry":  entry,
            "win":    win,
            "pct":    round(pct, 3),
            "reason": reason,
        })
        last_signal_idx = i

    # ── Signal table ───────────────────────────────────────────────────────────
    if not signals:
        print("⚠️   No signals fired during this period.")
        return

    hdr = f"{'Date':<12} {'Sig':<5} {'Score':<7} {'Entry':>9}  {'Result':<22} {'P&L':>7}"
    print(hdr)
    print("─" * 62)
    for s in signals:
        icon    = "🟢" if s["signal"] == "BUY" else "🔴"
        outcome = ("✅ WIN" if s["win"] else "❌ LOSS") + f" ({s['reason']})"
        print(
            f"{s['date']:<12} {icon}{s['signal']:<4} {s['score']:<7} "
            f"${s['entry']:>8.4f}  {outcome:<22} {s['pct']:>+6.2f}%"
        )

    # ── Summary stats ──────────────────────────────────────────────────────────
    valid   = [s for s in signals if s["win"] is not None]
    wins    = [s for s in valid if s["win"]]
    losses  = [s for s in valid if not s["win"]]
    buys    = [s for s in valid if s["signal"] == "BUY"]
    sells   = [s for s in valid if s["signal"] == "SELL"]

    win_rate  = len(wins)  / len(valid) * 100 if valid else 0.0
    avg_gain  = sum(s["pct"] for s in wins)   / len(wins)   if wins   else 0.0
    avg_loss  = sum(s["pct"] for s in losses) / len(losses) if losses else 0.0
    total_pnl = sum(s["pct"] for s in valid)

    stops   = sum(1 for s in valid if s["reason"] == "STOP_LOSS")
    targets = sum(1 for s in valid if s["reason"] == "TARGET_HIT")
    timeouts= sum(1 for s in valid if s["reason"] == "TIMEOUT")

    print("\n" + "═" * 62)
    print(f"{'  BACKTEST RESULTS — ' + ticker + '  ':═^62}")
    print("═" * 62)
    print(f"  Ticker          {ticker}")
    print(f"  Period          {days} days tested  ({forward}-bar forward window)")
    print(f"  Total signals   {len(valid)}   (🟢 BUY: {len(buys)}  🔴 SELL: {len(sells)})")
    print(f"  Win rate        {win_rate:.1f}%   ({len(wins)} wins / {len(losses)} losses)")
    print(f"  Avg gain        +{avg_gain:.2f}%")
    print(f"  Avg loss        {avg_loss:.2f}%")
    print(f"  Total P&L       {total_pnl:+.2f}%")
    print(f"  Stop hits       {stops}")
    print(f"  Target hits     {targets}")
    print(f"  Timeouts        {timeouts}")
    print("═" * 62 + "\n")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Stock AI Agent — Backtester")
    p.add_argument("--ticker",  required=True,        help="Ticker symbol  e.g. BZAI")
    p.add_argument("--days",    type=int, default=30,  help="Days to backtest (default 30)")
    p.add_argument("--forward", type=int, default=5,   help="Bars to look forward for outcome (default 5)")
    args = p.parse_args()
    run_backtest(args.ticker.upper(), args.days, args.forward)
