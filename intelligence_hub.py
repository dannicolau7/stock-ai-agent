"""
intelligence_hub.py — Central nervous system singleton.

IntelligenceHub connects all agents bidirectionally by providing:
  - Reflection weights (learned from historical trade outcomes)
  - Regime-adaptive thresholds (BULL/BEAR/FEAR/PANIC calibrated)
  - Portfolio context (open positions, sector exposure, daily P&L)
  - Tomorrow's watchlist setup (from eod_scanner)
  - Alert deduplication (was_alerted_today / mark_alerted)
  - Daily state reset (hub.reset_daily())

Usage:
    from intelligence_hub import hub
    weights    = hub.get_reflection_weights()
    thresholds = hub.get_regime_thresholds()
    hub.mark_alerted("BZAI", "BUY")
"""

import json
import os
import threading
from datetime import datetime, date
from typing import Optional

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)


class IntelligenceHub:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    FILES = {
        "reflection_weights": os.path.join(DATA_DIR, "learnings.json"),
        "portfolio_state":    os.path.join(DATA_DIR, "portfolio_state.json"),
        "alerted_today":      os.path.join(DATA_DIR, "alerted_today.json"),
        "tomorrow_watchlist": os.path.join(DATA_DIR, "tomorrow_watchlist.json"),
        "prep_alert_list":    os.path.join(DATA_DIR, "prep_alert_list.json"),
        "hub_state":          os.path.join(DATA_DIR, "hub_state.json"),
    }

    DEFAULT_WEIGHTS = {
        # Hub signal keys
        "edgar":          1.0,
        "rsi_oversold":   1.0,
        "macd":           1.0,
        "volume_spike":   1.0,
        "news_sentiment": 1.0,
        "reddit":         1.0,
        "ema_cross":      1.0,
        "vwap":           1.0,
        "fib_support":    1.0,
        "sr_level":       1.0,
        "stoch_rsi":      1.0,
        "obv":            1.0,
        "gap_up":         1.0,
        "pattern":        1.0,
        # self_learner.py aliases (kept in sync)
        "volume":         1.0,
        "rsi_bounce":     1.0,
        "rsi_momentum":   1.0,
        "ema_stack":      1.0,
        "bollinger":      1.0,
        "float_rot":      1.0,
        "sentiment":      1.0,
        "smart_money":    1.0,
        "support":        1.0,
        "gap":            1.0,
    }

    _REGIME_THRESHOLDS = {
        "BULL": {
            "rsi_oversold":     45,
            "rsi_overbought":   75,
            "volume_spike_min": 1.5,
            "confidence_min":   60,
            "confidence_cap":   100,
            "agreement_min":    50,
        },
        "BEAR": {
            "rsi_oversold":     32,
            "rsi_overbought":   65,
            "volume_spike_min": 2.5,
            "confidence_min":   70,
            "confidence_cap":   80,
            "agreement_min":    65,
        },
        "FEAR": {
            "rsi_oversold":     28,
            "rsi_overbought":   60,
            "volume_spike_min": 3.0,
            "confidence_min":   75,
            "confidence_cap":   70,
            "agreement_min":    70,
        },
        "PANIC": {
            "rsi_oversold":     20,
            "rsi_overbought":   55,
            "volume_spike_min": 4.0,
            "confidence_min":   80,
            "confidence_cap":   65,
            "agreement_min":    75,
        },
    }

    _DEFAULT_THRESHOLDS = {
        "rsi_oversold":     38,
        "rsi_overbought":   70,
        "volume_spike_min": 2.0,
        "confidence_min":   65,
        "confidence_cap":   100,
        "agreement_min":    55,
    }

    # ── File I/O ────────────────────────────────────────────────────────────────

    def _load(self, key: str, default=None):
        path = self.FILES.get(key)
        if not path or not os.path.exists(path):
            return default if default is not None else {}
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return default if default is not None else {}

    def _save(self, key: str, data):
        path = self.FILES.get(key)
        if not path:
            return
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"⚠️  [Hub] Could not save {key}: {e}")

    # ── Reflection weights ──────────────────────────────────────────────────────

    def get_reflection_weights(self) -> dict:
        """
        Returns per-signal weight multipliers from learnings.json.
        Falls back to DEFAULT_WEIGHTS if no learnings exist yet.
        """
        learnings = self._load("reflection_weights", {})
        sig_weights = learnings.get("signal_weights", {})
        if sig_weights and isinstance(sig_weights, dict):
            merged = dict(self.DEFAULT_WEIGHTS)
            merged.update(sig_weights)
            return merged
        return dict(self.DEFAULT_WEIGHTS)

    def update_reflection_weights(self, new_weights: dict):
        """
        Blend new per-signal weights into existing learnings using EMA:
            updated = old * 0.8 + new * 0.2
        Prevents oscillation from a single day's noisy results.
        """
        if not new_weights:
            return
        learnings = self._load("reflection_weights", {})
        existing = learnings.get("signal_weights", {})
        blended = dict(self.DEFAULT_WEIGHTS)
        for sig, new_val in new_weights.items():
            old_val = existing.get(sig, 1.0)
            blended[sig] = round(old_val * 0.8 + float(new_val) * 0.2, 4)
        learnings["signal_weights"] = blended
        learnings["weights_updated_at"] = datetime.now().isoformat()
        self._save("reflection_weights", learnings)
        print(f"🧠 [Hub] Reflection weights updated (EMA blend) for {len(new_weights)} signals")

    # ── Regime thresholds ───────────────────────────────────────────────────────

    def get_regime_thresholds(self) -> dict:
        """
        Returns regime-adaptive thresholds based on current macro regime from world_context.
        Falls back to NORMAL defaults if world_context is unavailable.
        """
        try:
            import world_context as wctx
            regime = wctx.get()["macro"].get("regime", "NORMAL")
        except Exception:
            regime = "NORMAL"
        thresholds = self._REGIME_THRESHOLDS.get(regime, self._DEFAULT_THRESHOLDS)
        return {**thresholds, "regime": regime}

    # ── Portfolio context ───────────────────────────────────────────────────────

    def get_portfolio_context(self, ticker: str, paper: bool = False) -> dict:
        """
        Returns portfolio context for a ticker:
          - already_open: bool (ticker already has an open position)
          - open_count: int
          - open_tickers: list[str]
        Paper mode always returns empty context to prevent contamination with live positions.
        """
        _default = {"already_open": False, "open_count": 0, "open_tickers": []}
        if paper:
            return _default
        try:
            import performance_tracker as pt
            open_signals = pt.get_open_signals(paper=False)
            tickers_open = [s.get("ticker", "") for s in open_signals]
            already_open = ticker in tickers_open
            return {
                "already_open": already_open,
                "open_count":   len(open_signals),
                "open_tickers": tickers_open,
            }
        except Exception:
            return _default

    # ── Tomorrow watchlist ──────────────────────────────────────────────────────

    def get_tomorrow_setup(self, ticker: str) -> Optional[dict]:
        """
        Returns the EOD scanner's setup dict for this ticker if it's in tomorrow's watchlist.
        Returns None if ticker is not pre-identified.
        """
        data = self._load("tomorrow_watchlist", {})
        for setup in data.get("setups", []):
            if setup.get("ticker") == ticker:
                return setup
        return None

    # ── Alert deduplication ─────────────────────────────────────────────────────

    def was_alerted_today(self, ticker: str, cooldown_hours: float = 4.0) -> bool:
        """
        Returns True if ticker was already alerted today within cooldown_hours.
        Prevents spamming the same ticker multiple times per day.
        """
        alerted = self._load("alerted_today", {})
        today   = date.today().isoformat()
        entry   = alerted.get(ticker)
        if not entry or entry.get("date") != today:
            return False
        alerted_at = entry.get("alerted_at", "")
        if not alerted_at:
            return False
        try:
            dt        = datetime.fromisoformat(alerted_at)
            hours_ago = (datetime.now() - dt).total_seconds() / 3600
            return hours_ago < cooldown_hours
        except Exception:
            return False

    def mark_alerted(self, ticker: str, signal: str):
        """Record that ticker was alerted now. Called after a BUY/SELL alert fires."""
        alerted = self._load("alerted_today", {})
        alerted[ticker] = {
            "signal":     signal,
            "date":       date.today().isoformat(),
            "alerted_at": datetime.now().isoformat(),
        }
        self._save("alerted_today", alerted)

    # ── Daily reset ─────────────────────────────────────────────────────────────

    def reset_daily(self):
        """
        Called at midnight to reset per-day state.
        Clears alerted_today registry so each ticker can fire once per day again.
        """
        self._save("alerted_today", {})
        hub_state = self._load("hub_state", {})
        hub_state["last_reset"] = datetime.now().isoformat()
        self._save("hub_state", hub_state)
        print("🔄 [Hub] Daily state reset — alerted_today cleared")

    # ── Generic key-value store ─────────────────────────────────────────────────

    def set(self, key: str, value):
        """Store a value in hub_state.json."""
        hub_state = self._load("hub_state", {})
        hub_state[key] = value
        hub_state["updated_at"] = datetime.now().isoformat()
        self._save("hub_state", hub_state)

    def get(self, key: str, default=None):
        """Retrieve a value from hub_state.json."""
        return self._load("hub_state", {}).get(key, default)

    def __repr__(self):
        thresholds = self.get_regime_thresholds()
        weights    = self.get_reflection_weights()
        boosted    = [k for k, v in weights.items() if v > 1.0]
        reduced    = [k for k, v in weights.items() if v < 1.0]
        return (
            f"<IntelligenceHub regime={thresholds.get('regime','?')}  "
            f"conf_cap={thresholds.get('confidence_cap','?')}  "
            f"agreement_min={thresholds.get('agreement_min','?')}  "
            f"boosted={boosted}  reduced={reduced}>"
        )


# Module-level singleton — all modules import this one object
hub = IntelligenceHub()


if __name__ == "__main__":
    print("=== IntelligenceHub Self-Test ===\n")
    print(f"Hub: {hub}\n")

    w = hub.get_reflection_weights()
    print(f"Reflection weights ({len(w)} signals):")
    for sig, val in sorted(w.items()):
        if val != 1.0:
            print(f"  {sig}: ×{val}")
    if all(v == 1.0 for v in w.values()):
        print("  (all default 1.0 — no reflection data yet)")

    print(f"\nRegime thresholds: {hub.get_regime_thresholds()}")

    port = hub.get_portfolio_context("BZAI")
    print(f"\nPortfolio context for BZAI: {port}")

    setup = hub.get_tomorrow_setup("BZAI")
    print(f"Tomorrow setup for BZAI: {setup}")

    print(f"\nwas_alerted_today('BZAI'): {hub.was_alerted_today('BZAI')}")
    hub.mark_alerted("BZAI", "BUY")
    print(f"After mark_alerted:        {hub.was_alerted_today('BZAI')}")
    hub.reset_daily()
    print(f"After reset_daily:         {hub.was_alerted_today('BZAI')}")
