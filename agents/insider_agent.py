"""
agents/insider_agent.py — SEC EDGAR Form 4 insider trading signal.

Uses the free SEC EDGAR API (no API key, just User-Agent required).

get_signal(ticker) → dict:
  insider_signal   str    "BULLISH" | "BEARISH" | "NEUTRAL"
  insider_score    int    signed, negative = bearish; |score| ≥ 25 = signal
  transactions     list   parsed Form 4 transactions (last 30 days)
  key_alerts       list   human-readable alert strings
  _source          str    "edgar" | "cache" | "no_filings" | "error"

Scoring rules:
  CEO/CFO buy  > $500K                     → +25 (bullish)
  CEO/CFO sell > 20% of holdings           → -30 (bearish)
  ≥ 2 insiders selling same calendar week  → -40 (bearish)
  Option exercise + same-day sale (dump)   → -15 (bearish)

The LWLG example: CFO sold 26% at $10.36 → score -30, BEARISH signal.
"""

import threading
import time as _time
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta
from pathlib import Path

import requests

_HEADERS  = {"User-Agent": "ArgusStockAgent research@argus.local"}
_BASE_SEC = "https://www.sec.gov"
_BASE_DATA = "https://data.sec.gov"
_RATE_SLEEP = 0.12    # seconds between EDGAR requests (max 10/s; stay conservative)

# Signal threshold
SIGNAL_THRESHOLD = 25   # |score| >= this → BULLISH or BEARISH
LOOKBACK_DAYS    = 30
CACHE_TTL_S      = 6 * 3600   # 6 hours

# Scoring weights
SCORE_CEO_BUY       = +25
SCORE_CEO_SELL      = -30
SCORE_CLUSTER_SELL  = -40
SCORE_DUMP          = -15    # option exercise + immediate sale
MIN_CEO_BUY_VALUE   = 500_000   # $500K
MIN_CEO_SELL_PCT    = 20.0      # 20% of holdings


# ── CIK mapping (cached in-process) ───────────────────────────────────────────

_cik_map: dict[str, int] | None = None
_cik_map_lock = threading.Lock()
_cik_map_loaded_at: float = 0.0
_CIK_MAP_TTL = 24 * 3600   # re-fetch daily


def _get_cik(ticker: str) -> int | None:
    """Return integer CIK for ticker, or None if not found."""
    global _cik_map, _cik_map_loaded_at
    with _cik_map_lock:
        age = _time.monotonic() - _cik_map_loaded_at
        if _cik_map is None or age > _CIK_MAP_TTL:
            try:
                r = requests.get(
                    f"{_BASE_SEC}/files/company_tickers.json",
                    headers=_HEADERS, timeout=15,
                )
                r.raise_for_status()
                raw = r.json()
                _cik_map = {
                    v["ticker"].upper(): int(v["cik_str"])
                    for v in raw.values()
                    if "ticker" in v and "cik_str" in v
                }
                _cik_map_loaded_at = _time.monotonic()
                print(f"📋 [InsiderAgent] CIK map loaded ({len(_cik_map)} tickers)")
            except Exception as e:
                print(f"⚠️  [InsiderAgent] CIK map fetch failed: {e}")
                if _cik_map is None:
                    return None
        return _cik_map.get(ticker.upper())


# ── EDGAR fetchers ─────────────────────────────────────────────────────────────

def _get(url: str, timeout: int = 15) -> requests.Response | None:
    """Rate-limited GET with EDGAR User-Agent."""
    try:
        _time.sleep(_RATE_SLEEP)
        r = requests.get(url, headers=_HEADERS, timeout=timeout)
        return r if r.status_code == 200 else None
    except Exception as e:
        print(f"⚠️  [InsiderAgent] GET {url[:60]}... failed: {e}")
        return None


def _fetch_form4_list(cik: int, since: date) -> list[dict]:
    """
    Return list of Form 4 filings since `since` from the submissions API.
    Each entry: {accn_nd, filing_date, primary_doc}
    """
    cik10 = str(cik).zfill(10)
    r = _get(f"{_BASE_DATA}/submissions/CIK{cik10}.json")
    if r is None:
        return []

    try:
        sub    = r.json()
        recent = sub.get("filings", {}).get("recent", {})
        forms  = recent.get("form", [])
        dates  = recent.get("filingDate", [])
        accns  = recent.get("accessionNumber", [])
        docs   = recent.get("primaryDocument", [])
    except Exception as e:
        print(f"⚠️  [InsiderAgent] submissions parse error: {e}")
        return []

    result = []
    for i, form in enumerate(forms):
        if form != "4":
            continue
        try:
            filing_date = date.fromisoformat(dates[i])
        except Exception:
            continue
        if filing_date < since:
            continue    # older than lookback window
        # primaryDocument may be "xslF345X06/form4.xml" — strip the prefix
        doc_name = Path(docs[i]).name
        accn_nd  = accns[i].replace("-", "")
        result.append({
            "accn_nd":     accn_nd,
            "filing_date": filing_date,
            "primary_doc": doc_name,
        })

    return result


def _fetch_form4_xml(cik: int, accn_nd: str, doc_name: str) -> str | None:
    """Fetch the raw Form 4 XML for one filing."""
    url = f"{_BASE_SEC}/Archives/edgar/data/{cik}/{accn_nd}/{doc_name}"
    r   = _get(url)
    return r.text if r else None


# ── XML parsing ────────────────────────────────────────────────────────────────

def _xval(node: ET.Element, path: str, default: str = "") -> str:
    el = node.find(path)
    return (el.text or "").strip() if el is not None else default


def _xfloat(node: ET.Element, path: str) -> float:
    v = _xval(node, path)
    try:
        return float(v) if v else 0.0
    except ValueError:
        return 0.0


def _role_label(owner_node: ET.Element) -> str:
    """Return a normalised role string: CEO, CFO, Director, etc."""
    rel  = owner_node.find("reportingOwnerRelationship")
    if rel is None:
        return "Unknown"
    title = _xval(rel, "officerTitle").upper()
    is_dir = _xval(rel, "isDirector").lower() in ("true", "1")
    is_off = _xval(rel, "isOfficer").lower() in ("true", "1")
    is_10pct = _xval(rel, "isTenPercentOwner").lower() in ("true", "1")

    if "CEO" in title:
        return "CEO"
    if "CFO" in title or "CHIEF FINANCIAL" in title:
        return "CFO"
    if "COO" in title or "CHIEF OPERATING" in title:
        return "COO"
    if "CTO" in title or "CHIEF TECHNOLOGY" in title:
        return "CTO"
    if "PRESIDENT" in title:
        return "President"
    if is_dir and not is_off:
        return "Director"
    if is_10pct:
        return "10% Owner"
    if is_off and title:
        # Truncate long officer titles
        return title[:30].title()
    return "Officer" if is_off else "Insider"


def _is_senior_exec(role: str) -> bool:
    return role in ("CEO", "CFO", "COO", "CTO", "President")


def _parse_form4_xml(xml_text: str, filing_date: date) -> list[dict]:
    """
    Parse Form 4 XML into a list of transaction dicts.
    Each dict has: insider_name, role, date, txn_type, code,
                   shares, price, value, shares_after, pct_held_sold,
                   is_derivative, is_senior_exec
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        print(f"⚠️  [InsiderAgent] XML parse error: {e}")
        return []

    txns: list[dict] = []

    for owner in root.findall("reportingOwner"):
        name = _xval(owner, "reportingOwnerId/rptOwnerName")
        role = _role_label(owner)
        exec_ = _is_senior_exec(role)

        # ── Non-derivative transactions (actual shares) ────────────────────
        for t in root.findall("nonDerivativeTable/nonDerivativeTransaction"):
            code    = _xval(t, "transactionCoding/transactionCode")
            ad_code = _xval(t, "transactionAmounts/transactionAcquiredDisposedCode/value")
            shares  = _xfloat(t, "transactionAmounts/transactionShares/value")
            price   = _xfloat(t, "transactionAmounts/transactionPricePerShare/value")
            after   = _xfloat(t, "postTransactionAmounts/sharesOwnedFollowingTransaction/value")
            txn_date_str = _xval(t, "transactionDate/value")
            try:
                txn_date = date.fromisoformat(txn_date_str) if txn_date_str else filing_date
            except ValueError:
                txn_date = filing_date

            if shares <= 0:
                continue

            # Transaction type classification
            if code == "P" or (code == "A" and ad_code == "A"):
                txn_type = "BUY"
            elif code in ("S", "D") or (ad_code == "D" and code not in ("F",)):
                txn_type = "SELL"
            elif code == "M" and ad_code == "A":
                txn_type = "VEST"          # RSU/option vesting
            elif code == "F":
                txn_type = "TAX_WITHHOLD"  # tax-related forced sell (less meaningful)
            elif code == "G":
                txn_type = "GIFT"
            else:
                txn_type = "OTHER"

            # % of total holdings sold/bought
            total_before = shares + after if ad_code == "D" else max(after - shares, 0)
            pct_affected = (shares / total_before * 100
                            if total_before > 0 else 0.0)

            txns.append({
                "insider_name":   name,
                "role":           role,
                "is_senior_exec": exec_,
                "date":           txn_date,
                "txn_type":       txn_type,
                "code":           code,
                "shares":         shares,
                "price":          price,
                "value":          round(shares * price, 2) if price > 0 else 0.0,
                "shares_after":   after,
                "pct_held":       round(pct_affected, 1),
                "is_derivative":  False,
            })

        # ── Derivative transactions (options, warrants, RSUs) ─────────────
        for t in root.findall("derivativeTable/derivativeTransaction"):
            code    = _xval(t, "transactionCoding/transactionCode")
            ad_code = _xval(t, "transactionAmounts/transactionAcquiredDisposedCode/value")
            shares  = _xfloat(t, "transactionAmounts/transactionShares/value")
            price   = _xfloat(t, "transactionAmounts/transactionPricePerShare/value")
            txn_date_str = _xval(t, "transactionDate/value")
            try:
                txn_date = date.fromisoformat(txn_date_str) if txn_date_str else filing_date
            except ValueError:
                txn_date = filing_date

            if shares <= 0:
                continue

            if code == "M" and ad_code == "D":
                txn_type = "OPTION_EXERCISE"
            elif code in ("S", "D"):
                txn_type = "SELL"
            else:
                txn_type = "OTHER"

            txns.append({
                "insider_name":   name,
                "role":           role,
                "is_senior_exec": exec_,
                "date":           txn_date,
                "txn_type":       txn_type,
                "code":           code,
                "shares":         shares,
                "price":          price,
                "value":          round(shares * price, 2) if price > 0 else 0.0,
                "shares_after":   0.0,
                "pct_held":       0.0,
                "is_derivative":  True,
            })

    return txns


# ── Scoring ────────────────────────────────────────────────────────────────────

def _week_key(d: date) -> str:
    """ISO year-week key for clustering check."""
    iso = d.isocalendar()
    return f"{iso[0]}-W{iso[1]:02d}"


def _score_transactions(
    transactions: list[dict],
) -> tuple[int, list[str]]:
    """
    Score insider activity. Returns (signed_score, alert_strings).
    Negative = bearish, positive = bullish.
    """
    score  = 0
    alerts: list[str] = []
    applied: set = set()   # dedup rule tags

    if not transactions:
        return 0, []

    # Only score BUY/SELL/OPTION_EXERCISE (ignore gifts, tax withholding, etc.)
    meaningful = [
        t for t in transactions
        if t["txn_type"] in ("BUY", "SELL", "OPTION_EXERCISE")
    ]

    # ── Rule 1: CEO/CFO buy > $500K → bullish ─────────────────────────────
    for t in meaningful:
        if (t["txn_type"] == "BUY"
                and t["is_senior_exec"]
                and t["value"] >= MIN_CEO_BUY_VALUE
                and f"ceo_buy_{t['insider_name']}" not in applied):
            score += SCORE_CEO_BUY
            applied.add(f"ceo_buy_{t['insider_name']}")
            alerts.append(
                f"🟢 {t['role']} {t['insider_name']} bought "
                f"{t['shares']:,.0f} shares "
                f"(${t['value']:,.0f}) on {t['date']}"
            )

    # ── Rule 2: CEO/CFO sell > 20% of holdings → bearish ──────────────────
    for t in meaningful:
        if (t["txn_type"] in ("SELL", "OPTION_EXERCISE")
                and t["is_senior_exec"]
                and t["pct_held"] >= MIN_CEO_SELL_PCT
                and f"ceo_sell_{t['insider_name']}" not in applied):
            score += SCORE_CEO_SELL
            applied.add(f"ceo_sell_{t['insider_name']}")
            alerts.append(
                f"🔴 {t['role']} {t['insider_name']} sold "
                f"{t['pct_held']:.0f}% of holdings "
                f"({t['shares']:,.0f} shares"
                + (f" @ ${t['price']:.2f}" if t["price"] > 0 else "")
                + f") on {t['date']}"
            )

    # ── Rule 3: ≥2 different insiders selling in same calendar week ────────
    week_sellers: dict[str, set] = {}   # week_key → set of insider names
    for t in meaningful:
        if t["txn_type"] in ("SELL", "OPTION_EXERCISE"):
            wk = _week_key(t["date"])
            week_sellers.setdefault(wk, set()).add(t["insider_name"])

    for wk, names in week_sellers.items():
        if len(names) >= 2 and "cluster_sell" not in applied:
            applied.add("cluster_sell")
            score += SCORE_CLUSTER_SELL
            alerts.append(
                f"🔴 Cluster sell: {len(names)} insiders sold "
                f"same week ({wk}): {', '.join(sorted(names))}"
            )

    # ── Rule 4: option exercise + same-day sale (exercise-and-dump) ────────
    # Group by insider + date, look for OPTION_EXERCISE and SELL on same day
    by_insider_date: dict[tuple, list] = {}
    for t in meaningful:
        key = (t["insider_name"], t["date"])
        by_insider_date.setdefault(key, []).append(t)

    for (name, dt), group in by_insider_date.items():
        has_exercise = any(t["txn_type"] == "OPTION_EXERCISE" for t in group)
        has_sell     = any(t["txn_type"] == "SELL" for t in group)
        if (has_exercise and has_sell
                and f"dump_{name}" not in applied):
            applied.add(f"dump_{name}")
            score += SCORE_DUMP
            total_val = sum(t["value"] for t in group if t["txn_type"] == "SELL")
            alerts.append(
                f"🔴 {name} exercised options AND sold same day ({dt})"
                + (f" — ${total_val:,.0f} realised" if total_val > 0 else "")
            )

    return score, alerts


# ── Signal builder ─────────────────────────────────────────────────────────────

def _build_signal(score: int, alerts: list[str], transactions: list[dict],
                  source: str) -> dict:
    if score >= SIGNAL_THRESHOLD:
        signal = "BULLISH"
    elif score <= -SIGNAL_THRESHOLD:
        signal = "BEARISH"
    else:
        signal = "NEUTRAL"

    return {
        "insider_signal":       signal,
        "insider_score":        score,
        "transactions":         transactions,
        "key_alerts":           alerts,
        "_source":              source,
        "_transaction_count":   len(transactions),
    }


# ── Per-ticker cache ───────────────────────────────────────────────────────────

_signal_cache: dict[str, dict]  = {}
_cache_times:  dict[str, float] = {}
_cache_lock    = threading.Lock()


def get_signal(ticker: str) -> dict:
    """
    Return insider trading signal for `ticker` from EDGAR Form 4 filings.
    Results are cached for 6 hours.

    Always returns a dict — never raises.
    """
    ticker = ticker.upper()
    _empty = _build_signal(0, [], [], "no_filings")

    # ── Cache check ───────────────────────────────────────────────────────
    with _cache_lock:
        age = _time.monotonic() - _cache_times.get(ticker, 0.0)
        if ticker in _signal_cache and age < CACHE_TTL_S:
            cached = dict(_signal_cache[ticker])
            cached["_source"] = "cache"
            return cached

    try:
        print(f"📋 [InsiderAgent] Fetching Form 4 filings for {ticker}...")

        # ── CIK lookup ────────────────────────────────────────────────────
        cik = _get_cik(ticker)
        if cik is None:
            print(f"   ℹ️  {ticker}: no CIK found (ETF or foreign)")
            return _build_signal(0, [], [], "no_cik")

        # ── Fetch Form 4 list ─────────────────────────────────────────────
        since   = date.today() - timedelta(days=LOOKBACK_DAYS)
        filings = _fetch_form4_list(cik, since)
        print(f"   {len(filings)} Form 4 filing(s) in last {LOOKBACK_DAYS} days")

        if not filings:
            result = _build_signal(0, [], [], "no_filings")
            with _cache_lock:
                _signal_cache[ticker] = result
                _cache_times[ticker]  = _time.monotonic()
            return result

        # ── Fetch + parse each Form 4 ────────────────────────────────────
        all_txns: list[dict] = []
        for filing in filings[:20]:    # cap at 20 filings per ticker
            xml_text = _fetch_form4_xml(cik, filing["accn_nd"], filing["primary_doc"])
            if not xml_text:
                continue
            txns = _parse_form4_xml(xml_text, filing["filing_date"])
            all_txns.extend(txns)

        print(f"   {len(all_txns)} total transactions parsed")

        # ── Score ─────────────────────────────────────────────────────────
        score, alerts = _score_transactions(all_txns)

        if alerts:
            for a in alerts:
                print(f"   {a}")
        else:
            print(f"   No significant insider activity detected (score={score})")

        result = _build_signal(score, alerts, all_txns, "edgar")

    except Exception as e:
        print(f"❌ [InsiderAgent] {ticker} error: {e}")
        result = _build_signal(0, [], [], "error")

    with _cache_lock:
        _signal_cache[ticker] = result
        _cache_times[ticker]  = _time.monotonic()

    return result


def invalidate_cache(ticker: str | None = None) -> None:
    """Clear cached results for one ticker or all."""
    with _cache_lock:
        if ticker is None:
            _signal_cache.clear()
            _cache_times.clear()
        else:
            _signal_cache.pop(ticker.upper(), None)
            _cache_times.pop(ticker.upper(), None)


# ── Standalone test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"

    print(f"\n=== InsiderAgent: {ticker} ===\n")
    result = get_signal(ticker)

    print(f"\nSignal:     {result['insider_signal']}")
    print(f"Score:      {result['insider_score']:+d}")
    print(f"Filings:    {result['_transaction_count']} transactions")
    print(f"Source:     {result['_source']}")

    if result["key_alerts"]:
        print("\nKey alerts:")
        for a in result["key_alerts"]:
            print(f"  {a}")
    else:
        print("\nNo significant insider activity in last 30 days.")

    if result["transactions"]:
        print(f"\nTransaction log ({len(result['transactions'])} entries):")
        buy_val  = sum(t["value"] for t in result["transactions"] if t["txn_type"] == "BUY")
        sell_val = sum(t["value"] for t in result["transactions"] if t["txn_type"] in ("SELL", "OPTION_EXERCISE"))
        print(f"  Total bought: ${buy_val:,.0f}")
        print(f"  Total sold:   ${sell_val:,.0f}")
        for t in result["transactions"][:10]:
            icon = "🟢" if t["txn_type"] == "BUY" else ("🔴" if t["txn_type"] in ("SELL","OPTION_EXERCISE") else "⚪")
            val  = f" ${t['value']:,.0f}" if t["value"] > 0 else ""
            pct  = f" ({t['pct_held']:.0f}% of holdings)" if t["pct_held"] > 0 else ""
            print(f"  {icon} {t['date']} {t['role']:12s} {t['txn_type']:15s} "
                  f"{t['shares']:8,.0f} shares{val}{pct}")
