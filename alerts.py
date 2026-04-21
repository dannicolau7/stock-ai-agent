import requests
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from twilio.rest import Client
import os
from dotenv import load_dotenv

load_dotenv()

TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')
WHATSAPP_FROM = 'whatsapp:+14155238886'
WHATSAPP_TO = f"whatsapp:{os.getenv('TWILIO_TO_NUMBER')}"
PUSHOVER_APP_TOKEN = os.getenv('PUSHOVER_APP_TOKEN')
PUSHOVER_USER_KEY = os.getenv('PUSHOVER_USER_KEY')

EMAIL_SMTP_HOST = os.getenv('EMAIL_SMTP_HOST', 'smtp.gmail.com')
EMAIL_SMTP_PORT = int(os.getenv('EMAIL_SMTP_PORT', '587'))
EMAIL_USER     = os.getenv('EMAIL_USER', '')
EMAIL_PASSWORD = os.getenv('EMAIL_PASSWORD', '')
EMAIL_TO       = os.getenv('EMAIL_TO', '')

def send_whatsapp(message: str):
    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        msg = client.messages.create(
            body=message,
            from_=WHATSAPP_FROM,
            to=WHATSAPP_TO
        )
        # Poll for final status (sandbox messages often go undelivered silently)
        import time
        time.sleep(3)
        updated = client.messages(msg.sid).fetch()
        status  = updated.status   # 'delivered', 'undelivered', 'failed', 'sent', 'queued'
        if status in ('delivered', 'sent', 'queued', 'read'):
            print(f'✅ WhatsApp sent: {msg.sid} [{status}]')
            return True
        else:
            print(
                f'❌ WhatsApp undelivered (status={status}). '
                f'Twilio sandbox may have expired — '
                f'send "join <keyword>" to +14155238886 on WhatsApp to re-activate.'
            )
            return False
    except Exception as e:
        print(f'❌ WhatsApp error: {e}')
        return False

def send_email(subject: str, body: str) -> bool:
    if not (EMAIL_USER and EMAIL_PASSWORD and EMAIL_TO):
        return False
    try:
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From']    = EMAIL_USER
        msg['To']      = EMAIL_TO
        # Plain text part
        msg.attach(MIMEText(body, 'plain'))
        # HTML part — monospace table for readability
        html_body = '<pre style="font-family:monospace;font-size:14px;">' + body + '</pre>'
        msg.attach(MIMEText(html_body, 'html'))

        ctx = ssl.create_default_context()
        with smtplib.SMTP(EMAIL_SMTP_HOST, EMAIL_SMTP_PORT) as server:
            server.ehlo()
            server.starttls(context=ctx)
            server.login(EMAIL_USER, EMAIL_PASSWORD)
            server.sendmail(EMAIL_USER, EMAIL_TO.split(','), msg.as_string())
        print(f'✅ Email sent → {EMAIL_TO}')
        return True
    except Exception as e:
        print(f'❌ Email error: {e}')
        return False


def send_push(title: str, message: str):
    try:
        r = requests.post('https://api.pushover.net/1/messages.json', data={
            'token': PUSHOVER_APP_TOKEN,
            'user': PUSHOVER_USER_KEY,
            'title': title,
            'message': message,
            'sound': 'cashregister'
        })
        if r.status_code == 200:
            print(f'✅ Push sent')
            return True
        else:
            print(f'❌ Push failed ({r.status_code}): {r.text}')
            return False
    except Exception as e:
        print(f'❌ Push error: {e}')
        return False

_SEP = '─' * 35


def send_alert(ticker: str, signal: str, price: float,
               entry_low: float, entry_high: float,
               targets: list, stop: float,
               reason: str, confidence: int,
               horizon: str = "swing", horizon_reason: str = "",
               # Aggregator / validator extras (all optional)
               agreement_score: float = None,
               signal_count_bull: int = 0,
               signal_count_bear: int = 0,
               top_3_signals: list = None,
               bullish_signals: list = None,
               bearish_signals: list = None,
               consensus: str = "",
               market_regime_str: str = "",
               sector_str: str = "",
               catalyst_str: str = "",
               main_risk: str = "",
               det_score: int = 0,
               bull_score: int = 0,
               bear_score: int = 0,
               bull_summary: str = "",
               bear_summary: str = "",
               ) -> bool:
    """
    Send a trade alert via WhatsApp, Pushover, and email.
    Includes aggregator summary when aggregator data is present.
    Returns True if any channel delivers.
    """
    emoji = 'BUY' if signal == 'BUY' else 'SELL' if signal == 'SELL' else 'HOLD'
    entry_mid = (entry_low + entry_high) / 2

    # Targets: only those above entry
    valid_targets = [t for t in (targets if isinstance(targets, list) else [targets])
                     if t > entry_mid]
    labels = ['T1', 'T2', 'T3']

    # Stop pct from current price
    ref_price = price if price > 0 else entry_mid
    stop_pct  = (stop - ref_price) / ref_price * 100 if ref_price > 0 else 0.0
    if stop_pct == 0.0 and stop > 0 and ref_price > 0:
        stop_pct = round((stop - ref_price) / ref_price * 100, 1)
        print(f"⚠️  [Alert] stop_pct was zero — recomputed: {stop_pct:.1f}%")

    horizon_labels = {
        "intraday": "Intraday",
        "swing":    "Swing (2–5d)",
        "position": "Position (1w+)",
    }
    horizon_label = horizon_labels.get(horizon, "Swing")

    # ── Build message ──────────────────────────────────────────────────────────
    if agreement_score is not None:
        # ── Rich aggregator format ─────────────────────────────────────────────
        # Header
        score_line = f"Score: {det_score} | Agreement: {agreement_score:.0f}%"
        if signal_count_bull or signal_count_bear:
            score_line += f" ({signal_count_bull}↑ {signal_count_bear}↓)"

        # Targets line
        if valid_targets:
            t_parts = []
            for i, t in enumerate(valid_targets[:2]):
                pct = (t - entry_mid) / entry_mid * 100 if entry_mid > 0 else 0
                t_parts.append(f'${t:.2f} (+{pct:.1f}%)')
            targets_line = ' | '.join(t_parts)
        else:
            targets_line = 'No valid target'

        # Top signals section
        sig_section = ""
        _top = top_3_signals or []
        _bull = bullish_signals or []
        _bear = bearish_signals or []
        if _top:
            sig_lines = [f'✅ {s}' for s in _top[:3]]
        elif _bull and consensus == "BULLISH":
            sig_lines = [f'✅ {n} ({w:.2f})' for n, w in sorted(_bull, key=lambda x: -x[1])[:3]]
        elif _bear and consensus == "BEARISH":
            sig_lines = [f'✅ {n} ({w:.2f})' for n, w in sorted(_bear, key=lambda x: -x[1])[:3]]
        else:
            sig_lines = []
        # Conflict signal (opposite direction)
        conflict_sig = ""
        if consensus == "BULLISH" and _bear:
            top_bear = max(_bear, key=lambda x: x[1])
            conflict_sig = f'⚠️  {top_bear[0]} ({top_bear[1]:.2f})'
        elif consensus == "BEARISH" and _bull:
            top_bull = max(_bull, key=lambda x: x[1])
            conflict_sig = f'⚠️  {top_bull[0]} ({top_bull[1]:.2f})'
        if sig_lines or conflict_sig:
            sig_section = (
                f'{_SEP}\n'
                f'TOP SIGNALS:\n' +
                '\n'.join(sig_lines) +
                (f'\n{conflict_sig}' if conflict_sig else '')
            )

        # ── Bull vs Bear debate section ────────────────────────────────────────
        debate_section = ""
        if bull_score or bear_score:
            net = bull_score - bear_score
            debate_section = (
                f'{_SEP}\n'
                f'🟢 Bull ({bull_score}): {bull_summary[:80]}\n'
                f'🔴 Bear ({bear_score}): {bear_summary[:80]}\n'
                f'Decision: {signal} (net {net:+d})\n'
            )

        message = (
            f'{emoji} {ticker} ${price:.2f}\n'
            f'{score_line}\n'
            f'{_SEP}\n'
            f'ENTRY:  ${entry_low:.2f} – ${entry_high:.2f}\n'
            f'TARGET: {targets_line}\n'
            f'STOP:   ${stop:.2f} ({stop_pct:.1f}%)\n'
            f'{sig_section}\n'
            f'{debate_section}'
            f'{_SEP}\n'
            f'Market: {market_regime_str or "—"}  Sector: {sector_str or "—"}\n'
            f'Catalyst: {catalyst_str or "No recent news"}\n'
            f'Horizon: {horizon_label}\n'
        )
        if main_risk:
            message += f'Risk: {main_risk[:120]}'

    else:
        # ── Legacy compact format (portfolio exits, partial exits) ─────────────
        target_lines = []
        for i, t in enumerate(valid_targets[:3]):
            pct = (t - entry_mid) / entry_mid * 100 if entry_mid > 0 else 0
            target_lines.append(f'  {labels[i]}: ${t:.2f} (+{pct:.1f}%)')
        if not target_lines:
            target_lines = ['  No target above entry']
        targets_str = '\n'.join(target_lines)

        message = (
            f'Argus - {emoji}\n'
            f'Ticker:     {ticker} at ${price:.2f}\n'
            f'Horizon:    {horizon_label}\n'
            f'Entry zone: ${entry_low:.2f} – ${entry_high:.2f}\n'
            f'Targets:\n{targets_str}\n'
            f'Stop loss:  ${stop:.2f} ({stop_pct:.1f}%)\n'
            f'Confidence: {confidence}%\n'
            f'Reason: {reason}'
        )

    whatsapp_ok = send_whatsapp(message)

    push_ok = False
    if PUSHOVER_APP_TOKEN and PUSHOVER_APP_TOKEN != 'your_key_here' and '@' not in PUSHOVER_APP_TOKEN:
        push_ok = send_push(f'Argus - {signal} {ticker}', message)

    email_ok = send_email(f'Argus {signal} — {ticker} @ ${price:.2f}', message)

    return bool(whatsapp_ok or push_ok or email_ok)

if __name__ == '__main__':
    send_alert(
        ticker='BZAI',
        signal='BUY',
        price=1.79,
        entry_low=1.75,
        entry_high=1.82,
        targets=[1.95, 2.10, 2.35],
        stop=1.65,
        reason='RSI oversold + volume spike 3.7x',
        confidence=72
    )
