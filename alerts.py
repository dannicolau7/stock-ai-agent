import requests
from twilio.rest import Client
from config import (
    TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN,
    TWILIO_FROM_NUMBER, TWILIO_TO_NUMBER,
    PUSHOVER_APP_TOKEN, PUSHOVER_USER_KEY,
)


def format_signal_message(state: dict) -> str:
    signal = state.get("signal", "HOLD")
    ticker = state.get("ticker", "")
    price = state.get("current_price", 0)
    entry = state.get("entry_zone", "N/A")
    targets = state.get("targets", [])
    stop_loss = state.get("stop_loss", 0)
    rsi = state.get("rsi", 0)
    macd = state.get("macd", {})
    reasoning = state.get("reasoning", "")

    emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}.get(signal, "⚪")
    targets_str = " / ".join(f"${t:.4f}" for t in targets) if targets else "N/A"

    lines = [
        f"{emoji} {signal} — {ticker}",
        f"Price:      ${price:.4f}",
        f"Entry Zone: {entry}",
        f"Targets:    {targets_str}",
        f"Stop Loss:  ${stop_loss:.4f}",
        f"RSI:        {rsi:.1f}",
        f"MACD Hist:  {macd.get('histogram', 0):+.6f}",
    ]
    if reasoning:
        lines.append(f"\n{reasoning[:300]}")
    return "\n".join(lines)


def send_sms(message: str) -> bool:
    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        client.messages.create(body=message, from_=TWILIO_FROM_NUMBER, to=TWILIO_TO_NUMBER)
        print("[Alerts] SMS sent")
        return True
    except Exception as e:
        print(f"[Alerts] SMS error: {e}")
        return False


def send_push(title: str, message: str, priority: int = 0) -> bool:
    try:
        r = requests.post(
            "https://api.pushover.net/1/messages.json",
            data={
                "token": PUSHOVER_APP_TOKEN,
                "user": PUSHOVER_USER_KEY,
                "title": title,
                "message": message,
                "priority": priority,
            },
            timeout=10,
        )
        ok = r.status_code == 200
        if ok:
            print("[Alerts] Push notification sent")
        return ok
    except Exception as e:
        print(f"[Alerts] Push error: {e}")
        return False


def send_signal_alert(state: dict) -> bool:
    message = format_signal_message(state)
    signal = state.get("signal", "HOLD")
    ticker = state.get("ticker", "")
    title = f"{signal} — {ticker}"
    priority = 1 if signal in ("BUY", "SELL") else 0
    sms_ok = send_sms(message)
    push_ok = send_push(title, message, priority=priority)
    return sms_ok or push_ok
