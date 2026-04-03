import requests
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

def send_whatsapp(message: str):
    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        msg = client.messages.create(
            body=message,
            from_=WHATSAPP_FROM,
            to=WHATSAPP_TO
        )
        print(f'✅ WhatsApp sent: {msg.sid}')
        return True
    except Exception as e:
        print(f'❌ WhatsApp error: {e}')
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
        print(f'✅ Push sent: {r.status_code}')
        return True
    except Exception as e:
        print(f'❌ Push error: {e}')
        return False

def send_alert(ticker: str, signal: str, price: float,
               entry_low: float, entry_high: float,
               target: float, stop: float,
               reason: str, confidence: int):

    emoji = 'BUY' if signal == 'BUY' else 'SELL' if signal == 'SELL' else 'HOLD'

    message = (
        f'Stock AI Agent - {emoji}\n'
        f'Ticker: {ticker} at ${price:.2f}\n'
        f'Zone: ${entry_low:.2f} - ${entry_high:.2f}\n'
        f'Target: ${target:.2f} | Stop: ${stop:.2f}\n'
        f'Reason: {reason}\n'
        f'Confidence: {confidence}%'
    )

    send_whatsapp(message)

    if PUSHOVER_APP_TOKEN and PUSHOVER_APP_TOKEN != 'your_key_here':
        send_push(f'Stock AI Agent - {signal} {ticker}', message)

if __name__ == '__main__':
    send_alert(
        ticker='BZAI',
        signal='BUY',
        price=1.79,
        entry_low=1.75,
        entry_high=1.82,
        target=2.10,
        stop=1.65,
        reason='RSI oversold + volume spike 3.7x',
        confidence=72
    )
