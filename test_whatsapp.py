from twilio.rest import Client
import os
from dotenv import load_dotenv

load_dotenv()

account_sid = os.getenv('TWILIO_ACCOUNT_SID')
auth_token = os.getenv('TWILIO_AUTH_TOKEN')
to_number = 'whatsapp:+17635685097'
from_number = 'whatsapp:+14155238886'

print(f'Account SID: {account_sid[:10]}...')
print(f'Sending WhatsApp to: {to_number}')

client = Client(account_sid, auth_token)

message = client.messages.create(
    body='Stock AI Agent - WhatsApp Test! BUY BZAI $1.79 Zone: $1.75-$1.82 Target: $2.10 Stop: $1.65 Confidence: 72%',
    from_=from_number,
    to=to_number
)

print(f'Status: {message.status}')
print(f'SID: {message.sid}')
print('Check your WhatsApp!')
