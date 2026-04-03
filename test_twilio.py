from twilio.rest import Client
import os
from dotenv import load_dotenv

load_dotenv()

account_sid = os.getenv('TWILIO_ACCOUNT_SID')
auth_token = os.getenv('TWILIO_AUTH_TOKEN')
messaging_sid = os.getenv('TWILIO_MESSAGING_SID')
to_number = os.getenv('TWILIO_TO_NUMBER')

print(f'Account SID: {account_sid[:10]}...')
print(f'Messaging SID: {messaging_sid[:10]}...')
print(f'To: {to_number}')

client = Client(account_sid, auth_token)

message = client.messages.create(
    body='Stock AI Agent - Test Alert! BUY BZAI $1.79 Zone: $1.75-$1.82 Target: $2.10 Stop: $1.65 Confidence: 72%',
    messaging_service_sid=messaging_sid,
    to=to_number
)

print(f'Status: {message.status}')
print(f'SID: {message.sid}')
print('Check your phone!')
