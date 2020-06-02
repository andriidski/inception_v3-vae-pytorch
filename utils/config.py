import os
from dotenv import load_dotenv

load_dotenv()
SLACK_WEBHOOK_URL = os.getenv('SLACK_WEBHOOK_URL')
SLACK_CHANNEL = '#general'
