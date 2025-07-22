import os
from dotenv import load_dotenv
load_dotenv()
print("OpenAI Key?", os.getenv("OPENAI_API_KEY") is not None)
print("Twilio SID?", os.getenv("TWILIO_ACCOUNT_SID") is not None)
