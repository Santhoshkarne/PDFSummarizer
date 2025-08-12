# Install the library first:
# pip install google-generativeai python-dotenv

import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables (optional, if you store API key in .env file)
load_dotenv()

# You can either set your API key in a .env file or directly here:
# os.environ["GOOGLE_API_KEY"] = "YOUR_API_KEY_HERE"

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# Pick a Gemini model
model = genai.GenerativeModel("gemini-1.5-flash")

# Send a test prompt
response = model.generate_content("what is the capital of France?")

# Print the output
print("Gemini Response:", response.text)
