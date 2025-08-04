import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load the environment variables from .env file
load_dotenv()

# Configure the API key
try:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY not found in .env file or environment.")
    else:
        genai.configure(api_key=api_key)

        print("--- Available Gemini Models ---")
        for m in genai.list_models():
          # Check if the 'generateContent' method is supported
          if 'generateContent' in m.supported_generation_methods:
            print(f"- {m.name}")
        print("-----------------------------")

except Exception as e:
    print(f"An error occurred: {e}")
