# check_gemini.py
import google.generativeai as genai
import os
from dotenv import load_dotenv
load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

print("Available embedding models:")
for m in genai.list_models():
    if "embed" in m.name.lower():
        print(f"  {m.name} | methods: {m.supported_generation_methods}")