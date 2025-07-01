import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    PROVIDER = os.getenv("PROVIDER", "openai").lower()

    # OpenAI
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4")
    
    #groq
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_MODEL = "mixtral-8x7b-32768" # "llama3-70b-8192" "llama3-8b-8192"  # or "mixtral-8x7b-32768", etc.
    # SYSTEM_PROMPT = "You are a sarcastic assistant. Reply with witty and dry humor."

    # Ollama
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    

settings = Settings()
