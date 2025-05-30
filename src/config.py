import os
# from dotenv import load_dotenv
from pydantic_settings import BaseSettings


# load_dotenv()
# meta-llama/Llama-3.1-8B
class Settings(BaseSettings):
    MODEL_DIR: str = os.path.abspath("./model_files/microsoft/Phi-3-mini-4k-instruct")
    MODEL_NAME: str = os.getenv('MODEL_NAME', "microsoft/Phi-3-mini-4k-instruct")
    MAX_TOKENS: int = int(os.getenv('MAX_TOKENS', 1024))
    DEVICE: str = os.getenv('DEVICE', 'cpu')
    MODEL_PATH: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    API_KEY: str = ""  # For authentication
    MEMORY_TTL: int = 3600  # Conversation expiry in seconds
    MAX_CONVERSATIONS: int = 1000  # Maximum number of stored conversations

    # CORS settings
    CORS_ORIGINS: list = ["*"]

    # Server settings
    HOST: str = "127.0.0.1"
    PORT: int = 8000

    class Config:
        env_prefix = "SLM_"
        env_file = ".env"


settings = Settings()
