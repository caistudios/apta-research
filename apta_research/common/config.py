import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

from apta_research.common.base import DIR_PATH

load_dotenv()


class Settings(BaseSettings):
    OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "FAKE")
    LLAMA_API_KEY: str = os.environ.get("LLAMA_API_KEY", "FAKE")
    FW_API_KEY: str = os.environ.get("FW_API_KEY", "FAKE")
    DEEP_INFRA_API_KEY: str = os.environ.get("DEEP_INFRA_API_KEY", "FAKE")
    GROQ_API_KEY: str = os.environ.get("GROQ_API_KEY", "FAKE")

    model_config = SettingsConfigDict(env_file=DIR_PATH / ".env")


settings = Settings()

if __name__ == "__main__":
    print(settings)
    # breakpoint()
