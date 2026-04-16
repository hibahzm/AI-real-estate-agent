"""
config.py — Reads environment variables from .env and exposes them as typed settings.

Why pydantic-settings? It validates that required keys exist at startup,
so the app fails fast with a clear error if a key is missing, instead of
crashing later in a request handler.
"""

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # Required API keys — app will NOT start if these are missing
    openai_api_key: str
    supabase_url: str
    supabase_key: str

    # File paths — have sensible defaults, can be overridden in .env
    model_file_path: str = "saved_model/trained_house_price_model.pkl"
    training_stats_path: str = "saved_model/training_statistics.json"

    class Config:
        env_file = ".env"           # loads from .env file automatically
        env_file_encoding = "utf-8"
        case_sensitive = False      # OPENAI_API_KEY and openai_api_key both work


@lru_cache()
def get_settings() -> Settings:
    """
    Returns a cached Settings instance.
    Using lru_cache means Settings() is only created once, not on every request.
    """
    return Settings()


# Convenience: import `settings` directly anywhere in the app
settings: Settings = get_settings()