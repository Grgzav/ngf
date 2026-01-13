from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration. Loads from env vars and optional .env file."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    onet_api_key: str | None = None
    onet_base_url: str = "https://services.onetcenter.org"


settings = Settings()