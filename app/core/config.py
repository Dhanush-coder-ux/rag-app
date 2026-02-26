from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    DATABASE_URL: str
    GEMINI_API_KEY: str
    APP_ENV: str = "development"
    CHUNK_SIZE: int = 1500
    CHUNK_OVERLAP: int = 150
    TOP_K_RESULTS: int = 5
    EMBEDDING_DIM: int = 768 
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()