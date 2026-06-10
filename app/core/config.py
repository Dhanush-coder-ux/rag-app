from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    DATABASE_URL: str
    GEMINI_API_KEY: str
    REDIS_URL: str 
    APP_ENV: str = "development"
    CHUNK_SIZE: int = 1500
    CHUNK_OVERLAP: int = 150
    TOP_K_RESULTS: int = 5
    EMBEDDING_DIM: int = 768 
    RERANKER_TOP_K: int = 5
    RETRIEVER_TOP_K: int = 10
    MAX_MESSAGES_PER_SESSION:int = 15  
    GZIP_LEVEL :int= 6    
    OLLAMA_URL: str = "http://localhost:11434"
    UPLOADS_DIR: str = "uploads"              # folder to persist uploaded files
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama-3.3-70b-versatile"  # default Groq model
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
  
  

settings = Settings()