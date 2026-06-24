from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    DATABASE_URL: str
    GEMINI_API_KEY: str
    REDIS_URL: str 
    APP_ENV: str = "development"
    CHUNK_SIZE: int = 1500
    CHUNK_OVERLAP: int = 150
    TOP_K_RESULTS: int = 5
    EMBEDDING_DIM: int = 1024 
    RERANKER_TOP_K: int = 5
    RETRIEVER_TOP_K: int = 10
    MAX_MESSAGES_PER_SESSION:int = 15  
    GZIP_LEVEL :int= 6    

    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama-3.3-70b-versatile" 
    NVIDIA_API_KEY: str = ""
    NVIDIA_MODEL: str = "z-ai/glm-5.1"           
    NEMOTRON_API_KEY: str = ""
    NEMOTRON_VOICE_MODEL: str = "nvidia/nemotron-voicechat" 
    EMBEDDING_PROVIDER: str = "gemini"            
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
  
  

settings = Settings()