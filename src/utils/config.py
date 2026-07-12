from pydantic_settings import BaseSettings, SettingsConfigDict




class Settings(BaseSettings):
    #mysql_host: str
    #mysql_pass: str
    #mysql_db: str
    #mysql_user: str
    #PORT: int
    queue_url: str
    access_key: str
    access_key_id: str
    oauth_token: str
    folder_id: str
    api_key: str
    worker_url: str
    img_limit: int
    res_limit: int

    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()
