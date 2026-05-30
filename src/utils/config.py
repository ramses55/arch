from pydantic_settings import BaseSettings, SettingsConfigDict




class Settings(BaseSettings):
    mysql_host: str
    mysql_pass: str
    queue_url: str
    mysql_user: str
    mysql_db: str
    access_key: str
    access_key_id: str
    oauth_token: str

    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()
