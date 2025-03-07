import sys
import logging

from loguru import logger
from starlette.config import Config
from starlette.datastructures import Secret

from .logging import InterceptHandler

config = Config(".env")
HF_TOKEN="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"
HF_API_URL="https://api-inference.huggingface.co/models/essogbe/groupe2-sentiment-analysis"
API_PREFIX = "/api"
VERSION = "0.1.0"
DEBUG: bool = config("DEBUG", cast=bool, default=False)
MAX_CONNECTIONS_COUNT: int = config("MAX_CONNECTIONS_COUNT", cast=int, default=10)
MIN_CONNECTIONS_COUNT: int = config("MIN_CONNECTIONS_COUNT", cast=int, default=10)
SECRET_KEY: Secret = config("SECRET_KEY", cast=Secret, default="")
HF_TOKEN: Secret=config("HF_TOKEN",cast=Secret,default=HF_TOKEN)
HF_API_URL: str = config("HF_API_URL",default="https://api-inference.huggingface.co/models/essogbe/groupe2-sentiment-analysis")
PROJECT_NAME: str = config("PROJECT_NAME", default="Sentiment analysis")

# logging configuration
LOGGING_LEVEL = logging.DEBUG if DEBUG else logging.INFO
logging.basicConfig(
    handlers=[InterceptHandler(level=LOGGING_LEVEL)], level=LOGGING_LEVEL
)
logger.configure(handlers=[{"sink": sys.stderr, "level": LOGGING_LEVEL}])

MODEL_PATH = config("MODEL_PATH", default="./ml/model/")
MODEL_NAME = config("MODEL_NAME", default="model.pkl")
INPUT_EXAMPLE = config("INPUT_EXAMPLE", default="./ml/model/examples/example.json")
