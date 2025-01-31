import os
from dotenv import load_dotenv

load_dotenv()

# Получаем значения из переменных окружения
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
VSEGPT_API_KEY = os.getenv("VSEGPT_API_KEY")
