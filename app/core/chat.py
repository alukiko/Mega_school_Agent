import os
from langchain_openai import ChatOpenAI
from langchain_community.tools import TavilySearchResults

from config.settings import TAVILY_API_KEY, VSEGPT_API_KEY


def init_llm_and_tools():
    """
    Создаем LLM-модель для работы с VseGPT API и инструменты поиска.
    Возвращаем (llm, tools).
    """
    # Установим API-ключи из конфига (при желании — через setdefault)
    os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
    os.environ["VSEGPT_API_KEY"] = VSEGPT_API_KEY

    # Инициализируем инструмент TavilySearchResults
    search_tool = TavilySearchResults(
        max_results=3,
        include_answer=True,
        include_raw_content=False
    )

    # Список инструментов
    tools = [search_tool]

    # Инициализируем ChatOpenAI
    llm = ChatOpenAI(
        model="openai/gpt-4o-mini",  # укажите свою нужную модель
        temperature=0.7,
        max_tokens=None,
        openai_api_base="https://api.vsegpt.ru/v1",
        openai_api_key=VSEGPT_API_KEY
    ).bind_tools(tools)

    print("Модель VseGPT успешно инициализирована!")

    return llm, tools
