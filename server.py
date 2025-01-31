import joblib
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

import json
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, MessagesState
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.tools import TavilySearchResults
from langchain.agents import load_tools
import os

import re


# Устанавливаем API-ключи
os.environ["TAVILY_API_KEY"] = "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"
os.environ["VSEGPT_API_KEY"] = "sk-or-vv-e7db6d74cd45b77c93abd65ed86459fa942e5e5e437fd62777dc29956f24334d"

# Tavily для поиска в интернете
search_tool = TavilySearchResults(max_results=3, include_answer=True, include_raw_content=False)


# Список инструментов в корректном формате
tools = [search_tool]

# Создаем LLM-модель для работы с VseGPT API
llm = ChatOpenAI(
    model="openai/gpt-4o-mini",  # Используем нужную модель VseGPT
    temperature=0.7,
    max_tokens=None,
    openai_api_base="https://api.vsegpt.ru/v1",  # Указываем кастомный URL API
    openai_api_key=os.getenv("VSEGPT_API_KEY")  # Берем API-ключ из переменной окружения
).bind_tools(tools)  # <-- Теперь инструменты передаются корректно

print("Модель VseGPT успешно инициализирована!")

# функция, которая определяет нужно ли вызывать инструменты 
# или результат уже получен
def should_continue(state: MessagesState):
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return END

# функция для ноды взаимодейтсвия с LLM
def gather_data(state: MessagesState):

    messages = state["messages"]

    messages.append(SystemMessage(content='''
    Ты – ассистент Университета ИТМО. Отображай информацию актуальную только для ИТМО
    Если это вопрос с выбором ответа, укажи в конце сообщения номер павильного ответа в формате - answer(...) где вместо ... ответ
    Твоя задача – предоставлять информацию о вузе, его факультетах, образовательных программах, поступлении, стипендиях и других аспектах студенческой жизни.
    Ищи актуальную информацию на 2025 год, именно в ИТМО.
    Если спрашивают про новости, найди их на сайте - https://news.itmo.ru/ru/
    
    Ты можешь отвечать на вопросы о:
    * Истории и достижениях Университета ИТМО
    * Факультетах и кафедрах
    * Доступных образовательных программах (бакалавриат, магистратура, аспирантура)
    * Условиях поступления и вступительных испытаниях
    * Стоимости обучения и стипендиях
    * Студенческой жизни, кружках и мероприятиях
    * Программах обмена и международном сотрудничестве
    * Кампусе, общежитиях и инфраструктуре
    * Последних новостях Университета ИТМО

    Всегда прикладывай ссылки на материалы откуда берешь информацию

    '''))

    response = llm.invoke(messages)

    # информация для отладки
    #print(json.dumps(response.tool_calls, indent=2,ensure_ascii=False))
    #print(json.dumps(response.content, indent=2,ensure_ascii=False))
   
    return {"messages": [response]}

# встроенная в langgraph нода вызова инструментов
tool_node = ToolNode(tools)

workflow = StateGraph(MessagesState)

# задаём ноды
workflow.add_node("gather_data_node", gather_data)
workflow.add_node("tools", tool_node)

# задаём переходы между нодами
# входная нода - gather_data_node
workflow.set_entry_point("gather_data_node")
# после gather_data_node вызываем should_continue, 
# чтобы определить что делать дальше
workflow.add_conditional_edges("gather_data_node", 
                               should_continue, 
                               ["tools", END])
# после вызова инструментов всегда возвращаемся к ноде LLM, 
# чтобы отдать ей результат вызова инструментов
workflow.add_edge("tools", "gather_data_node")

graph = workflow.compile()


# App creation and model loading
app = FastAPI()


class User_request(BaseModel):
    """
    Input features validation for the ML model
    """
    query: str
    id: int
   


@app.post('/predict')
def predict(User: User_request):


    Quary = User.query
    id_u = User.id
    
    input_messages = [HumanMessage(Quary)]  
    output = graph.invoke({"messages": input_messages})
    
    # извлекаем ответ
    messages = output["messages"]

    # Извлекаем последнее сообщение от AI
    ai_response = messages[-1]

    # извлекаем ответ
    text = ai_response.content

    # Регулярное выражение для поиска URL
    url_pattern = r"https?://[^\s)\"\]]+"  # Поддержка HTTP/HTTPS, исключение лишних символов
    try:
        # Поиск всех ссылок в тексте
        urls = re.findall(url_pattern, text)

        # Вывод результатов
        urls = urls[:2]
    except:
        urls = []

    # Регулярное выражение для поиска ответа
    pattern = r"answer\((.*?)\)"

    # Вывод результатов
    try:
        ans = re.findall(pattern, text)
    except:
        ans = 1
    

    return {
        "id": id_u,
        "answer": ans,
        "reasoning": text,
        "sources": urls
    }



if __name__ == '__main__':
    # Run server using given host and port
    uvicorn.run(app, host='127.0.0.1', port=80)
