import re
from fastapi import FastAPI
from langchain_core.messages import HumanMessage

from app.models import UserRequest
from app.core.graph import build_workflow

# Создаём приложение
app = FastAPI()

# При инициализации модуля создаём граф
graph = build_workflow()


@app.post('/api/request')
async def predict(user: UserRequest):
    query = user.query
    user_id = user.id

    # Формируем входное сообщение
    input_messages = [HumanMessage(query)]
    # Запускаем граф
    output = graph.invoke({"messages": input_messages})

    # Извлекаем результат
    messages = output["messages"]
    ai_response = messages[-1]  # последнее сообщение от AI
    text = ai_response.content

    # Ищем ссылки
    url_pattern = r"https?://[^\s)\"\]]+"
    urls = re.findall(url_pattern, text) or []
    # Возьмём не более 2 ссылок
    urls = urls[:2]

    # Ищем ответ (answer(...))
    pattern = r"answer\((.*?)\)"

    # Вывод результатов
    try:
        ans = int(re.findall(pattern, text)[0])
    except Exception as e:
        print(e)
        ans = 1

    text = 'openai/gpt-4o-mini\n' + text
    return {
        "id": user_id,
        "answer": ans,
        "reasoning": text,
        "sources": urls
    }
