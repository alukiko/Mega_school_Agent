# app/core/graph.py

import json
import re
from langgraph.graph import StateGraph, END, MessagesState
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, SystemMessage

# Импортируем свою функцию инициализации из chat.py
from app.core.chat import init_llm_and_tools


def should_continue(state: MessagesState):
    """
    Функция, которая определяет, нужно ли вызывать инструменты,
    или результат уже получен.
    """
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return END


def gather_data(state: MessagesState, llm):
    """
    Узел, который добавляет SystemMessage и вызывает LLM.
    """
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
    return {"messages": [response]}


def build_workflow():
    """
    Собираем и компилируем StateGraph со всеми нодами и переходами.
    Возвращаем объект graph для дальнейшего использования.
    """
    llm, tools = init_llm_and_tools()

    # Создаем узел вызова инструментов
    tool_node = ToolNode(tools)

    # Создаем объект StateGraph
    workflow = StateGraph(MessagesState)

    # Добавляем узлы
    # gather_data_node будет вызываться с дополнительным параметром llm
    def gather_data_node(state: MessagesState):
        return gather_data(state, llm)

    workflow.add_node("gather_data_node", gather_data_node)
    workflow.add_node("tools", tool_node)

    # Задаём переходы
    workflow.set_entry_point("gather_data_node")
    workflow.add_conditional_edges("gather_data_node", should_continue, ["tools", END])
    workflow.add_edge("tools", "gather_data_node")

    # Компилируем граф
    graph = workflow.compile()
    return graph
