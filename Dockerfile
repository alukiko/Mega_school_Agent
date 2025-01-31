# Используем официальный Python-образ в качестве базового
FROM python:3.10-slim

# Объявляем рабочую директорию внутри контейнера
WORKDIR /app

# Скопируем файл зависимостей
COPY requirements.txt .

# Установка зависимостей
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Если хотите копировать .env (НЕ рекомендуется для публичных репозиториев):
# COPY .env .

# Копируем исходный код приложения в контейнер
COPY . /app

# Открываем порт 80 для доступа
EXPOSE 80

# Запуск uvicorn-сервера
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
