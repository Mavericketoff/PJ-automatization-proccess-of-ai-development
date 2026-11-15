FROM python:3.11-slim AS base

# устанавливаем poetry
RUN pip install --no-cache-dir poetry==1.7.1

WORKDIR /app

# копируем только файлы зависимостей для кэширования слоя
COPY pyproject.toml poetry.lock* ./

# ставим зависимости без создания виртуалки
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --only main

# теперь копируем весь проект
COPY . .

EXPOSE 8000

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
