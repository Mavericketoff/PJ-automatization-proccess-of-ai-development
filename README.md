# Credit Default Prediction — ML Pipeline

End-to-end пайплайн для предсказания вероятности дефолта по кредитной карте
## Стек

| Компонент | Технология |
|---|---|
| Управление зависимостями | Poetry |
| Валидация данных | Pandera |
| Модель | GradientBoostingClassifier (sklearn) |
| Подбор гиперпараметров | RandomizedSearchCV |
| Трекинг экспериментов | MLflow |
| Версионирование данных/моделей | DVC |
| API | FastAPI |
| Контейнеризация | Docker |
| CI | GitHub Actions (flake8 + black + pytest + Pandera) |
| Мониторинг | PSI-скрипт |

## Требования

- Python 3.10+
- [Poetry](https://python-poetry.org/docs/#installation)
- [DVC](https://dvc.org/doc/install)
- Docker (опционально)

## Быстрый старт

```bash
git clone https://github.com/Mavericketoff/PJ-automatization-proccess-of-ai-development.git
cd PJ-automatization-proccess-of-ai-development

# установка зависимостей через Poetry
poetry install

# подтянуть данные и модель из DVC-хранилища
dvc pull

# воспроизвести пайплайн (подготовка данных + обучение)
dvc repro

# запустить API
poetry run uvicorn src.api.app:app --reload --port 8000