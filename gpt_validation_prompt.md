# YandexGPT: валидация разметки отзывов

Скрипт `analysis_reviews.py` использует этот промпт по той же схеме: YandexGPT получает детерминированно выбранную случайную выборку отзывов и возвращает JSON-разметку.

Поля:

- `service` — упоминание сервиса;
- `quality` — проблема/упоминание качества мебели;
- `delivery` — упоминание доставки;
- `description_mismatch` — именно несоответствие товара описанию, а не простое слово «описание»;
- `return_claim` — возврат/претензия;
- `soft_furniture` — мягкая мебель;
- `cabinet_furniture` — корпусная мебель;
- `sentiment` — `positive`, `neutral` или `negative`.

Для запуска задаются переменные окружения `YANDEX_GPT_API_KEY`, `YANDEX_CLOUD_FOLDER_ID` и `RUN_GPT=1`.
