# Cross-domain Sentiment Data Agents

Учебный data-centric проект для бинарной классификации тональности английских отзывов из двух доменов: товары Amazon и игры Steam. Проект реализует задания 1–4 и объединяет их в обычный Python-пайплайн без внешнего оркестратора.

Дополнительно поддерживается конфигурируемая разметка фотографий и видеоклипов через доступного мультимодального ИИ-агента: подготовка кадров → skill → настоящая ручная проверка → необязательные Active Learning и обучение. Отдельный API-ключ разметчика не нужен. Начальная конфигурация — `configs/media.yaml`; инструкция, ограничения и команды продолжения описаны в [руководстве по фото и видео](docs/media_pipeline.md). Ниже сохранено описание исходного текстового сценария.

Важно: автоматическая разметка не называется человеческой. Первый запуск формирует очередь проверки и останавливается со статусом `review_required`. Только после того, как человек просмотрит очередь, укажет имя и сохранит исправления, второй запуск отмечает HITL как подтверждённый и продолжает Active Learning и обучение модели.

## Что реализовано

- `DataCollectionAgent`: сбалансированные выборки двух готовых Hugging Face-датасетов — `mteb/amazon_polarity` и `reapxdev/steam-reviews-scraper`, приоритет проверенных файлов проекта, изоляция ошибок источников, единая схема и EDA-артефакты.
- `DataQualityAgent`: пропуски/пустые тексты, точные и нормализованные текстовые дубликаты, выбросы длины по IQR и z-score, дисбаланс, две стратегии очистки и отчёт до/после.
- `AnnotationAgent`: авторазметка Transformer-моделью, confidence, спецификация, метрики agreement/Cohen's κ, Label Studio JSON и очередь по низкой уверенности без показа исходной метки.
- `ActiveLearningAgent` (Track A): TF-IDF + Logistic Regression, стратегии `entropy`, `margin`, `random`, одинаковый split для честного сравнения, пять итераций по 20 примеров со стартом 50.
- `TrainAgent`: финальная модель после AL, сохранение `joblib`, aggregate и per-domain accuracy/F1 на внешнем holdout, который не участвовал в выборе AL-стратегии.
- реальная точка HITL: Streamlit-интерфейс либо терминальная проверка; сохраняются `reviewer` и `reviewed_at`.
- один CLI: `./.venv/bin/python run_pipeline.py`.

## Архитектура

```text
Amazon Polarity (HF dataset) ────────────┐
                                           ├─ DataCollectionAgent ─ data/raw
Steam Reviews Scraper (HF dataset) ─────┘
                                          │
                                          ▼
                               DataQualityAgent ─ data/processed + reports/quality
                                          │
                                          ▼
                                AnnotationAgent ─ auto labels + review queue
                                          │
                            ┌─────────────┴─────────────┐
                            ▼                           │
                    человек проверяет                  │ status=review_required
                    Streamlit/terminal                  │
                            │                           │
                            └─────────────┬─────────────┘
                                          ▼
                               data/labeled/reviews_final
                                          │
                               ActiveLearningAgent
                                          │ selected + reviewed rows
                                          ▼
                                      TrainAgent
                                          │
                                          ▼
                                  reports + model
```

`pipeline/runner.py` сохраняет промежуточные артефакты, поэтому после ручной паузы запуск продолжается с уже собранных данных. `record_id` стабилен и используется для безопасного объединения исправлений.

## Быстрый запуск на macOS Apple Silicon

Требуется Python 3.11 или новее. Рекомендуется Python 3.12 arm64. Клонируйте репозиторий и создайте изолированное окружение внутри проекта:

```bash
git clone https://github.com/GulkoMI/data-agent-project.git
cd data-agent-project
python3.12 -m venv .venv
uname -m
./.venv/bin/python --version
./.venv/bin/python -m pip install -r requirements.txt
```

`uname -m` на Apple Silicon должен вывести `arm64`. `.runtime` и `.venv` являются локальными runtime-артефактами и не добавляются в Git.

Проверка PyTorch/MPS:

```bash
./.venv/bin/python -c "import platform, torch; print(platform.machine()); print('MPS built:', torch.backends.mps.is_built()); print('MPS available:', torch.backends.mps.is_available())"
```

При `annotation.device: auto` агент выбирает MPS, если он доступен, иначе CPU. Для редкой неподдерживаемой MPS-операции можно разрешить CPU fallback перед запуском:

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

Конфигурация содержит официальные Hugging Face ID и пути к сохранённым снимкам датасетов и модели. Большие локальные снимки исключены из Git: при их отсутствии пайплайн загружает публичные ресурсы по HF ID. Если полный локальный комплект присутствует в `data/external/huggingface/` и `models/huggingface/`, пайплайн не обращается к сети. Неполный локальный комплект модели считается ошибкой, чтобы не допустить скрытой сетевой догрузки. Пайплайн не обращается к Steam API и не требует Steam-токена или API-ключа.

Для проверяемого запуска без сети:

```bash
export TMPDIR="$PWD/.runtime/tmp"
export HF_HOME="$PWD/.runtime/cache/huggingface"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

Альтернативная установка для разработки, включая Ruff и editable package:

```bash
./.venv/bin/python -m pip install -e ".[annotation,ui,notebooks,dev]"
```

## End-to-end: честный двухзапусковый HITL

### 1. Сбор, очистка и авторазметка

```bash
./.venv/bin/python run_pipeline.py --review-mode required --force
```

Ожидаемое состояние процесса — `review_required`, а не `completed`. Будут созданы, среди прочего:

- `data/raw/reviews_raw.parquet`;
- `data/processed/reviews_clean.parquet`;
- `data/labeled/reviews_auto_labeled.parquet`;
- `data/review/review_queue.csv`;
- `data/review/labelstudio_import.json`;
- `reports/annotation_spec.md` и отчёты качества.

`--force` пересчитывает автоматические этапы. Не используйте его без необходимости на втором запуске.

### 2. Проверка человеком в Streamlit

```bash
./.venv/bin/python -m streamlit run hitl_app.py
```

В интерфейсе нужно прочитать каждый текст, поставить флажок `reviewed`, указать имя проверяющего и:

- оставить `human_label` пустым, если `auto_label` подтверждён;
- выбрать `negative` или `positive`, если метку нужно исправить;
- нажать **Save verified corrections**.

Результат сохраняется в `data/review/review_queue_corrected.csv`. Пустая метка означает явное подтверждение авторазметки, а не отсутствие проверки. Имя проверяющего обязательно для каждой строки.

Терминальная альтернатива без Streamlit:

```bash
./.venv/bin/python run_pipeline.py --review-mode terminal --reviewer "Имя проверяющего"
```

### 3. Продолжение после проверки

```bash
./.venv/bin/python run_pipeline.py --review-mode required
```

Runner валидирует `record_id`, метки и заполненного `reviewer`, объединяет исправления, затем запускает Active Learning, финальное обучение и отчёты. Проверить фактический статус можно в `reports/pipeline_state.json`: только `status: completed` вместе с `metrics.hitl.verified: true` означает завершённый HITL-проход.

Нельзя заранее создавать фиктивный `reviewer` или выдавать `source_label`/`auto_label` за решение человека. История реальной проверки хранится в колонках `human_label`, `reviewer`, `reviewed_at` и `review_changed`.

## Offline/smoke режим

Для проверки структуры и первых этапов без загрузки большого датасета:

```bash
./.venv/bin/python run_pipeline.py --offline --review-mode required --force
```

Этот флаг заменяет основной сбор на маленький детерминированный fixture из `data/fixtures/reviews_fixture.csv`, принудительно использует локальный lexicon backend и пишет всё в изолированные `data/smoke`, `reports/smoke`, `models/smoke`. Сеть не требуется.

Для полного технического smoke-теста используется уменьшенный AL-цикл. Он проверяет весь data flow, но не заменяет обязательный основной эксперимент `N=50 + 5×20` и не является финальным результатом.

Режим `--review-mode auto-only` предназначен только для технических проверок и никогда не засчитывается как HITL: итоговый статус, если остальные этапы могут выполниться, будет `completed_without_hitl`.

## Конфигурация

Все основные параметры находятся в `config.yaml`:

- объём и баланс каждого источника;
- HF ID, локальный путь, формат, split, streaming-режим и размер выборки каждого датасета, а для Steam — фильтр `language: english`;
- выбранная стратегия качества (`conservative` или `strict`);
- Transformer-модель, batch size, confidence threshold и размер review queue;
- параметры AL (`initial_size: 50`, `n_iterations: 5`, `batch_size: 20`);
- TF-IDF и параметры финального holdout.

Текущая конфигурация нацелена на сбалансированные 300 Amazon + 300 Steam строк. Фактическое число фиксируется в `data/raw/collection_manifest.json`; оно может быть меньше, если в отфильтрованной англоязычной части HF-датасета не хватит одного из классов; пайплайн требует минимум два непустых источника.

## Контракт данных

Единая сырая схема всегда содержит:

| Колонка | Смысл |
|---|---|
| `record_id` | стабильный уникальный ID для lineage и HITL merge |
| `text` | полный английский текст отзыва |
| `audio`, `image` | фиксированные поля общей схемы; для text-модальности пусты |
| `label` | исходная weak/gold метка источника |
| `source` | `amazon_polarity` или `steam_reviews` |
| `source_id` | ID записи внутри источника |
| `collected_at` | UTC-время сбора |

Для Steam-источника агент читает готовую схему `reapxdev/steam-reviews-scraper`: `reviewText` становится `text`, boolean-поле `votedUp` преобразуется в `negative`/`positive`, а `language` используется для отбора только англоязычных отзывов.

После авторазметки добавляются:

| Колонка | Смысл |
|---|---|
| `source_label` | сохранённая исходная метка; не перезаписывается агентом |
| `auto_label` | прогноз AnnotationAgent |
| `confidence` | уверенность авторазметки в диапазоне 0–1 |
| `annotation_backend` | реально использованный backend; `lexicon` используется только в offline smoke |
| `needs_review`, `review_reason` | флаг и причина попадания в очередь |
| `human_label` | подтверждённая/исправленная человеком метка только после review |
| `final_label` | `human_label`, если он проверен, иначе `auto_label` |
| `reviewer`, `reviewed_at`, `review_changed` | проверяемое происхождение ручного решения |

Исходные рейтинги — полезные, но несовершенные weak labels. Они используются для диагностики, holdout-оценки и как скрытый oracle в учебной AL-симуляции; это не утверждение, что их заново разметил человек.

## Active Learning: интерпретация эксперимента

Track A сравнивает `entropy`, `margin` и `random` на одном и том же стратифицированном initial/pool/test split. Модель не видит `source_label` пула при выборе: метка раскрывается только после query, имитируя получение аннотации. Поэтому эксперимент воспроизводим и без утечки в selection, но остаётся симуляцией, а не дополнительной человеческой разметкой.

Артефакты после завершённого второго запуска:

- `reports/active_learning/al_results.json`;
- `reports/active_learning/learning_curves.png`;
- `reports/active_learning/strategy_comparison.json`;
- `reports/active_learning/al_report.md`.

«Сэкономленные метки» вычисляются только если стратегия действительно достигла финального macro F1 random baseline; иначе отчёт честно пишет `not reached`/`n/a`.

## Ноутбуки

Ноутбуки не копируют реализацию агентов: они импортируют production-классы и читают сохранённые артефакты.

- `notebooks/eda.ipynb` — схема, срезы и EDA-артефакты collection;
- `notebooks/quality_analysis.ipynb` — детекция, сравнение `conservative`/`strict` и обоснование;
- `notebooks/annotation_workflow.ipynb` — метрики, очередь, спецификация и HITL hand-off;
- `notebooks/al_experiment.ipynb` — воспроизводимый вызов production AL и готовые curves/results.

Запуск:

```bash
./.venv/bin/python -m jupyter lab notebooks
```

Каждый notebook безопасно сообщает об отсутствующем артефакте. Тяжёлые операции по умолчанию в notebook выключены; сначала рекомендуется запустить CLI.

## Label Studio

`data/review/labelstudio_import.json` — корневой JSON-массив задач с predictions, совместимый с импортом Label Studio. Интерфейс можно настроить файлом `label_studio_config.xml`. Основной воспроизводимый merge в этом проекте сделан через Streamlit; экспорт Label Studio нужно преобразовать обратно в таблицу `record_id,human_label,reviewer` перед объединением.

## Опциональный локальный LLM: Ollama

Ядро проекта и оценочные метрики не зависят от LLM. В `utils/ollama.py` есть отказоустойчивый локальный helper для бонусного текстового объяснения проблем. При `annotation.ollama.enabled: true` `run_pipeline.py` вызывает локальный Ollama после quality-stage и сохраняет `reports/llm_quality_advice.md`; при недоступном сервисе основной пайплайн продолжает работу. По умолчанию интеграция выключена, поэтому наличие кода само по себе не выдаётся за выполненный LLM-прогон.

Ollama не требует облачного API-ключа:

```bash
ollama serve
ollama pull gemma3:4b
export OLLAMA_BASE_URL=http://localhost:11434
export OLLAMA_MODEL=gemma3:4b
./.venv/bin/python -c "from utils.ollama import ollama_chat; print(ollama_chat('Explain why duplicate reviews can bias a classifier.'))"
```

Документация локального Chat API: https://docs.ollama.com/api/chat

## Тесты и проверка качества кода

```bash
./.venv/bin/python -m pytest
./.venv/bin/python -m pytest --cov=agents --cov=pipeline
./.venv/bin/python -m ruff check .
```

Тесты агентов используют локальные DataFrame и mock backend/HTTP; обычный unit test run не загружает данные из сети. Проверку удалённого HF fallback выполняйте отдельно только при необходимости.

Проверка публичных импортов из технических контрактов:

```bash
./.venv/bin/python -c "from data_collection_agent import DataCollectionAgent; from data_quality_agent import DataQualityAgent; from annotation_agent import AnnotationAgent; from al_agent import ActiveLearningAgent; print('imports ok')"
```

## Соответствие заданиям

| Блок | Реализация | Артефакты после запуска |
|---|---|---|
| Задание 1 | два HF-источника через `load_dataset`, универсальные skills `scrape` и `fetch_api`, `merge`, единая схема | `agents/data_collection_agent.py`, `data/raw`, `reports/eda`, `notebooks/eda.ipynb` |
| Задание 2 | missing, duplicate, IQR/z-score, imbalance, 2 стратегии, до/после | `agents/data_quality_agent.py`, `reports/quality`, `notebooks/quality_analysis.ipynb` |
| Задание 3 | auto-label, spec, κ/agreement, Label Studio, low-confidence HITL | `agents/annotation_agent.py`, `reports/annotation_spec.md`, `data/review`, `notebooks/annotation_workflow.ipynb` |
| Задание 4 Track A | N=50, 5×20, entropy/margin/random, accuracy/F1, learning curves | `agents/al_agent.py`, `reports/active_learning`, `notebooks/al_experiment.ipynb` |
| Data Project | единый Python runner, реальный HITL, модель, data card, 5 разделов отчёта | `run_pipeline.py`, `hitl_app.py`, `models`, `data/labeled/DATA_CARD.md`, `reports/final_report.md` |

Для буквального чтения рубрики: `scrape` и `fetch_api` сохранены и тестируются как общие публичные skills, но не выдаются за источник Steam-данных. Оба настроенных источника пайплайна загружаются из Hugging Face.

## Структура проекта

```text
agents/                    production agents
pipeline/runner.py         последовательная оркестрация и checkpoint HITL
run_pipeline.py            единая CLI-команда
hitl_app.py                Streamlit review UI
notebooks/                 четыре тонких аналитических notebook
data/fixtures/             маленький deterministic smoke fixture
data/raw/                  сбор (создаётся при запуске)
data/processed/            очищенные данные (создаётся при запуске)
data/review/               очередь и исправления (создаётся при запуске)
data/labeled/              финальный датасет и data card (после review)
reports/                   stage/final отчёты (создаются при запуске)
models/                    сохранённая модель (после полного прогона)
tests/                     unit tests
config.yaml                единая конфигурация
requirements.txt           воспроизводимые зависимости
```

## Ограничения и ответственное использование

- Amazon и Steam различаются по лексике, длине, аудитории и механике исходной оценки; это и есть полезный domain shift, но метрики нельзя без проверки переносить на другие домены.
- Положительный/отрицательный рейтинг не всегда совпадает с тональностью текста; сарказм, смешанные отзывы и review bombing создают label noise.
- Confidence Transformer-модели не является гарантированно калиброванной вероятностью.
- Тексты могут содержать персональные данные, токсичность и авторский контент. Перед публикацией полного датасета нужно проверить условия источников и при необходимости редактировать чувствительные поля.
- Для сдачи фиксируйте фактические числа и метрики только из созданных JSON/CSV-отчётов, а не из ожидаемых значений конфигурации.

## Официальные источники

- Hugging Face Datasets — загрузка: https://huggingface.co/docs/datasets/loading
- Hugging Face Datasets — streaming: https://huggingface.co/docs/datasets/stream
- Карточка `mteb/amazon_polarity`: https://huggingface.co/datasets/mteb/amazon_polarity
- Карточка `reapxdev/steam-reviews-scraper`: https://huggingface.co/datasets/reapxdev/steam-reviews-scraper
- Transformers и Apple Silicon/MPS: https://huggingface.co/docs/transformers/perf_train_special
- Transformers pipeline: https://huggingface.co/docs/transformers/pipeline_tutorial
- Streamlit `st.data_editor`: https://docs.streamlit.io/develop/api-reference/data/st.data_editor
- Label Studio — import tasks: https://labelstud.io/guide/tasks
- Label Studio — model predictions: https://labelstud.io/guide/predictions.html
- scikit-learn Logistic Regression: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
