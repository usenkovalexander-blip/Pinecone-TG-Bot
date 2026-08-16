# Pinecone TG Bot

Telegram-бот-помощник, который отвечает **только на основе данных** из векторной базы [Pinecone](https://www.pinecone.io/) (индекс `nemo`). Перед каждым ответом бот ищет топ-5 релевантных записей через `search_vectors()` и передаёт их в контекст LLM.

**Репозиторий:** [github.com/usenkovalexander-blip/Pinecone-TG-Bot](https://github.com/usenkovalexander-blip/Pinecone-TG-Bot)

## Возможности

- Семантический поиск по базе знаний Pinecone
- Ответы строго по найденным записям (без «знаний из интернета»)
- Автоматическое сохранение новых фактов из диалога
- Явное добавление записей через команду `/remember`
- Управление памятью и статистикой через команды бота

## Структура проекта

```
Pinecone-TG-Bot/
├── bot.py           # Telegram-бот (pyTelegramBotAPI)
├── pine.py          # Клиент Pinecone (PineconeClient)
├── test.py          # Пример загрузки и поиска по индексу nemo
├── requirements.txt # Зависимости Python
├── .env.example     # Шаблон переменных окружения
├── .gitignore       # Исключения для Git
└── README.md
```

## Требования

- Python 3.10+
- Аккаунт [Pinecone](https://www.pinecone.io/)
- API-ключ OpenAI (или совместимый прокси)
- Telegram-бот ([@BotFather](https://t.me/BotFather))

## Быстрый старт

### 1. Клонирование репозитория

```powershell
git clone https://github.com/usenkovalexander-blip/Pinecone-TG-Bot.git
cd Pinecone-TG-Bot
```

### 2. Виртуальное окружение

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 3. Настройка переменных окружения

```powershell
copy .env.example .env
```

Откройте `.env` и укажите свои ключи. **Не коммитьте файл `.env` в Git.**

### 4. Загрузка тестовых данных

```powershell
python test.py
```

### 5. Запуск бота

```powershell
python bot.py
```

## Переменные окружения

| Переменная | Обязательная | Описание |
|------------|:------------:|----------|
| `PINECONE_API_KEY` | да | API-ключ Pinecone |
| `PINECONE_INDEX_NAME` | нет | Имя индекса (по умолчанию: `nemo`) |
| `PINECONE_DIMENSION` | нет | Размерность векторов (по умолчанию: `1536`) |
| `OPENAI_KEY` | да | API-ключ OpenAI или прокси |
| `OPENAI_BASE_URL` | нет | URL API (по умолчанию: proxyapi.ru) |
| `OPENAI_CHAT_MODEL` | нет | Модель для ответов (по умолчанию: `gpt-4o-mini`) |
| `TELEGRAM_BOT_TOKEN` | да | Токен Telegram-бота от @BotFather |

## Как это работает

1. Пользователь отправляет сообщение в Telegram.
2. Текст преобразуется в embedding (`text-embedding-3-small`).
3. Через `PineconeClient.search_vectors()` выполняется поиск **топ-5** записей в индексе `nemo`.
4. Найденные записи передаются в LLM как контекст.
5. Модель формирует ответ **только** на основе этих записей.
6. При необходимости новые факты из диалога сохраняются обратно в Pinecone.

Если подходящих записей не найдено, бот сообщает, что в базе `nemo` нет информации по запросу.

## Команды бота

| Команда | Описание |
|---------|----------|
| `/start`, `/help` | Справка |
| `/remember <текст>` | Добавить запись в базу |
| `/memories` | Показать релевантные записи |
| `/forget` | Удалить записи, добавленные вами |
| `/stats` | Статистика индекса Pinecone |
| `/clear` | Очистить историю текущего диалога |

## Модуль pine.py

`PineconeClient` предоставляет методы для работы с Pinecone:

- `create_index()` — создание serverless-индекса
- `connect()` — подключение к индексу
- `upsert()` — запись векторов
- `fetch()` — чтение по ID
- `search()` / `search_vectors()` — семантический поиск
- `delete()` — удаление векторов
- `get_stats()` — статистика индекса

## Пример использования pine.py

```python
from pine import PineconeClient

client = PineconeClient(index_name="nemo")
client.create_index(name="nemo", dimension=1536, connect_after=True)

results = client.search_vectors(
    query_vector=[0.1, 0.2, ...],  # embedding запроса
    top_k=5,
)
```

## Настройки бота

Основные параметры задаются в `bot.py`:

| Параметр | Значение | Описание |
|----------|----------|----------|
| `TOP_K` | `5` | Количество записей для контекста |
| `temperature` | `0.6` | Температура генерации ответов |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Модель для embeddings |
| `KNOWLEDGE_NAMESPACE` | `""` | Namespace индекса nemo |

## Безопасность

### Секреты и ключи

- **Никогда не коммитьте** файл `.env` — он уже добавлен в `.gitignore`.
- Используйте `.env.example` как шаблон без реальных значений.
- Если ключ случайно попал в Git — **немедленно отзовите его** в Pinecone, OpenAI и @BotFather и создайте новый.
- Не публикуйте токены в issues, pull requests и скриншотах.

### Рекомендации для production

- Храните секреты в переменных окружения сервера или в GitHub Secrets (для CI/CD).
- Ограничьте доступ к Pinecone-индексу только нужными API-ключами.
- Регулярно ротируйте API-ключи и токен бота.
- Команда `/forget` удаляет только записи, добавленные конкретным пользователем (по `user_id` в metadata), не затрагивая общую базу знаний.

### Что не попадает в репозиторий

Благодаря `.gitignore` в Git не включаются:

- `.env` и другие файлы с секретами
- `venv/` и виртуальные окружения
- `__pycache__/` и артефакты Python
- файлы IDE и локальные логи

## Разработка

### Проверка перед коммитом

```powershell
git status
```

Убедитесь, что в списке изменений **нет** файла `.env`.

### Публикация изменений

```powershell
git add .
git commit -m "Описание изменений"
git push origin main
```

## Лицензия

MIT — используйте свободно с указанием авторства.
