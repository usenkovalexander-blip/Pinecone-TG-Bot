from __future__ import annotations

import json
import os
import sys
import uuid
from datetime import datetime, timezone
from typing import Any

import telebot
from dotenv import load_dotenv
from openai import OpenAI
from telebot import types

from pine import PineconeClient

load_dotenv()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "nemo")
EMBEDDING_DIMENSION = int(os.getenv("PINECONE_DIMENSION", "1536"))
TOP_K = 5
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = "text-embedding-3-small"
KNOWLEDGE_NAMESPACE = ""
DUPLICATE_SCORE_THRESHOLD = 0.92
MIN_MEMORY_SCORE = 0.25
NO_DATA_MESSAGE = (
    "В базе данных Pinecone (индекс nemo) не найдено информации по вашему запросу."
)

SYSTEM_PROMPT = """Ты помощник, который отвечает ТОЛЬКО на основе данных из базы Pinecone (индекс nemo).

Строгие правила:
- Используй исключительно факты из блока «Релевантные записи из базы» ниже.
- ЗАПРЕЩЕНО использовать общие знания, интернет или информацию вне базы.
- ЗАПРЕЩЕНО выдумывать факты, которых нет в записях базы.
- Если записей недостаточно для ответа — честно скажи, что в базе nemo нет нужной информации.
- Отвечай на языке пользователя, кратко и по делу.
- Формулируй ответ своими словами, опираясь на найденные записи."""

EXTRACT_MEMORIES_PROMPT = """Проанализируй диалог и извлеки новые полезные факты о пользователе,
которые стоит сохранить для будущих разговоров: имя, предпочтения, даты, контакты,
рабочие детали, привычки, важные события.

Не сохраняй:
- общие вопросы без личной информации;
- то, что уже было в списке существующих воспоминаний;
- временный small talk.

Верни JSON: {"memories": ["факт 1", "факт 2"]}
Если нечего запоминать — {"memories": []}"""

telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
if not telegram_token:
    raise ValueError("TELEGRAM_BOT_TOKEN не задан в .env")

openai_api_key = os.getenv("OPENAI_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_KEY не задан в .env")

bot = telebot.TeleBot(telegram_token, parse_mode="HTML")
openai_client = OpenAI(
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.proxyapi.ru/openai/v1"),
    api_key=openai_api_key,
)
pinecone_client = PineconeClient(index_name=INDEX_NAME)

chat_history: dict[int, list[dict[str, str]]] = {}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_embeddings(texts: list[str]) -> list[list[float]]:
    response = openai_client.embeddings.create(
        input=texts,
        model=EMBEDDING_MODEL,
    )
    return [item.embedding for item in response.data]


def get_embedding(text: str) -> list[float]:
    return get_embeddings([text])[0]


def search_memories(query: str, top_k: int = TOP_K) -> list[dict[str, Any]]:
    query_vector = get_embedding(query)
    results = pinecone_client.search_vectors(
        query_vector=query_vector,
        top_k=top_k,
        namespace=KNOWLEDGE_NAMESPACE,
        include_metadata=True,
    )
    return [item for item in results if item.get("score", 0) >= MIN_MEMORY_SCORE]


def format_memories_for_context(memories: list[dict[str, Any]]) -> str:
    if not memories:
        return "Записей не найдено."

    lines: list[str] = []
    for index, memory in enumerate(memories, start=1):
        metadata = memory.get("metadata") or {}
        text = metadata.get("text", "—")
        category = metadata.get("category")
        score = memory.get("score", 0)
        category_suffix = f" [категория: {category}]" if category else ""
        lines.append(f"{index}. {text}{category_suffix} [релевантность: {score:.2f}]")
    return "\n".join(lines)


def save_memory(text: str, user_id: int) -> bool:
    text = text.strip()
    if not text:
        return False

    existing = search_memories(text, top_k=1)
    if existing:
        top = existing[0]
        if top.get("score", 0) >= DUPLICATE_SCORE_THRESHOLD:
            existing_text = (top.get("metadata") or {}).get("text", "")
            if existing_text.strip().lower() == text.lower():
                return False

    vector = {
        "id": str(uuid.uuid4()),
        "values": get_embedding(text),
        "metadata": {
            "text": text,
            "user_id": str(user_id),
            "created_at": now_iso(),
        },
    }
    pinecone_client.upsert(vectors=[vector], namespace=KNOWLEDGE_NAMESPACE)
    return True


def extract_new_memories(
    user_message: str,
    assistant_response: str,
    existing_memories: list[dict[str, Any]],
) -> list[str]:
    existing_texts = [
        (item.get("metadata") or {}).get("text", "")
        for item in existing_memories
    ]
    existing_block = "\n".join(f"- {text}" for text in existing_texts if text) or "—"

    response = openai_client.chat.completions.create(
        model=CHAT_MODEL,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": EXTRACT_MEMORIES_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Существующие воспоминания:\n{existing_block}\n\n"
                    f"Пользователь: {user_message}\n"
                    f"Ассистент: {assistant_response}"
                ),
            },
        ],
        temperature=0.2,
    )

    content = response.choices[0].message.content or "{}"
    payload = json.loads(content)
    memories = payload.get("memories", [])
    if not isinstance(memories, list):
        return []

    return [str(item).strip() for item in memories if str(item).strip()]


def generate_response(
    user_message: str,
    user_id: int,
    memories: list[dict[str, Any]],
) -> str:
    if not memories:
        return NO_DATA_MESSAGE

    memory_context = format_memories_for_context(memories)
    history = chat_history.get(user_id, [])[-6:]

    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "system",
            "content": (
                f"Релевантные записи из базы Pinecone nemo (топ-{TOP_K}):\n"
                f"{memory_context}"
            ),
        },
    ]
    messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    response = openai_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.6,
    )
    return (response.choices[0].message.content or "").strip()


def remember_user_message(user_id: int, role: str, content: str) -> None:
    history = chat_history.setdefault(user_id, [])
    history.append({"role": role, "content": content})
    chat_history[user_id] = history[-12:]


def send_long_message(chat_id: int, text: str) -> None:
    if len(text) <= 4096:
        bot.send_message(chat_id, text)
        return

    chunk = ""
    for line in text.split("\n"):
        if len(chunk) + len(line) + 1 > 4096:
            bot.send_message(chat_id, chunk)
            chunk = line
        else:
            chunk = f"{chunk}\n{line}" if chunk else line
    if chunk:
        bot.send_message(chat_id, chunk)


def init_pinecone() -> None:
    pinecone_client.create_index(
        name=INDEX_NAME,
        dimension=EMBEDDING_DIMENSION,
        connect_after=True,
    )


@bot.message_handler(commands=["start", "help"])
def handle_start(message: types.Message) -> None:
    send_long_message(
        message.chat.id,
        (
            "Привет! Я помощник по базе знаний Pinecone (индекс <b>nemo</b>).\n\n"
            "Перед каждым ответом я ищу топ-{top_k} релевантных записей через "
            "<code>search_vectors()</code> и отвечаю <b>только</b> на их основе.\n\n"
            "<b>Команды:</b>\n"
            "/remember &lt;текст&gt; — добавить запись в базу\n"
            "/memories — показать релевантные записи\n"
            "/forget — удалить записи, добавленные тобой\n"
            "/stats — статистика базы\n"
            "/clear — очистить историю текущего диалога"
        ).format(top_k=TOP_K),
    )


@bot.message_handler(commands=["remember"])
def handle_remember(message: types.Message) -> None:
    text = (message.text or "").partition(" ")[2].strip()
    if not text:
        bot.reply_to(message, "Укажи текст: /remember Мой любимый цвет — синий")
        return

    saved = save_memory(text, message.from_user.id)
    if saved:
        bot.reply_to(message, f"Запомнил: {text}")
    else:
        bot.reply_to(message, "Такое воспоминание уже есть в базе.")


@bot.message_handler(commands=["memories"])
def handle_memories(message: types.Message) -> None:
    bot.send_chat_action(message.chat.id, "typing")
    query = "все записи в базе знаний"
    memories = search_memories(query, top_k=TOP_K)

    if not memories:
        bot.reply_to(message, "В базе nemo пока нет записей по этому запросу.")
        return

    lines = ["<b>Записи из базы nemo:</b>"]
    for index, memory in enumerate(memories, start=1):
        metadata = memory.get("metadata") or {}
        text = metadata.get("text", "—")
        score = memory.get("score", 0)
        lines.append(f"{index}. {text} <i>({score:.2f})</i>")

    send_long_message(message.chat.id, "\n".join(lines))


@bot.message_handler(commands=["forget"])
def handle_forget(message: types.Message) -> None:
    user_id = str(message.from_user.id)
    pinecone_client.delete(
        namespace=KNOWLEDGE_NAMESPACE,
        metadata_filter={"user_id": {"$eq": user_id}},
    )
    chat_history.pop(message.from_user.id, None)
    bot.reply_to(message, "Твои добавленные записи удалены из базы nemo.")


@bot.message_handler(commands=["clear"])
def handle_clear(message: types.Message) -> None:
    chat_history.pop(message.from_user.id, None)
    bot.reply_to(message, "История текущего диалога очищена.")


@bot.message_handler(commands=["stats"])
def handle_stats(message: types.Message) -> None:
    stats = pinecone_client.get_stats()
    bot.reply_to(
        message,
        (
            f"Векторов в индексе: {stats['total_vector_count']}\n"
            f"Размерность: {stats['dimension']}"
        ),
    )


@bot.message_handler(func=lambda message: message.content_type == "text")
def handle_text(message: types.Message) -> None:
    user_text = (message.text or "").strip()
    if not user_text:
        return

    user_id = message.from_user.id
    bot.send_chat_action(message.chat.id, "typing")

    try:
        memories = search_memories(user_text, top_k=TOP_K)
        answer = generate_response(user_text, user_id, memories)

        remember_user_message(user_id, "user", user_text)
        remember_user_message(user_id, "assistant", answer)

        new_memories = extract_new_memories(user_text, answer, memories)
        saved_count = sum(1 for item in new_memories if save_memory(item, user_id))

        if saved_count:
            answer += f"\n\n<i>💾 Сохранил {saved_count} новое воспоминание.</i>"

        send_long_message(message.chat.id, answer)
    except Exception as exc:
        bot.reply_to(
            message,
            f"Произошла ошибка при обработке сообщения: {exc}",
        )


def main() -> None:
    print(f"Запуск бота. Индекс Pinecone: {INDEX_NAME}")
    init_pinecone()
    stats = pinecone_client.get_stats()
    print(f"Pinecone готов. Векторов: {stats['total_vector_count']}")
    print("Бот слушает сообщения...")
    bot.infinity_polling(timeout=60, long_polling_timeout=60)


if __name__ == "__main__":
    main()
