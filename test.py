import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

from pine import PineconeClient

load_dotenv()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

INDEX_NAME = "nemo"
EMBEDDING_DIMENSION = 1536
UPLOAD_LIMIT = 20

TEST_ITEMS = [
    {"id": "pizza-pepperoni", "text": "Пицца Пепперони", "category": "food"},
    {"id": "pizza-margherita", "text": "Пицца Маргарита", "category": "food"},
    {"id": "pizza-four-cheese", "text": "Пицца Четыре сыра", "category": "food"},
    {"id": "pizza-hawaiian", "text": "Пицца Гавайская", "category": "food"},
    {"id": "pasta-carbonara", "text": "Паста Карбонара", "category": "food"},
    {"id": "pasta-bolognese", "text": "Паста Болоньезе", "category": "food"},
    {"id": "pasta-pesto", "text": "Паста с песто", "category": "food"},
    {"id": "pasta-alfredo", "text": "Паста Альфредо", "category": "food"},
    {"id": "salad-caesar", "text": "Салат Цезарь", "category": "food"},
    {"id": "salad-greek", "text": "Греческий салат", "category": "food"},
    {"id": "salad-vegetable", "text": "Овощной салат", "category": "food"},
    {"id": "burger-classic", "text": "Классический бургер", "category": "food"},
    {"id": "burger-cheese", "text": "Чизбургер", "category": "food"},
    {"id": "burger-chicken", "text": "Куриный бургер", "category": "food"},
    {"id": "sushi-california", "text": "Ролл Калифорния", "category": "food"},
    {"id": "sushi-philadelphia", "text": "Ролл Филадельфия", "category": "food"},
    {"id": "sushi-salmon", "text": "Суши с лососем", "category": "food"},
    {"id": "soup-borscht", "text": "Борщ", "category": "food"},
    {"id": "soup-chicken", "text": "Куриный суп", "category": "food"},
    {"id": "soup-tomato", "text": "Томатный суп", "category": "food"},
    {"id": "soup-mushroom", "text": "Грибной крем-суп", "category": "food"},
    {"id": "steak-beef", "text": "Стейк из говядины", "category": "food"},
    {"id": "steak-pork", "text": "Свиная отбивная", "category": "food"},
    {"id": "chicken-grilled", "text": "Курица гриль", "category": "food"},
    {"id": "fish-grilled", "text": "Рыба на гриле", "category": "food"},
    {"id": "shrimp-fried", "text": "Креветки во фритюре", "category": "food"},
    {"id": "rice-fried", "text": "Жареный рис", "category": "food"},
    {"id": "rice-pilaf", "text": "Плов", "category": "food"},
    {"id": "noodles-ramen", "text": "Рамен", "category": "food"},
    {"id": "noodles-wok", "text": "Лапша вок", "category": "food"},
    {"id": "sandwich-club", "text": "Клубный сэндвич", "category": "food"},
    {"id": "sandwich-ham", "text": "Сэндвич с ветчиной", "category": "food"},
    {"id": "breakfast-omelette", "text": "Омлет", "category": "food"},
    {"id": "breakfast-pancakes", "text": "Блины", "category": "food"},
    {"id": "breakfast-waffles", "text": "Вафли", "category": "food"},
    {"id": "dessert-cheesecake", "text": "Чизкейк", "category": "food"},
    {"id": "dessert-tiramisu", "text": "Тирамису", "category": "food"},
    {"id": "dessert-ice-cream", "text": "Мороженое", "category": "food"},
    {"id": "dessert-brownie", "text": "Брауни", "category": "food"},
    {"id": "side-fries", "text": "Картофель фри", "category": "food"},
    {"id": "side-mashed-potato", "text": "Пюре", "category": "food"},
    {"id": "coffee-latte", "text": "Кофе латте", "category": "drink"},
    {"id": "coffee-espresso", "text": "Эспрессо", "category": "drink"},
    {"id": "coffee-cappuccino", "text": "Капучино", "category": "drink"},
    {"id": "tea-green", "text": "Зелёный чай", "category": "drink"},
    {"id": "tea-black", "text": "Чёрный чай", "category": "drink"},
    {"id": "juice-orange", "text": "Апельсиновый сок", "category": "drink"},
    {"id": "juice-apple", "text": "Яблочный сок", "category": "drink"},
    {"id": "smoothie-berry", "text": "Ягодный смузи", "category": "drink"},
    {"id": "water-sparkling", "text": "Газированная вода", "category": "drink"},
]

openai_client = OpenAI(
    base_url="https://api.proxyapi.ru/openai/v1",
    api_key=os.getenv("OPENAI_KEY"),
)


def get_embeddings(texts: list[str]) -> list[list[float]]:
    response = openai_client.embeddings.create(
        input=texts,
        model="text-embedding-3-small",
    )
    return [item.embedding for item in response.data]


def get_embedding(text: str) -> list[float]:
    return get_embeddings([text])[0]


def search(query: str, top_k: int = 5) -> list[dict]:
    query_vector = get_embedding(query)
    return pinecone_client.search_vectors(query_vector=query_vector, top_k=top_k)


pinecone_client = PineconeClient(index_name=INDEX_NAME)

pinecone_client.create_index(
    name=INDEX_NAME,
    dimension=EMBEDDING_DIMENSION,
    connect_after=True,
)

stats = pinecone_client.get_stats()
if stats["total_vector_count"] < UPLOAD_LIMIT:
    items_to_upload = TEST_ITEMS[:UPLOAD_LIMIT]
    phrases = [item["text"] for item in items_to_upload]
    embeddings = get_embeddings(phrases)

    for number, (phrase, values) in enumerate(zip(phrases, embeddings), start=1):
        vector = {
            "id": str(number),
            "values": values,
            "metadata": {"text": phrase},
        }
        pinecone_client.upsert(vectors=[vector])

query_text = "Какие пиццы у нас есть."
search_results = search(query_text)

print(f"Поиск по запросу: '{query_text}'")
if not search_results:
    print("Совпадений не найдено.")
else:
    for result in search_results:
        metadata = result.get("metadata") or {}
        text = metadata.get("text", "—")
        print(
            f"- id={result['id']}, score={result['score']:.4f}, text={text}"
        )
