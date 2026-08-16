from __future__ import annotations

import os
import time
from typing import Any, Callable, TypeVar

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from pinecone.errors.exceptions import PineconeConnectionError, PineconeTimeoutError

load_dotenv()

T = TypeVar("T")


class PineconeClient:
    """Клиент для работы с векторной базой данных Pinecone."""

    def __init__(
        self,
        api_key: str | None = None,
        index_name: str | None = None,
    ) -> None:
        """
        Инициализирует клиент Pinecone.

        Args:
            api_key: API-ключ Pinecone. Если не передан, берётся из переменной
                окружения PINECONE_API_KEY.
            index_name: Имя индекса для подключения. Можно указать позже в connect().

        Raises:
            ValueError: Если API-ключ не найден.
        """
        self.api_key = api_key or os.getenv("PINECONE_API_KEY")
        if not self.api_key:
            raise ValueError("PINECONE_API_KEY не задан в окружении или аргументах.")

        self.index_name = index_name
        self._client: Pinecone | None = None
        self._index = None

    def _ensure_client(self) -> Pinecone:
        """
        Создаёт клиент Pinecone, если он ещё не инициализирован.

        Returns:
            Pinecone: Активный control-plane клиент Pinecone.
        """
        if self._client is None:
            self._client = Pinecone(api_key=self.api_key, timeout=20.0)
        return self._client

    def _wait_for_index_ready(self, name: str, timeout: int = 300) -> None:
        """
        Ожидает, пока индекс станет доступен для операций с векторами.

        Args:
            name: Имя индекса.
            timeout: Максимальное время ожидания в секундах.

        Returns:
            None
        """
        client = self._ensure_client()
        start = time.monotonic()

        while True:
            index_model = client.indexes.describe(name)
            if index_model.status.ready:
                return
            if index_model.status.state == "InitializationFailed":
                raise RuntimeError(f"Индекс '{name}' не удалось инициализировать.")
            if timeout is not None and time.monotonic() - start >= timeout:
                raise PineconeTimeoutError(f"Индекс '{name}' не готов после {timeout} с.")
            time.sleep(5)

    def _run_with_retry(
        self,
        operation: Callable[[], T],
        max_attempts: int = 5,
        delay: float = 3.0,
    ) -> T:
        """
        Выполняет операцию с повторными попытками при сетевых ошибках.

        Args:
            operation: Callable без аргументов, выполняющий запрос к Pinecone.
            max_attempts: Максимальное число попыток.
            delay: Базовая пауза между попытками в секундах.

        Returns:
            T: Результат успешного вызова operation().
        """
        last_error: Exception | None = None

        for attempt in range(1, max_attempts + 1):
            try:
                return operation()
            except (PineconeConnectionError, TimeoutError, OSError) as exc:
                last_error = exc
                if attempt == max_attempts:
                    break
                time.sleep(delay * attempt)
                if self.index_name:
                    self._index = None
                    self.connect(self.index_name)

        raise last_error  # type: ignore[misc]

    def create_index(
        self,
        name: str,
        dimension: int,
        metric: str = "cosine",
        cloud: str = "aws",
        region: str = "us-east-1",
        connect_after: bool = False,
    ) -> dict[str, str | bool]:
        """
        Создаёт serverless-индекс Pinecone, если он ещё не существует.

        Args:
            name: Имя создаваемого индекса.
            dimension: Размерность векторов, например 1536 для text-embedding-3-small.
            metric: Метрика сходства: cosine, euclidean или dotproduct.
            cloud: Облачный провайдер, например aws или gcp.
            region: Регион размещения индекса, например us-east-1.
            connect_after: Если True, после создания автоматически вызывает connect().

        Returns:
            dict: Информация об операции:
                {
                    "name": str,
                    "created": bool,
                    "status": "ready" | "already_exists",
                }

        Raises:
            ValueError: Если имя индекса или dimension не указаны.
        """
        if not name:
            raise ValueError("Имя индекса не указано.")
        if dimension <= 0:
            raise ValueError("dimension должна быть положительным числом.")

        client = self._ensure_client()
        if client.indexes.exists(name):
            result = {"name": name, "created": False, "status": "already_exists"}
            self._wait_for_index_ready(name)
        else:
            client.indexes.create(
                name=name,
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(cloud=cloud, region=region),
            )
            result = {"name": name, "created": True, "status": "ready"}

        if connect_after:
            self.connect(name)

        return result

    def connect(self, index_name: str | None = None) -> None:
        """
        Подключается к индексу Pinecone.

        Args:
            index_name: Имя индекса. Если не передан, используется index_name
                из конструктора.

        Returns:
            None

        Raises:
            ValueError: Если имя индекса не указано.
        """
        target_index = index_name or self.index_name
        if not target_index:
            raise ValueError("Имя индекса не указано.")

        self.index_name = target_index
        client = self._ensure_client()
        self._wait_for_index_ready(target_index)
        index_model = client.indexes.describe(target_index)
        self._index = client.index(host=index_model.host)

    @property
    def index(self):
        """
        Возвращает активное подключение к индексу.

        Returns:
            Index: Клиент индекса Pinecone.

        Raises:
            RuntimeError: Если connect() ещё не был вызван.
        """
        if self._index is None:
            raise RuntimeError("Сначала вызовите connect() для подключения к индексу.")
        return self._index

    def get_stats(self) -> dict[str, Any]:
        """
        Возвращает статистику индекса.

        Returns:
            dict: Количество векторов и размерность индекса, например
                {"total_vector_count": 3, "dimension": 1536}.
        """
        response = self._run_with_retry(lambda: self.index.describe_index_stats())
        return {
            "total_vector_count": response.total_vector_count,
            "dimension": response.dimension,
        }

    def upsert(
        self,
        vectors: list[dict[str, Any] | tuple],
        namespace: str = "",
    ) -> dict[str, int]:
        """
        Записывает или обновляет векторы в индексе.

        Args:
            vectors: Список векторов. Каждый элемент может быть:
                - словарём {"id": str, "values": list[float], "metadata": dict | None};
                - кортежем (id, values) или (id, values, metadata).
            namespace: Логическое пространство имён внутри индекса.

        Returns:
            dict: Словарь с количеством записанных векторов, например
                {"upserted_count": 3}.
        """
        prepared_vectors = [self._normalize_vector(vector) for vector in vectors]
        total_upserted = 0

        for vector in prepared_vectors:
            response = self._run_with_retry(
                lambda v=vector: self.index.upsert(vectors=[v], namespace=namespace)
            )
            total_upserted += response.upserted_count

        return {"upserted_count": total_upserted}

    def fetch(
        self,
        ids: list[str],
        namespace: str = "",
    ) -> dict[str, dict[str, Any]]:
        """
        Читает векторы из индекса по их идентификаторам.

        Args:
            ids: Список ID векторов для чтения.
            namespace: Логическое пространство имён внутри индекса.

        Returns:
            dict: Словарь найденных векторов вида
                {
                    "vector_id": {
                        "id": str,
                        "values": list[float],
                        "metadata": dict,
                    }
                }
                Отсутствующие ID в результат не включаются.
        """
        response = self._run_with_retry(
            lambda: self.index.fetch(ids=ids, namespace=namespace)
        )
        return {
            vector_id: {
                "id": vector.id,
                "values": vector.values,
                "metadata": vector.metadata or {},
            }
            for vector_id, vector in response.vectors.items()
        }

    def search(
        self,
        vector: list[float],
        top_k: int = 5,
        namespace: str = "",
        metadata_filter: dict[str, Any] | None = None,
        include_metadata: bool = True,
        include_values: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Ищет наиболее похожие векторы в индексе.

        Args:
            vector: Вектор запроса (список float).
            top_k: Количество ближайших результатов.
            namespace: Логическое пространство имён внутри индекса.
            metadata_filter: Фильтр по метаданным, например
                {"category": {"$eq": "food"}}.
            include_metadata: Включать ли метаданные в ответ.
            include_values: Включать ли значения векторов в ответ.

        Returns:
            list[dict]: Список найденных векторов, отсортированный по убыванию
                сходства. Каждый элемент имеет вид:
                {
                    "id": str,
                    "score": float,
                    "metadata": dict | None,
                    "values": list[float] | None,
                }
        """
        query_vector = [round(float(value), 6) for value in vector]
        response = self._run_with_retry(
            lambda: self.index.query(
                vector=query_vector,
                top_k=top_k,
                namespace=namespace,
                filter=metadata_filter,
                include_metadata=include_metadata,
                include_values=include_values,
            )
        )

        return [
            {
                "id": match.id,
                "score": match.score,
                "metadata": match.metadata if include_metadata else None,
                "values": match.values if include_values else None,
            }
            for match in response.matches
        ]

    def search_vectors(
        self,
        query_vector: list[float],
        top_k: int = 5,
        namespace: str = "",
        metadata_filter: dict[str, Any] | None = None,
        include_metadata: bool = True,
        include_values: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Ищет наиболее похожие векторы в индексе по query_vector.

        Args:
            query_vector: Вектор запроса (список float).
            top_k: Количество ближайших результатов.
            namespace: Логическое пространство имён внутри индекса.
            metadata_filter: Фильтр по метаданным.
            include_metadata: Включать ли метаданные в ответ.
            include_values: Включать ли значения векторов в ответ.

        Returns:
            list[dict]: Список найденных векторов, отсортированный по убыванию сходства.
        """
        return self.search(
            vector=query_vector,
            top_k=top_k,
            namespace=namespace,
            metadata_filter=metadata_filter,
            include_metadata=include_metadata,
            include_values=include_values,
        )

    def delete(
        self,
        ids: list[str] | None = None,
        namespace: str = "",
        metadata_filter: dict[str, Any] | None = None,
        delete_all: bool = False,
    ) -> dict[str, str]:
        """
        Удаляет векторы из индекса.

        Args:
            ids: Список ID векторов для удаления.
            namespace: Логическое пространство имён внутри индекса.
            metadata_filter: Фильтр по метаданным для массового удаления.
            delete_all: Удалить все векторы в namespace. Используйте с осторожностью.

        Returns:
            dict: Статус операции, например {"status": "deleted"}.

        Raises:
            ValueError: Если не указан ни один способ удаления.
        """
        if not delete_all and not ids and not metadata_filter:
            raise ValueError("Укажите ids, metadata_filter или delete_all=True.")

        self._run_with_retry(
            lambda: self.index.delete(
                ids=ids,
                namespace=namespace,
                filter=metadata_filter,
                delete_all=delete_all,
            )
        )
        return {"status": "deleted"}

    @staticmethod
    def _normalize_vector(vector: dict[str, Any] | tuple) -> dict[str, Any]:
        """
        Приводит вектор к формату, поддерживаемому Pinecone SDK.

        Args:
            vector: Вектор в виде словаря или кортежа.

        Returns:
            dict: Нормализованное представление вектора.
        """
        if isinstance(vector, dict):
            normalized: dict[str, Any] = {
                "id": vector["id"],
                "values": [round(float(value), 6) for value in vector["values"]],
            }
            if vector.get("metadata") is not None:
                normalized["metadata"] = vector["metadata"]
            return normalized

        if isinstance(vector, tuple):
            normalized = {
                "id": vector[0],
                "values": [round(float(value), 6) for value in vector[1]],
            }
            if len(vector) > 2:
                normalized["metadata"] = vector[2]
            return normalized

        raise TypeError("Вектор должен быть dict или tuple.")


def main() -> None:
    """
    Демонстрация работы PineconeClient при запуске файла напрямую.

    Returns:
        None
    """
    index_name = os.getenv("PINECONE_INDEX_NAME", "food-embeddings")
    dimension = int(os.getenv("PINECONE_DIMENSION", "1536"))

    print("pine.py — модуль с классом PineconeClient")
    print(f"Проверка подключения к индексу: {index_name}")

    client = PineconeClient(index_name=index_name)
    index_info = client.create_index(
        name=index_name,
        dimension=dimension,
        connect_after=True,
    )
    print(f"Индекс: {index_info}")
    print("Подключение успешно. Для полного примера запустите: python test.py")


if __name__ == "__main__":
    main()
