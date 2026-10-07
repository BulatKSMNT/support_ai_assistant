import math
import re
import threading
from collections import Counter, defaultdict

import snowballstemmer
from qdrant_client import AsyncQdrantClient, models

from app.config import Settings
from app.errors import ServiceError

STOPWORDS = set(
    """а без был была были быть в вам вас ваш ваши вы где да для до его ее её если есть
же за и из или им их к как какой какие какая ко когда который кто ли мы на над не но о об
он она они от по под при про с со так также то у уже что чтобы это этой этот я the a an of
and or to for in is on with подготовь ответ черновик клиент клиента заявка заявке нужно можно
пожалуйста расскажи скажи какие нужно узнать""".split()
)
STEMMERS = threading.local()


def tokenize(text: str) -> list[str]:
    # Snowball instances mutate internal buffers; searches run in a thread pool.
    if not hasattr(STEMMERS, "ru"):
        STEMMERS.ru = snowballstemmer.stemmer("russian")
        STEMMERS.en = snowballstemmer.stemmer("english")
    words = re.findall(r"[a-zа-яё0-9]{2,}", text.lower().replace("ё", "е"))
    return [
        (STEMMERS.ru if re.search("[а-я]", word) else STEMMERS.en).stemWord(word)
        for word in words
        if word not in STOPWORDS
    ]


def lexical_search(query: str, chunks: list[dict], limit: int) -> list[tuple[str, float]]:
    terms = set(tokenize(query))
    if not terms or not chunks:
        return []
    bags = [Counter(tokenize(c["title"] + " " + c["text"])) for c in chunks]
    lengths = [sum(bag.values()) for bag in bags]
    average = sum(lengths) / len(lengths) or 1
    frequency = {term: sum(term in bag for bag in bags) for term in terms}
    ranked = []
    for chunk, bag, length in zip(chunks, bags, lengths, strict=True):
        score = 0.0
        for term in terms:
            count = bag.get(term, 0)
            if count:
                idf = math.log(1 + (len(chunks) - frequency[term] + 0.5) / (frequency[term] + 0.5))
                score += idf * count * 2.5 / (count + 1.5 * (0.25 + 0.75 * length / average))
        if score > 0:
            ranked.append((chunk["id"], score))
    return sorted(ranked, key=lambda item: (-item[1], item[0]))[:limit]


def fuse(*rankings: list[tuple[str, float]]) -> list[tuple[str, float]]:
    scores = defaultdict(float)
    for ranking in rankings:
        for rank, (identifier, _) in enumerate(ranking, 1):
            scores[identifier] += 1 / (60 + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


class VectorIndex:
    def __init__(self, settings: Settings):
        self.settings = settings
        if settings.qdrant_url:
            self.client = AsyncQdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key.get_secret_value()
                if settings.qdrant_api_key
                else None,
                timeout=15,
            )
        else:
            self.client = AsyncQdrantClient(path=str(settings.data_dir / "vectors"))

    async def ensure_collection(self, size: int):
        if not await self.client.collection_exists(self.settings.collection):
            await self.client.create_collection(
                self.settings.collection,
                vectors_config=models.VectorParams(size=size, distance=models.Distance.COSINE),
            )
        else:
            info = await self.client.get_collection(self.settings.collection)
            if info.config.params.vectors.size != size:
                raise ServiceError(
                    "index_dimension_mismatch", "Размерность индекса не совпадает.", 409
                )

    async def upsert(self, chunks: list[dict], embeddings: list[list[float]]):
        try:
            await self.ensure_collection(len(embeddings[0]))
            await self.client.upsert(
                self.settings.collection,
                points=[
                    models.PointStruct(id=c["id"], vector=vector)
                    for c, vector in zip(chunks, embeddings, strict=True)
                ],
                wait=True,
            )
        except ServiceError:
            raise
        except Exception as exc:
            raise ServiceError(
                "index_unavailable", "Не удалось обновить векторный индекс."
            ) from exc

    async def search(self, vector: list[float], active_ids: list[str], limit: int):
        if not active_ids:
            return []
        try:
            result = await self.client.query_points(
                self.settings.collection,
                query=vector,
                query_filter=models.Filter(must=[models.HasIdCondition(has_id=active_ids)]),
                limit=limit,
                with_payload=False,
                score_threshold=self.settings.min_vector_score,
            )
            return [(str(point.id), float(point.score)) for point in result.points]
        except Exception as exc:
            raise ServiceError("index_unavailable", "Векторный поиск недоступен.") from exc

    async def delete(self, identifiers: list[str]):
        if identifiers:
            await self.client.delete(
                self.settings.collection,
                points_selector=models.PointIdsList(points=identifiers),
                wait=True,
            )

    async def ready(self) -> bool:
        try:
            await self.client.get_collections()
            return True
        except Exception:
            return False

    async def close(self):
        await self.client.close()
