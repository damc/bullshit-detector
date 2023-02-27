from dataclasses import dataclass
from typing import Any, List, Callable

from openai.embeddings_utils import cosine_similarity

from .basic import Embedding, embedding


@dataclass
class SearchItem:
    item: Any
    embedding: Embedding


@dataclass
class SearchResult:
    item: Any
    similarity: float


def search_by_embedding(
        embedding_: Embedding,
        items: List[SearchItem],
        min_similarity: float = 0
) -> List[SearchResult]:
    results = []
    for item in items:
        similarity = cosine_similarity(item.embedding, embedding_)
        if similarity > min_similarity:
            results.append(SearchResult(item.item, similarity))
    return sorted(results, key=lambda result: -result.similarity)


def search_text_by_text(query: str, items: List[str]) -> List[SearchResult]:
    items = [
        SearchItem(item, text_document_for_text_search_embedding(item))
        for item in items
    ]
    query_embedding = text_query_for_text_search_embedding(query)
    return search_by_embedding(query_embedding, items)


def text_query_for_text_search_embedding(text: str, **kwargs) -> Embedding:
    return embedding('text-search-babbage-query-001', text, **kwargs)


def text_document_for_text_search_embedding(text: str, **kwargs) -> Embedding:
    return embedding('text-search-babbage-doc-001', text, **kwargs)


def search_code_by_text(
        text: str,
        items: List[Any],
        key: Callable = lambda item: item
) -> List[SearchResult]:
    items = [
        SearchItem(item, code_document_for_code_search_embedding(key(item)))
        for item in items
    ]
    query_embedding = text_query_for_code_search_embedding(text)
    return search_by_embedding(query_embedding, items)


def text_query_for_code_search_embedding(text: str, **kwargs) -> Embedding:
    return embedding('code-search-babbage-text-001', text, **kwargs)


def code_document_for_code_search_embedding(code: str, **kwargs) -> Embedding:
    return embedding('code-search-babbage-code-001', code, **kwargs)


def classify_text(text: str, classes: List[str]) -> str:
    if not classes:
        raise ValueError("There must be at least one class")
    results = search_text_by_text(text, classes)
    return results[0].item
