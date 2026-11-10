from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from .config import HybridConfig
from .errors import ValidationError


def reciprocal_rank_fusion(
    dense: list[tuple[str, float]],
    bm25: list[tuple[str, float]],
    config: HybridConfig | None = None,
    *,
    k: int = 100,
) -> list[tuple[str, float]]:
    """Fuse truncated candidate lists; absent documents contribute zero."""
    config = config or HybridConfig()
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValidationError("Fusion k must be a positive integer")
    scores: dict[str, float] = {}
    for ranking, weight in ((dense, config.dense_weight), (bm25, config.bm25_weight)):
        seen = set()
        for rank, (chunk_id, _) in enumerate(ranking[: config.candidate_k], 1):
            if chunk_id in seen:
                raise ValidationError(f"Duplicate chunk ID in fusion branch: {chunk_id}")
            seen.add(chunk_id)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + weight / (config.rank_constant + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:k]


class DenseIndex:
    """Keep corpus metadata once while scoring query batches against a memory map."""

    def __init__(
        self,
        vectors: np.ndarray,
        ids: list[str],
        languages: list[str],
        *,
        document_ids: list[str] | None = None,
    ):
        if vectors.ndim != 2 or len(vectors) != len(ids) or len(ids) != len(languages):
            raise ValidationError("Document embedding metadata length/shape mismatch")
        if len(ids) != len(set(ids)):
            raise ValidationError("Duplicate document embedding IDs")
        self.vectors = vectors
        self.ids = np.asarray(ids)
        self.languages = np.asarray(languages)
        self.document_ids = None
        if document_ids is not None:
            if len(document_ids) != len(ids) or any(
                not isinstance(value, str) or not value for value in document_ids
            ):
                raise ValidationError("Document IDs must identify every chunk's source unit")
            document_languages = {}
            for document_id, language in zip(document_ids, languages, strict=True):
                if document_languages.setdefault(document_id, language) != language:
                    raise ValidationError(
                        "A document ID cannot refer to multiple language versions"
                    )
            self.document_ids = np.asarray(document_ids)

    def search(
        self,
        queries: np.ndarray,
        target_languages: tuple[str, ...],
        *,
        k: int = 100,
        query_batch_size: int = 16,
        corpus_block_size: int = 8192,
    ) -> Iterator[list[tuple[str, float]]]:
        yield from _search(
            queries,
            self.vectors,
            self.ids,
            self.languages,
            target_languages,
            k=k,
            query_batch_size=query_batch_size,
            corpus_block_size=corpus_block_size,
        )

    def search_documents(
        self,
        queries: np.ndarray,
        target_languages: tuple[str, ...],
        *,
        k: int = 100,
        query_batch_size: int = 16,
        corpus_block_size: int = 8192,
    ) -> Iterator[list[tuple[str, str, float]]]:
        """Rank unique source language versions by maximum chunk similarity.

        Tuples contain (document_id, winning_chunk_id, score). Every searched
        chunk participates before top-k document pruning; a long document
        therefore cannot consume multiple candidate slots. Tied documents use
        document ID order, and tied winning chunks use chunk ID order.
        """
        if self.document_ids is None:
            raise ValidationError("Document search requires a source document ID for every chunk")
        yield from _search_documents(
            queries,
            self.vectors,
            self.ids,
            self.document_ids,
            self.languages,
            target_languages,
            k=k,
            query_batch_size=query_batch_size,
            corpus_block_size=corpus_block_size,
        )


def exact_topk(
    query_vectors: np.ndarray,
    document_vectors: np.ndarray,
    document_ids: list[str],
    document_languages: list[str],
    target_languages: tuple[str, ...],
    *,
    k: int = 100,
    query_batch_size: int = 16,
    corpus_block_size: int = 8192,
) -> Iterator[list[tuple[str, float]]]:
    """Exact cosine ranking of normalized vectors; bounded score-matrix memory.

    Lexicographic chunk-ID ties are resolved before each pruning operation, so
    ties at a block/top-k boundary are deterministic, not just presentation ties.
    """
    index = DenseIndex(document_vectors, document_ids, document_languages)
    yield from index.search(
        query_vectors,
        target_languages,
        k=k,
        query_batch_size=query_batch_size,
        corpus_block_size=corpus_block_size,
    )


def _search(
    query_vectors: np.ndarray,
    document_vectors: np.ndarray,
    ids: np.ndarray,
    languages: np.ndarray,
    target_languages: tuple[str, ...],
    *,
    k: int,
    query_batch_size: int,
    corpus_block_size: int,
) -> Iterator[list[tuple[str, float]]]:
    if query_vectors.ndim != 2 or query_vectors.shape[1] != document_vectors.shape[1]:
        raise ValidationError("Query/document vectors must be matrices with matching dimensions")
    if min(k, query_batch_size, corpus_block_size) < 1:
        raise ValidationError("Retrieval k and batch/block sizes must be positive")
    for query_start in range(0, len(query_vectors), query_batch_size):
        queries = np.asarray(
            query_vectors[query_start : query_start + query_batch_size], dtype=np.float32
        )
        if not np.isfinite(queries).all():
            raise ValidationError("Query embeddings contain nonfinite values")
        best_ids = [np.array([], dtype=ids.dtype) for _ in queries]
        best_scores = [np.array([], dtype=np.float32) for _ in queries]
        for start in range(0, len(document_vectors), corpus_block_size):
            stop = min(start + corpus_block_size, len(document_vectors))
            mask = np.isin(languages[start:stop], target_languages)
            if not mask.any():
                continue
            vectors = np.asarray(document_vectors[start:stop][mask], dtype=np.float32)
            if not np.isfinite(vectors).all():
                raise ValidationError("Document embeddings contain nonfinite values")
            block_ids = ids[start:stop][mask]
            scores = queries @ vectors.T
            for index, score_row in enumerate(scores):
                order = np.lexsort((block_ids, -score_row))[:k]
                candidates = np.concatenate((best_ids[index], block_ids[order]))
                candidate_scores = np.concatenate((best_scores[index], score_row[order]))
                keep = np.lexsort((candidates, -candidate_scores))[:k]
                best_ids[index], best_scores[index] = candidates[keep], candidate_scores[keep]
        for selected_ids, selected_scores in zip(best_ids, best_scores, strict=True):
            yield [
                (str(chunk_id), float(score))
                for chunk_id, score in zip(selected_ids, selected_scores, strict=True)
            ]


def _document_topk(
    document_ids: np.ndarray, chunk_ids: np.ndarray, scores: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Merge document maxima before pruning, with stable provenance for ties."""
    order = np.lexsort((chunk_ids, -scores, document_ids))
    ordered_documents = document_ids[order]
    unique = np.r_[True, ordered_documents[1:] != ordered_documents[:-1]]
    winners = order[unique]
    keep = winners[np.lexsort((document_ids[winners], -scores[winners]))[:k]]
    return document_ids[keep], chunk_ids[keep], scores[keep]


def _search_documents(
    query_vectors: np.ndarray,
    chunk_vectors: np.ndarray,
    chunk_ids: np.ndarray,
    document_ids: np.ndarray,
    languages: np.ndarray,
    target_languages: tuple[str, ...],
    *,
    k: int,
    query_batch_size: int,
    corpus_block_size: int,
) -> Iterator[list[tuple[str, str, float]]]:
    if query_vectors.ndim != 2 or query_vectors.shape[1] != chunk_vectors.shape[1]:
        raise ValidationError("Query/document vectors must be matrices with matching dimensions")
    if min(k, query_batch_size, corpus_block_size) < 1:
        raise ValidationError("Retrieval k and batch/block sizes must be positive")
    for query_start in range(0, len(query_vectors), query_batch_size):
        queries = np.asarray(
            query_vectors[query_start : query_start + query_batch_size], dtype=np.float32
        )
        if not np.isfinite(queries).all():
            raise ValidationError("Query embeddings contain nonfinite values")
        best = [
            (
                np.array([], dtype=document_ids.dtype),
                np.array([], dtype=chunk_ids.dtype),
                np.array([], dtype=np.float32),
            )
            for _ in queries
        ]
        for start in range(0, len(chunk_vectors), corpus_block_size):
            stop = min(start + corpus_block_size, len(chunk_vectors))
            mask = np.isin(languages[start:stop], target_languages)
            if not mask.any():
                continue
            vectors = np.asarray(chunk_vectors[start:stop][mask], dtype=np.float32)
            if not np.isfinite(vectors).all():
                raise ValidationError("Document embeddings contain nonfinite values")
            block_documents = document_ids[start:stop][mask]
            block_chunks = chunk_ids[start:stop][mask]
            # Sort static metadata once per block. Within each source group,
            # lexical chunk order makes the first maximum the canonical winner.
            group_order = np.lexsort((block_chunks, block_documents))
            ordered_documents = block_documents[group_order]
            starts = np.r_[0, np.flatnonzero(ordered_documents[1:] != ordered_documents[:-1]) + 1]
            sizes = np.diff(np.r_[starts, len(group_order)])
            unique_documents = ordered_documents[starts]
            positions = np.arange(len(group_order))
            for index, score_row in enumerate(queries @ vectors.T):
                ordered_scores = score_row[group_order]
                maxima = np.maximum.reduceat(ordered_scores, starts)
                winner_positions = np.minimum.reduceat(
                    np.where(
                        ordered_scores == np.repeat(maxima, sizes), positions, len(group_order)
                    ),
                    starts,
                )
                winning_chunks = block_chunks[group_order[winner_positions]]
                keep = np.lexsort((unique_documents, -maxima))[:k]
                best[index] = _document_topk(
                    np.concatenate((best[index][0], unique_documents[keep])),
                    np.concatenate((best[index][1], winning_chunks[keep])),
                    np.concatenate((best[index][2], maxima[keep])),
                    k,
                )
        for selected_documents, selected_chunks, selected_scores in best:
            yield [
                (str(document_id), str(chunk_id), float(score))
                for document_id, chunk_id, score in zip(
                    selected_documents, selected_chunks, selected_scores, strict=True
                )
            ]
