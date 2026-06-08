"""Carrega config.yaml e expõe como dataclasses tipadas."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ChunkingConfig:
    tokenizer_source: str = "tiktoken"   # tiktoken | huggingface
    tokenizer_model: str = "cl100k_base"
    chunk_size: int = 512
    chunk_overlap: int = 50


@dataclass
class EmbeddingConfig:
    provider: str = "gemini"
    model: str = "gemini-embedding-001"


@dataclass
class LLMConfig:
    provider: str = "gemini"
    cheap_model: str = "gemini-2.5-flash-lite"
    premium_model: str = "gemini-2.5-pro"
    temperature: float = 0.1
    max_tokens: int = 1024


@dataclass
class RetrieverConfig:
    technique: str = "similarity"
    top_k: int = 5
    mmr_lambda: float = 0.5
    score_threshold: float = 0.0


@dataclass
class MemoryConfig:
    technique: str = "redis"
    redis_url: str = "redis://localhost:6379"
    redis_ttl_seconds: int = 3600
    window_size: int = 10
    session_prefix: str = "lgpd_chat"


@dataclass
class CacheConfig:
    exact_enabled: bool = True
    semantic_enabled: bool = True
    semantic_threshold: float = 0.93


@dataclass
class AppConfig:
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)


def _from_dict(cls, data: dict[str, Any]):
    fields = {f.name for f in cls.__dataclass_fields__.values()}
    return cls(**{k: v for k, v in data.items() if k in fields})


def load_config(path: str | Path | None = None) -> AppConfig:
    """Carrega config.yaml do projeto. Usa defaults se arquivo não encontrado."""
    if path is None:
        path = Path(__file__).resolve().parents[2] / "config.yaml"

    if not Path(path).exists():
        return AppConfig()

    with open(path) as f:
        raw: dict = yaml.safe_load(f) or {}

    chunking = _from_dict(ChunkingConfig, raw.get("chunking", {}))
    embedding = _from_dict(EmbeddingConfig, raw.get("embedding", {}))
    llm = _from_dict(LLMConfig, raw.get("llm", {}))
    retriever = _from_dict(RetrieverConfig, raw.get("retriever", {}))
    memory = _from_dict(MemoryConfig, raw.get("memory", {}))

    cache_raw = raw.get("cache", {})
    cache = CacheConfig(
        exact_enabled=cache_raw.get("exact", {}).get("enabled", True),
        semantic_enabled=cache_raw.get("semantic", {}).get("enabled", True),
        semantic_threshold=cache_raw.get("semantic", {}).get("threshold", 0.93),
    )

    return AppConfig(
        chunking=chunking,
        embedding=embedding,
        llm=llm,
        retriever=retriever,
        memory=memory,
        cache=cache,
    )
