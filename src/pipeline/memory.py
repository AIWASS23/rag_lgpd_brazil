"""Memória conversacional com backends: Redis, in-memory e windowed.
"""

import json
import time
from abc import ABC, abstractmethod
from typing import Any

from .config_loader import MemoryConfig


class Message:
    __slots__ = ("role", "content", "timestamp")

    def __init__(self, role: str, content: str) -> None:
        self.role = role
        self.content = content
        self.timestamp = time.time()

    def to_dict(self) -> dict[str, Any]:
        return {"role": self.role, "content": self.content, "timestamp": self.timestamp}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Message":
        m = cls(d["role"], d["content"])
        m.timestamp = d.get("timestamp", time.time())
        return m


class BaseMemory(ABC):
    """Interface comum para todos os backends de memória."""

    @abstractmethod
    def add(self, session_id: str, role: str, content: str) -> None: ...

    @abstractmethod
    def get_history(self, session_id: str) -> list[Message]: ...

    @abstractmethod
    def clear(self, session_id: str) -> None: ...

    def to_openai_messages(self, session_id: str) -> list[dict[str, str]]:
        """Converte histórico para formato de messages da API OpenAI/Gemini."""
        return [{"role": m.role, "content": m.content} for m in self.get_history(session_id)]


class InMemoryMemory(BaseMemory):
    """Backend simples em dicionário Python. Não persiste entre reinícios."""

    def __init__(self, window_size: int = 10) -> None:
        self._store: dict[str, list[Message]] = {}
        self.window_size = window_size

    def add(self, session_id: str, role: str, content: str) -> None:
        if session_id not in self._store:
            self._store[session_id] = []
        self._store[session_id].append(Message(role, content))

    def get_history(self, session_id: str) -> list[Message]:
        msgs = self._store.get(session_id, [])
        return msgs[-self.window_size :] if self.window_size > 0 else msgs

    def clear(self, session_id: str) -> None:
        self._store.pop(session_id, None)


class WindowedMemory(InMemoryMemory):
    """Igual ao InMemoryMemory, mas com janela deslizante explícita.

    Mantém apenas as últimas `window_size` mensagens (par user+assistant = 2 msgs).
    """


class RedisMemory(BaseMemory):
    """Backend Redis — persiste histórico entre sessões e reinícios do app.

    Cada sessão é uma Redis List com chave `{prefix}:{session_id}`.
    Mensagens são serializadas como JSON. TTL configurável via config.yaml.
    """

    def __init__(self, config: MemoryConfig) -> None:
        self.config = config
        self._client = self._connect()

    def _connect(self):
        try:
            import redis  # type: ignore

            client = redis.from_url(self.config.redis_url, decode_responses=True)
            client.ping()
            return client
        except Exception as exc:
            raise RuntimeError(
                f"Não foi possível conectar ao Redis em {self.config.redis_url}. "
                "Verifique se o Redis está rodando ou mude memory.technique para 'in_memory' no config.yaml.\n"
                f"Erro original: {exc}"
            ) from exc

    def _key(self, session_id: str) -> str:
        return f"{self.config.session_prefix}:{session_id}"

    def add(self, session_id: str, role: str, content: str) -> None:
        key = self._key(session_id)
        msg = Message(role, content)
        self._client.rpush(key, json.dumps(msg.to_dict()))
        self._client.expire(key, self.config.redis_ttl_seconds)
        # Mantém janela deslizante
        total = self._client.llen(key)
        if total > self.config.window_size * 2:
            self._client.ltrim(key, total - self.config.window_size * 2, -1)

    def get_history(self, session_id: str) -> list[Message]:
        key = self._key(session_id)
        raw = self._client.lrange(key, 0, -1)
        return [Message.from_dict(json.loads(r)) for r in raw]

    def clear(self, session_id: str) -> None:
        self._client.delete(self._key(session_id))

    def list_sessions(self) -> list[str]:
        prefix = f"{self.config.session_prefix}:*"
        keys = self._client.keys(prefix)
        strip = len(self.config.session_prefix) + 1
        return [k[strip:] for k in keys]


def build_memory(config: MemoryConfig) -> BaseMemory:
    """Factory: instancia o backend correto conforme config.yaml → memory.technique."""
    technique = config.technique.lower()

    if technique == "redis":
        try:
            return RedisMemory(config)
        except RuntimeError:
            # Fallback gracioso para in_memory se Redis não disponível
            import warnings
            warnings.warn(
                "Redis indisponível — usando InMemoryMemory como fallback. "
                "Histórico não será persistido.",
                stacklevel=2,
            )
            return InMemoryMemory(window_size=config.window_size)

    if technique == "windowed":
        return WindowedMemory(window_size=config.window_size)

    return InMemoryMemory(window_size=config.window_size)
