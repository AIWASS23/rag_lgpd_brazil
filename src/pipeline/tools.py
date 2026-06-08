"""Function-calling / tool-use — registro de tools usadas pelo agente."""

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

_ARTICLES_PATH = Path(__file__).resolve().parents[3] / "data" / "lgpd_articles.json"


@lru_cache(maxsize=1)
def _load_articles() -> dict[str, str]:
    """Carrega lgpd_articles.json uma única vez por processo."""
    if not _ARTICLES_PATH.exists():
        return {}
    return json.loads(_ARTICLES_PATH.read_text(encoding="utf-8"))


def cite_article(article_number: str) -> str:
    """Retorna o texto integral do artigo N da LGPD (Lei 13.709/2018).

    Aceita número simples ("7") ou composto ("55-A").
    """
    articles = _load_articles()
    if not articles:
        return (
            "Base de artigos não encontrada. "
            "Execute scripts/download_corpus.py para gerar data/lgpd_articles.json."
        )
    text = articles.get(str(article_number))
    if text is None:
        available = ", ".join(sorted(articles, key=lambda x: (len(x), x)))
        return (
            f"Artigo {article_number} não encontrado na LGPD. "
            f"Artigos disponíveis: {available}"
        )
    return text


TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "cite_article",
            "description": (
                "Retorna o texto oficial e integral do artigo N da LGPD (Lei 13.709/2018). "
                "Use sempre que precisar citar ou verificar o conteúdo de um artigo específico — "
                "nunca invente ou parafraseie o texto sem chamar esta tool primeiro."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "article_number": {
                        "type": "string",
                        "description": (
                            "Número do artigo da LGPD. "
                            "Use string para suportar artigos compostos: '7', '18', '55-A'."
                        ),
                    },
                },
                "required": ["article_number"],
            },
        },
    },
]

TOOL_REGISTRY: dict[str, Callable[..., str]] = {
    "cite_article": cite_article,
}


def run_tool_call(name: str, arguments_json: str) -> str:
    """Executa uma tool call e retorna o resultado como string."""
    if name not in TOOL_REGISTRY:
        return f"ERROR: tool '{name}' não registrada"
    try:
        kwargs = json.loads(arguments_json)
        return TOOL_REGISTRY[name](**kwargs)
    except Exception as e:
        return f"ERROR ao executar {name}: {e}"
