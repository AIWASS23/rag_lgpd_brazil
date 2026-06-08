"""Model routing cheap-first com fallback.
"""

import os
import re
from dataclasses import dataclass

from openai import OpenAI

from src.pipeline.config_loader import load_config


@dataclass(frozen=True)
class RouteDecision:
    model: str
    complexity: str  # "simple" | "complex"
    reason: str


_COMPLEX_KEYWORDS = re.compile(
    r"\b(compare|comparar|analise|analisar|explique|explicar|"
    r"diferen[çc]a|todos os artigos?|liste todos|"
    r"implica[çc][õo]es?|consequências?|contradiz|conflita|"
    r"contextualize|aprofunde|detalhe)\b",
    re.IGNORECASE,
)


def classify_complexity(query: str) -> RouteDecision:
    """Classifica complexidade da query para escolher modelo (cheap vs premium).

    Estratégia heurística baseada em sinais léxicos e comprimento.
    Em produção evoluiria para um classifier treinado.
    """
    cfg = load_config().llm
    cheap_model = os.environ.get("CHEAP_MODEL", cfg.cheap_model)
    premium_model = os.environ.get("PREMIUM_MODEL", cfg.premium_model)

    q = query.strip()

    # Sinal 1 — keywords que indicam análise profunda
    match = _COMPLEX_KEYWORDS.search(q)
    if match:
        return RouteDecision(
            model=premium_model,
            complexity="complex",
            reason=f"keyword '{match.group()}' indica análise jurídica aprofundada",
        )

    # Sinal 2 — cenário descritivo longo (> 200 chars)
    if len(q) > 200:
        return RouteDecision(
            model=premium_model,
            complexity="complex",
            reason=f"query longa ({len(q)} chars) sugere cenário descritivo complexo",
        )

    # Sinal 3 — pergunta curta e direta (< 80 chars terminando em ?)
    if len(q) < 80 and q.endswith("?"):
        return RouteDecision(
            model=cheap_model,
            complexity="simple",
            reason="pergunta curta e direta — cheap model suficiente",
        )

    # Default
    return RouteDecision(
        model=cheap_model,
        complexity="simple",
        reason="nenhum sinal de complexidade detectado — cheap model suficiente",
    )


def make_client() -> OpenAI:
    """Cliente OpenAI-compatible para o provider configurado."""
    if "GEMINI_API_KEY" in os.environ:
        return OpenAI(
            api_key=os.environ["GEMINI_API_KEY"],
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
    return OpenAI()
