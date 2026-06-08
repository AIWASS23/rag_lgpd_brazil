"""Avaliação RAGAS do pipeline RAG LGPD.

Métricas avaliadas:
  - faithfulness        : resposta fundamentada no contexto recuperado
  - answer_relevancy    : resposta pertinente à pergunta
  - context_precision   : contexto recuperado relevante para a pergunta

Uso:
    uv sync --extra eval
    python scripts/eval_ragas.py

    # Para salvar resultados detalhados em CSV:
    python scripts/eval_ragas.py --output results/ragas_eval.csv
"""

import argparse
import json
import os
import sys
import types
import warnings
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

# Shim de compatibilidade: RAGAS 0.2 importa ChatVertexAI de langchain_community,
# removido na versão 0.4+. O stub evita ImportError sem afetar o comportamento,
# pois só é usado no llm_factory() que não é chamado quando passamos o LLM explicitamente.
_vertexai_stub = types.ModuleType("langchain_community.chat_models.vertexai")
_vertexai_stub.ChatVertexAI = type("ChatVertexAI", (), {})
sys.modules.setdefault("langchain_community.chat_models.vertexai", _vertexai_stub)

from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# LLM e embeddings para avaliação RAGAS
# ---------------------------------------------------------------------------

def _build_ragas_llm():
    """Cria wrapper RAGAS de LLM usando o provider disponível no .env."""
    from langchain_openai import ChatOpenAI
    from ragas.llms import LangchainLLMWrapper

    if os.environ.get("GEMINI_API_KEY"):
        llm = ChatOpenAI(
            api_key=os.environ["GEMINI_API_KEY"],
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            model="gemini-2.5-flash-lite",
            temperature=0,
        )
    elif os.environ.get("OPENAI_API_KEY"):
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    elif os.environ.get("HF_TOKEN"):
        llm = ChatOpenAI(
            api_key=os.environ["HF_TOKEN"],
            base_url="https://router.huggingface.co/v1/",
            model=os.environ.get("PREMIUM_MODEL", "meta-llama/Llama-3.3-70B-Instruct"),
            temperature=0,
        )
    else:
        raise EnvironmentError(
            "Nenhuma API key encontrada. Configure GEMINI_API_KEY, OPENAI_API_KEY ou HF_TOKEN no .env"
        )

    return LangchainLLMWrapper(llm)


def _build_ragas_embeddings():
    """Cria wrapper RAGAS de embeddings (local via sentence-transformers ou API)."""
    if os.environ.get("GEMINI_API_KEY"):
        from langchain_openai import OpenAIEmbeddings
        from ragas.embeddings import LangchainEmbeddingsWrapper
        embeddings = OpenAIEmbeddings(
            api_key=os.environ["GEMINI_API_KEY"],
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            model="gemini-embedding-001",
        )
        return LangchainEmbeddingsWrapper(embeddings)

    if os.environ.get("OPENAI_API_KEY"):
        from langchain_openai import OpenAIEmbeddings
        from ragas.embeddings import LangchainEmbeddingsWrapper
        return LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))

    # HuggingFace: embedding local via sentence-transformers + wrapper LangChain
    from langchain_huggingface import HuggingFaceEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper
    model_name = os.environ.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    return LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=model_name))


# ---------------------------------------------------------------------------
# Pipeline RAG
# ---------------------------------------------------------------------------

def _build_pipeline():
    from src.pipeline.rag import build_rag_pipeline

    corpus_dir = _ROOT / "data" / "corpus"
    if not list(corpus_dir.glob("*.pdf")):
        sys.exit(
            "ERRO: data/corpus/ vazio. Execute primeiro:\n"
            "  python scripts/download_corpus.py"
        )
    return build_rag_pipeline(corpus_dir=str(corpus_dir))


# ---------------------------------------------------------------------------
# Avaliação
# ---------------------------------------------------------------------------

def run_eval(output_path: Path | None = None) -> dict[str, float]:
    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.metrics import answer_relevancy, context_precision, faithfulness

    golden = json.loads((_ROOT / "data" / "golden_set.json").read_text(encoding="utf-8"))
    print(f"Golden set carregado: {len(golden)} queries")

    print("Inicializando pipeline RAG...")
    pipeline = _build_pipeline()

    print("Configurando LLM e embeddings para RAGAS...")
    ragas_llm = _build_ragas_llm()
    ragas_embeddings = _build_ragas_embeddings()

    print("\nColetando respostas do pipeline RAG...")
    samples: list[SingleTurnSample] = []
    for i, item in enumerate(golden, 1):
        question = item["question"]
        reference = item.get("reference", "")
        print(f"  [{i:2d}/{len(golden)}] {question[:70]}...")

        hits = pipeline.retrieve(question)
        result = pipeline.answer(question)

        samples.append(
            SingleTurnSample(
                user_input=question,
                response=result["answer"],
                retrieved_contexts=[h["text"] for h in hits],
                reference=reference,
            )
        )

    dataset = EvaluationDataset(samples=samples)

    metrics = [faithfulness, answer_relevancy, context_precision]
    print(f"\nRodando avaliação RAGAS ({len(samples)} amostras × {len(metrics)} métricas)...")
    warnings.filterwarnings("ignore")

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        raise_exceptions=False,
    )

    df = result.to_pandas()

    faith_val = float(df["faithfulness"].mean())
    rel_val   = float(df["answer_relevancy"].mean())
    prec_val  = float(df["context_precision"].mean())

    print("\n" + "=" * 60)
    print("RAGAS — Resultados finais")
    print("=" * 60)
    print(f"  faithfulness      = {faith_val:.2f}")
    print(f"  answer_relevancy  = {rel_val:.2f}")
    print(f"  context_precision = {prec_val:.2f}")
    print("=" * 60)
    print(
        f"\nfaithfulness={faith_val:.2f}, "
        f"answer_relevancy={rel_val:.2f}, "
        f"context_precision={prec_val:.2f}"
    )

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"\nResultados detalhados salvos em: {output_path}")

    return {
        "faithfulness": faith_val,
        "answer_relevancy": rel_val,
        "context_precision": prec_val,
    }


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Avaliação RAGAS do pipeline RAG LGPD")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        metavar="FILE",
        help="Caminho para salvar resultados detalhados em CSV (ex: results/ragas_eval.csv)",
    )
    args = parser.parse_args()
    run_eval(output_path=args.output)
