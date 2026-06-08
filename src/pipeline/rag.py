"""RAG pipeline — chunk, embed, index, retrieve, generate.
"""

import os
import re
from pathlib import Path

import json

import chromadb
import numpy as np
from chromadb.utils.embedding_functions import (
    OpenAIEmbeddingFunction,
    SentenceTransformerEmbeddingFunction,
)
from langchain_text_splitters import TokenTextSplitter
from openai import OpenAI
from pypdf import PdfReader
from rank_bm25 import BM25Okapi

from src.pipeline.config_loader import load_config
from src.pipeline.memory import BaseMemory
from src.pipeline.tools import TOOLS, run_tool_call


def _normalize(text: str) -> str:
    """Limpeza mínima do texto extraído do PDF antes do chunking."""
    text = re.sub(r"-\n(\w)", r"\1", text)   # rehifenização de fim de linha
    text = re.sub(r"\n{3,}", "\n\n", text)   # colapsa quebras excessivas
    text = re.sub(r"[ \t]{2,}", " ", text)   # colapsa espaços/tabs extras
    return text.strip()


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def _apply_threshold(hits: list[dict], threshold: float) -> list[dict]:
    """Remove hits com similaridade abaixo do threshold (0.0 = sem filtro)."""
    if threshold <= 0.0:
        return hits
    # distance Chroma (L2): similaridade ≈ 1 - distance/2 para vetores normalizados
    return [h for h in hits if (1.0 - h["distance"] / 2.0) >= threshold]


def _make_llm_client(provider: str) -> OpenAI:
    """Retorna cliente OpenAI-compatible para o provider configurado."""
    if provider == "gemini":
        return OpenAI(
            api_key=os.environ["GEMINI_API_KEY"],
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
    if provider == "openai":
        return OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    if provider == "huggingface":
        return OpenAI(
            api_key=os.environ["HF_TOKEN"],
            base_url="https://router.huggingface.co/v1/",
        )
    raise ValueError(f"Provider LLM desconhecido: {provider!r}. Use gemini | openai | huggingface")


def _make_embed_fn(provider: str, model: str):
    """Retorna embedding function compatível com Chroma para o provider configurado."""
    if provider == "gemini":
        return OpenAIEmbeddingFunction(
            api_key=os.environ["GEMINI_API_KEY"],
            model_name=model,
            api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
    if provider == "openai":
        return OpenAIEmbeddingFunction(
            api_key=os.environ["OPENAI_API_KEY"],
            model_name=model,
        )
    if provider == "huggingface":
        # Roda localmente via sentence-transformers — sem chamada de API
        return SentenceTransformerEmbeddingFunction(model_name=model)
    raise ValueError(f"Provider embedding desconhecido: {provider!r}. Use gemini | openai | huggingface")


def _make_splitter(cfg):
    """Retorna TokenTextSplitter usando tiktoken ou tokenizador HuggingFace."""
    if cfg.tokenizer_source == "huggingface":
        from transformers import AutoTokenizer
        hf_tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_model)
        return TokenTextSplitter.from_huggingface_tokenizer(
            hf_tokenizer,
            chunk_size=cfg.chunk_size,
            chunk_overlap=cfg.chunk_overlap,
        )
    # tiktoken (padrão)
    return TokenTextSplitter(
        encoding_name=cfg.tokenizer_model,
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
    )


class RAGPipeline:
    """Pipeline RAG end-to-end com Chroma local."""

    def __init__(
        self,
        corpus_dir: str = "data/corpus",
        persist_dir: str = "data/chroma",
        collection_name: str = "docs",
        llm_model: str | None = None,
        embed_model: str | None = None,
    ) -> None:
        cfg = load_config()

        self.client = _make_llm_client(cfg.llm.provider)
        self.llm_model = llm_model or cfg.llm.cheap_model
        self.embed_fn = _make_embed_fn(
            cfg.embedding.provider,
            embed_model or cfg.embedding.model,
        )

        self.corpus_dir = Path(corpus_dir)
        self.persist_dir = persist_dir
        self.collection_name = collection_name

        chroma = chromadb.PersistentClient(path=persist_dir)
        self.collection = chroma.get_or_create_collection(
            name=collection_name, embedding_function=self.embed_fn
        )

    def ingest_and_index(self) -> int:
        """Le PDFs de `corpus_dir`, faz chunking por token e indexa em Chroma.

        Retorna numero de chunks indexados.
        """
        # 1.A — extração e normalização
        docs: list[dict] = []
        for pdf_path in sorted(self.corpus_dir.glob("*.pdf")):
            reader = PdfReader(str(pdf_path))
            for page_num, page in enumerate(reader.pages, start=1):
                raw = page.extract_text() or ""
                text = _normalize(raw)
                if text:
                    docs.append({"text": text, "source": pdf_path.name, "page": page_num})

        if not docs:
            return 0

        # 1.B — chunking por token (tiktoken ou HuggingFace conforme config)
        cfg = load_config().chunking
        splitter = _make_splitter(cfg)
        chunks: list[dict] = []
        for doc in docs:
            for i, chunk_text in enumerate(splitter.split_text(doc["text"])):
                chunks.append({
                    "id": f"{doc['source']}__p{doc['page']}__c{i}",
                    "text": chunk_text,
                    "source": doc["source"],
                    "page": doc["page"],
                })

        # 1.C — indexação no Chroma (em lotes para evitar timeout de embedding)
        batch_size = 100
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            self.collection.add(
                ids=[c["id"] for c in batch],
                documents=[c["text"] for c in batch],
                metadatas=[{"source": c["source"], "page": c["page"]} for c in batch],
            )

        return self.collection.count()

    def retrieve(self, query: str, k: int | None = None) -> list[dict]:
        """Busca top-k chunks via similarity, MMR ou hybrid (BM25 + similarity)."""
        cfg = load_config().retriever
        k = k or cfg.top_k

        if cfg.technique == "mmr":
            return self._retrieve_mmr(query, k, cfg.mmr_lambda, cfg.score_threshold)
        if cfg.technique == "hybrid":
            return self._retrieve_hybrid(query, k, cfg.score_threshold)
        return self._retrieve_similarity(query, k, cfg.score_threshold)

    def _retrieve_similarity(self, query: str, k: int, threshold: float) -> list[dict]:
        results = self.collection.query(query_texts=[query], n_results=k)
        return self._pack_results(results, threshold)

    def _retrieve_mmr(self, query: str, k: int, lambda_mult: float, threshold: float) -> list[dict]:
        # Busca candidatos extras com embeddings para calcular diversidade
        n_candidates = min(k * 3, self.collection.count() or 1)
        results = self.collection.query(
            query_texts=[query],
            n_results=n_candidates,
            include=["documents", "metadatas", "distances", "embeddings"],
        )

        docs = results["documents"][0]
        metas = results["metadatas"][0]
        dists = results["distances"][0]
        embeddings = np.array(results["embeddings"][0])  # shape (n_candidates, dim)

        # Embedding da query via embed_fn para calcular similaridade
        query_emb = np.array(self.embed_fn([query])[0])

        selected_idx: list[int] = []
        remaining = list(range(len(docs)))

        while len(selected_idx) < k and remaining:
            query_sims = np.array([_cosine(query_emb, embeddings[i]) for i in remaining])
            if not selected_idx:
                best_pos = int(np.argmax(query_sims))
            else:
                sel_embs = embeddings[selected_idx]
                redundancy = np.array([
                    float(np.max([_cosine(embeddings[i], s) for s in sel_embs]))
                    for i in remaining
                ])
                mmr_scores = lambda_mult * query_sims - (1 - lambda_mult) * redundancy
                best_pos = int(np.argmax(mmr_scores))

            chosen = remaining.pop(best_pos)
            selected_idx.append(chosen)

        hits = [
            {"text": docs[i], "source": metas[i]["source"], "page": metas[i]["page"], "distance": dists[i]}
            for i in selected_idx
        ]
        return _apply_threshold(hits, threshold)

    def _retrieve_hybrid(self, query: str, k: int, threshold: float) -> list[dict]:
        # Carrega todos os chunks para montar índice BM25
        all_data = self.collection.get(include=["documents", "metadatas"])
        all_docs = all_data["documents"]
        all_metas = all_data["metadatas"]

        # BM25
        tokenized = [d.lower().split() for d in all_docs]
        bm25 = BM25Okapi(tokenized)
        bm25_scores = bm25.get_scores(query.lower().split())
        bm25_ranks = {i: r for r, i in enumerate(np.argsort(bm25_scores)[::-1])}

        # Similarity (top candidates)
        n_candidates = min(k * 3, len(all_docs))
        sim_results = self.collection.query(query_texts=[query], n_results=n_candidates)
        sim_ids = sim_results["ids"][0]
        all_ids = all_data["ids"]
        sim_ranks = {all_ids[j]: r for r, j_id in enumerate(sim_ids)
                     for j in range(len(all_ids)) if all_ids[j] == j_id}

        # Reciprocal Rank Fusion
        def rrf(rank: int, c: int = 60) -> float:
            return 1.0 / (c + rank)

        scored: dict[int, float] = {}
        for i in range(len(all_docs)):
            doc_id = all_ids[i]
            score = rrf(bm25_ranks.get(i, len(all_docs)))
            score += rrf(sim_ranks.get(doc_id, len(all_docs)))
            scored[i] = score

        top_indices = sorted(scored, key=scored.__getitem__, reverse=True)[:k]
        hits = [
            {"text": all_docs[i], "source": all_metas[i]["source"], "page": all_metas[i]["page"], "distance": 0.0}
            for i in top_indices
        ]
        return _apply_threshold(hits, threshold)

    @staticmethod
    def _pack_results(results: dict, threshold: float) -> list[dict]:
        hits = [
            {"text": t, "source": m["source"], "page": m["page"], "distance": d}
            for t, m, d in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]
        return _apply_threshold(hits, threshold)

    def answer(
        self,
        question: str,
        k: int | None = None,
        memory: BaseMemory | None = None,
        session_id: str = "default",
        model: str | None = None,
    ) -> dict:
        """Pipeline completo: retrieve → augment → generate com tool calling e memória.

        Retorna {"answer": str, "sources": list[dict], "articles_cited": list[int]}.
        `model` sobrescreve o modelo padrão do pipeline (usado pelo routing cheap-first).
        """
        hits = self.retrieve(question, k=k)
        cfg = load_config()

        # Contexto RAG com cabeçalho de fonte
        context = "\n\n---\n\n".join(
            f"[{h['source']}:p{h['page']}]\n{h['text']}" for h in hits
        ) or "Nenhum trecho relevante encontrado no corpus."

        # Histórico conversacional (vazio se sem memória)
        history = memory.to_openai_messages(session_id) if memory else []

        messages = (
            [{"role": "system", "content": SYSTEM_PROMPT}]
            + history
            + [{"role": "user", "content": PROMPT_TEMPLATE.format(context=context, question=question)}]
        )

        call_kwargs: dict = {
            "model": model or self.llm_model,
            "messages": messages,
            "temperature": cfg.llm.temperature,
            "max_tokens": cfg.llm.max_tokens,
        }
        if TOOLS:
            call_kwargs["tools"] = TOOLS

        # Agentic loop — LLM pode chamar cite_article antes de responder
        articles_cited: list[int] = []
        try:
            response = self.client.chat.completions.create(**call_kwargs)
        except Exception as exc:
            # Alguns providers (ex: HF router) não suportam tools — retry sem elas
            if "tools" in str(exc).lower() or "UNSUPPORTED_OPENAI_PARAMS" in str(exc):
                call_kwargs.pop("tools", None)
                response = self.client.chat.completions.create(**call_kwargs)
            else:
                raise

        while response.choices[0].finish_reason == "tool_calls":
            tool_calls = response.choices[0].message.tool_calls
            messages.append(response.choices[0].message)

            for tc in tool_calls:
                if tc.function.name == "cite_article":
                    args = json.loads(tc.function.arguments)
                    art_num = args.get("article_number")
                    if art_num is not None:
                        articles_cited.append(int(art_num))

                result = run_tool_call(tc.function.name, tc.function.arguments)
                messages.append({
                    "role": "tool",
                    "content": result,
                    "tool_call_id": tc.id,
                })

            call_kwargs["messages"] = messages
            response = self.client.chat.completions.create(**call_kwargs)

        answer_text = response.choices[0].message.content or ""

        # Persiste turno na memória
        if memory:
            memory.add(session_id, "user", question)
            memory.add(session_id, "assistant", answer_text)

        return {
            "answer": answer_text,
            "sources": [{"source": h["source"], "page": h["page"]} for h in hits],
            "articles_cited": sorted(set(articles_cited)),
        }


SYSTEM_PROMPT = """Voce e um assistente especializado em compliance LGPD (Lei 13.709/2018).
Regras obrigatorias:
- Responda APENAS com base no contexto RAG fornecido pelo usuario.
- Se a informacao nao estiver no contexto, responda: "Nao encontrado no corpus."
- Ao mencionar um artigo da LGPD, use a tool cite_article para buscar o texto real — NUNCA invente ou parafraseie o numero do artigo sem verificar.
- Cite a fonte de cada afirmacao usando o formato [arquivo:pagina].
- Seja objetivo e direto; evite linguagem juridica desnecessaria."""

PROMPT_TEMPLATE = """CONTEXTO DO CORPUS:
{context}

PERGUNTA: {question}

Responda com base estrita no contexto acima. Se precisar citar um artigo da LGPD, chame cite_article antes de responder."""


def build_rag_pipeline(corpus_dir: str = "data/corpus") -> RAGPipeline:
    """Factory: cria pipeline e indexa corpus se ainda nao indexado."""
    pipeline = RAGPipeline(corpus_dir=corpus_dir)
    if pipeline.collection.count() == 0:
        pipeline.ingest_and_index()
    return pipeline
