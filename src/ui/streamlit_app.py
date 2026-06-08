"""Streamlit UI — entrada principal do app. Pronta para deploy 1-click no Streamlit Cloud.
"""

import sys
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

load_dotenv()

import streamlit as st 

from src.observability.trace import trace, log_event  
from src.pipeline.cache import ExactCache, SemanticCache  
from src.pipeline.config_loader import load_config  
from src.pipeline.memory import build_memory  
from src.pipeline.rag import build_rag_pipeline  
from src.pipeline.routing import classify_complexity  


st.set_page_config(page_title="Portfolio LLM Demo", page_icon=":robot:", layout="centered")
st.title(":robot: TODO — Substitua pelo titulo do seu projeto")
st.caption("TODO — Substitua: 1-sentence pitch do seu projeto")


# ---------------------------------------------------------------- Recursos cached
@st.cache_resource
def get_pipeline():
    return build_rag_pipeline(corpus_dir=str(_ROOT / "data" / "corpus"))


@st.cache_resource
def get_exact_cache():
    return ExactCache()


@st.cache_resource
def get_semantic_cache():
    return SemanticCache(threshold=0.93)


@st.cache_resource
def get_memory():
    cfg = load_config()
    return build_memory(cfg.memory)


with st.spinner("Inicializando pipeline RAG..."):
    pipeline = get_pipeline()
    exact_cache = get_exact_cache()
    semantic_cache = get_semantic_cache()
    memory = get_memory()


if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "cache_hits" not in st.session_state:
    st.session_state.cache_hits = {"exact": 0, "semantic": 0}
if "last_model" not in st.session_state:
    st.session_state.last_model = "—"
if "last_complexity" not in st.session_state:
    st.session_state.last_complexity = "—"


# ---------------------------------------------------------------- Sidebar
_QUICK_QUERIES = {
    "Art. 1 — Objetivo": "Qual é o objetivo da LGPD?",
    "Art. 5 — Definições": "O que são dados pessoais sensíveis segundo a LGPD?",
    "Art. 7 — Bases legais": "Quais são as bases legais para tratamento de dados pessoais?",
    "Art. 11 — Dados sensíveis": "Quando é permitido tratar dados pessoais sensíveis?",
    "Art. 46 — Segurança": "Quais são as obrigações de segurança no tratamento de dados?",
}

with st.sidebar:
    st.header("Sessão")
    st.code(st.session_state.session_id[:8] + "...", language=None)

    st.header("Métricas")
    st.metric("Chunks indexados", pipeline.collection.count())
    st.metric("Cache hits (exact)", st.session_state.cache_hits["exact"])
    st.metric("Cache hits (semântico)", st.session_state.cache_hits["semantic"])
    st.metric("Último modelo", st.session_state.last_model)
    st.metric("Complexidade", st.session_state.last_complexity)

    st.divider()
    if st.button("Limpar histórico"):
        memory.clear(st.session_state.session_id)
        st.session_state.messages = []
        st.session_state.cache_hits = {"exact": 0, "semantic": 0}
        st.session_state.last_model = "—"
        st.session_state.last_complexity = "—"
        st.success("Histórico limpo.")

    if st.button("Limpar caches"):
        get_exact_cache.clear()
        get_semantic_cache.clear()
        st.success("Caches limpos. Recarregue a pagina.")

    st.divider()
    st.header("Artigos de atalho")
    for label, question in _QUICK_QUERIES.items():
        if st.button(label, key=f"quick_{label}"):
            st.session_state.pending_query = question
            st.rerun()


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("sources"):
            with st.expander("Fontes citadas"):
                for s in msg["sources"]:
                    st.write(f"- `{s['source']}:p{s['page']}`")
        if msg.get("articles_cited"):
            with st.expander("Artigos da LGPD citados"):
                for art in msg["articles_cited"]:
                    st.write(f"- Art. {art}")

chat_input = st.chat_input("Pergunte algo sobre o corpus indexado...")
query: str | None = st.session_state.pop("pending_query", None) or chat_input

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        with trace("query_handle", query=query) as ctx:
            trace_id = ctx["trace_id"]

            # 1. Exact cache
            cached = exact_cache.get(query)
            if cached:
                st.session_state.cache_hits["exact"] += 1
                st.success("Cache hit (exact)")
                st.write(cached)
                log_event("cache_hit", trace_id=trace_id, layer="exact")
                st.session_state.messages.append({"role": "assistant", "content": cached})
                st.stop()

            # 2. Semantic cache
            try:
                cached = semantic_cache.get(query)
            except NotImplementedError:
                cached = None
                st.warning("Semantic cache nao implementado (TODO 5). Caindo no LLM real.")

            if cached:
                st.session_state.cache_hits["semantic"] += 1
                st.success("Cache hit (semântico)")
                st.write(cached)
                log_event("cache_hit", trace_id=trace_id, layer="semantic")
                st.session_state.messages.append({"role": "assistant", "content": cached})
                st.stop()

            # 3. Routing
            routed_model: str | None = None
            try:
                decision = classify_complexity(query)
                routed_model = decision.model
                st.session_state.last_model = decision.model
                st.session_state.last_complexity = decision.complexity
                log_event("route_decision", trace_id=trace_id, **decision.__dict__)
            except NotImplementedError:
                st.warning("Routing nao implementado (TODO 6). Usando modelo default.")

            # 4. Pipeline RAG com memória
            try:
                result = pipeline.answer(
                    query,
                    memory=memory,
                    session_id=st.session_state.session_id,
                    model=routed_model,
                )
            except NotImplementedError as e:
                st.error(f"Pipeline nao implementado: {e}")
                st.info("Implemente TODOs 1-3 em `src/pipeline/rag.py` para destravar.")
                st.stop()

            answer = result["answer"]
            sources = result.get("sources", [])
            articles = result.get("articles_cited", [])

            st.write(answer)

            if sources:
                with st.expander("Fontes citadas"):
                    for s in sources:
                        st.write(f"- `{s['source']}:p{s['page']}`")

            if articles:
                with st.expander("Artigos da LGPD citados"):
                    for art in articles:
                        st.write(f"- Art. {art}")

            exact_cache.put(query, answer)
            semantic_cache.put(query, answer)
            log_event("answer_generated", trace_id=trace_id, sources=len(sources))

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
                "articles_cited": articles,
            })


st.divider()
st.caption(
    "TODO README — substitua por: problem statement, arquitetura, custo/latencia, decisoes de design. "
    "Veja `README.md` do projeto para a estrutura completa."
)
