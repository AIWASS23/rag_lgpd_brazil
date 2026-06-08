# RAG LGPD Brazil

Assistente de compliance LGPD baseado em RAG — responde perguntas sobre a Lei 13.709/2018 e guias da ANPD com base em documentos oficiais, sem alucinação de artigos.

## Problema

Profissionais de compliance precisam consultar a LGPD com frequência, mas a lei tem 65 artigos com linguagem jurídica densa. Uma busca simples retorna trechos sem contexto; um LLM genérico alucina números de artigos. A combinação RAG + tool calling resolve os dois problemas: recupera os trechos relevantes do corpus oficial e verifica o texto real do artigo antes de citar.

## Arquitetura

```
User query
  → ExactCache          (SHA256 — replica exata)
  → SemanticCache       (cosine ≥ 0.93 — paráfrases)
  → classify_complexity (heurística léxica → cheap | premium model)
  → RAGPipeline.answer
      → retrieve        (similarity | MMR | BM25+vector hybrid)
      → LLM agentic loop → cite_article tool → resposta final
  → put em ambos os caches
```

```mermaid
flowchart LR
    USER([User]) --> UI[Streamlit UI]
    UI --> EC{Exact cache?}
    EC -->|hit| RESP[Response]
    EC -->|miss| SC{Semantic cache?}
    SC -->|hit| RESP
    SC -->|miss| CLS[classify_complexity]
    CLS -->|simple| CHEAP[cheap model]
    CLS -->|complex| PREM[premium model]
    CHEAP & PREM --> RAG[(Chroma RAG)]
    RAG --> TOOL[cite_article tool]
    TOOL --> RESP
```

## Corpus

3 documentos públicos governamentais (~5 MB total):

| Arquivo | Fonte | Descrição |
|---|---|---|
| `lgpd-lei-13709-2018.pdf` | egov.df.gov.br | Texto oficial da Lei 13.709/2018 (65 artigos) |
| `anpd-guia-agentes-de-tratamento.pdf` | gov.br/anpd | Guia: Agentes de Tratamento e Encarregado (ANPD, 2021) |
| `anpd-guia-seguranca-da-informacao.pdf` | gov.br/anpd | Guia: Segurança da Informação para Agentes de Tratamento (ANPD, 2021) |

## Setup

```bash
# 1. Dependências
uv venv && source .venv/bin/activate
uv sync

# Para embeddings locais (HuggingFace — sem chamada de API)
uv sync --extra huggingface

# 2. API key — copie e edite com sua chave
cp .env.example .env

# 3. Corpus — baixa PDFs e extrai artigos para data/lgpd_articles.json
python scripts/download_corpus.py

# 4. Rodar
streamlit run src/ui/streamlit_app.py
```

### Providers suportados

Configure em `config.yaml` + `.env`:

| Provider | Variável de ambiente | Embedding | LLM |
|---|---|---|---|
| Gemini (padrão) | `GEMINI_API_KEY` | `gemini-embedding-001` | `gemini-2.5-flash-lite` / `gemini-2.5-pro` |
| OpenAI | `OPENAI_API_KEY` | `text-embedding-3-small` | `gpt-4o-mini` |
| HuggingFace | `HF_TOKEN` | `all-MiniLM-L6-v2` (local) | `Llama-3.1-8B` via router |

## Configuração

Todos os parâmetros ficam em `config.yaml`:

```yaml
retriever:
  technique: similarity   # similarity | mmr | hybrid
  top_k: 10
  score_threshold: 0.3

memory:
  technique: in_memory    # in_memory | windowed | redis

cache:
  semantic:
    threshold: 0.93       # cosine similarity mínima para cache hit
```

`CHEAP_MODEL` e `PREMIUM_MODEL` como variáveis de ambiente sobrescrevem os valores do config em runtime.

## Tech stack

- **LLM / Embedding:** Gemini 2.5 Flash-Lite (cheap) · Gemini 2.5 Pro (premium) · ou OpenAI · ou HuggingFace
- **Vector store:** Chroma local (persistido em `data/chroma/`)
- **Chunking:** `langchain-text-splitters` TokenTextSplitter (tiktoken ou HuggingFace tokenizer)
- **Retrieval:** similarity · MMR (diversidade via cosine) · hybrid BM25+vector (Reciprocal Rank Fusion)
- **Tool calling:** `cite_article` — busca o texto oficial do artigo em `data/lgpd_articles.json`
- **Cache:** ExactCache (SHA256 in-memory) + SemanticCache (cosine sobre embeddings armazenados)
- **Memória conversacional:** InMemoryMemory · WindowedMemory · RedisMemory (com fallback automático)
- **Observability:** structured JSON logs com `trace_id` por requisição (`src/observability/trace.py`)
- **UI:** Streamlit

## Decisões de design

- **`cite_article` tool:** o LLM é instruído no system prompt a chamar a tool antes de citar qualquer artigo. Isso elimina alucinação de números de artigos sem depender de pós-processamento.
- **Chunk size 512 tokens:** testado com 256 e 1024; 512 equilibrou precisão de retrieval e custo de embedding para o tamanho deste corpus (~3 PDFs).
- **Routing por heurística léxica:** keywords como "compare", "analise", "explique" disparam o modelo premium. Para o volume de queries esperado, um classifier treinado seria overfitting.
- **Sem re-ranking:** corpus pequeno (~3 PDFs) — a latência adicional de um cross-encoder não compensa.
- **Cache semântico in-memory:** sem Redis por padrão para simplificar o deploy no Streamlit Cloud. `memory.technique: redis` ativa persistência entre sessões quando Redis está disponível.

## Limitações

- O corpus é fixo; o app não suporta upload de PDF pelo usuário.
- Embeddings HuggingFace rodam localmente — o primeiro uso baixa ~100 MB do modelo.
- O cache semântico não persiste entre reinícios do app (backend in-memory padrão).

## Observabilidade

O módulo `src/observability/trace.py` emite JSON estruturado para stdout com `trace_id`, `event`, `latency_ms` e campos adicionais por requisição. Para integração com Langfuse (dashboard de traces, custo e latência em produção), veja `docs/observability.md`.

## Estrutura

```
RAG_LGPD_BRAZIL/
├── data/
│   ├── corpus/           # PDFs oficiais (download_corpus.py)
│   ├── chroma/           # vector store persistido (gitignored)
│   └── lgpd_articles.json
├── src/
│   ├── pipeline/
│   │   ├── rag.py        # RAGPipeline — ingest, retrieve, answer
│   │   ├── tools.py      # cite_article tool
│   │   ├── cache.py      # ExactCache + SemanticCache
│   │   ├── routing.py    # classify_complexity
│   │   ├── memory.py     # InMemory / Windowed / Redis
│   │   └── config_loader.py
│   ├── observability/trace.py
│   └── ui/streamlit_app.py
├── scripts/download_corpus.py
├── tests/test_smoke.py
├── config.yaml
└── .env.example
```
