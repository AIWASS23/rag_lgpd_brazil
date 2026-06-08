# Corpus — LGPD Compliance Assistant

Esta pasta recebe os PDFs do projeto. Execute o script abaixo para baixá-los automaticamente.

## Download

```bash
python scripts/download_corpus.py
```

Após o download você terá 3 arquivos (~5 MB total):

| Arquivo | Fonte | Descrição |
|---|---|---|
| `lgpd-lei-13709-2018.pdf` | egov.df.gov.br | Texto oficial da Lei 13.709/2018 (65 artigos) |
| `anpd-guia-agentes-de-tratamento.pdf` | gov.br/anpd | Guia Orientativo: Agentes de Tratamento e Encarregado (ANPD, 2021) |
| `anpd-guia-seguranca-da-informacao.pdf` | gov.br/anpd | Guia Orientativo: Segurança da Informação para Agentes de Tratamento (ANPD, 2021) |

## Restrições

- Apenas texto extraível (PDFs escaneados sem OCR não funcionam — use `ocrmypdf` antes)
- Todos os documentos são públicos e de domínio governamental
