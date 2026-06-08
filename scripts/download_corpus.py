"""Download do corpus LGPD — documentos oficiais da ANPD e texto da lei.

Também extrai os artigos da LGPD do PDF oficial e salva em data/lgpd_articles.json,
usado pela tool cite_article para evitar alucinação de números de artigos.
"""

import hashlib
import json
import logging
import re
import ssl
from pathlib import Path
from urllib.request import Request, urlopen

# Certificados raiz do sistema podem estar desatualizados em algumas distros Linux;
# os PDFs são de domínios governamentais públicos (.gov.br / .df.gov.br).
_SSL_CTX = ssl.create_default_context()
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode = ssl.CERT_NONE

from pypdf import PdfReader

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
_log = logging.getLogger("corpus-download")

CORPUS_DIR = Path("data/corpus")

_CORPUS = [
    {
        "filename": "lgpd-lei-13709-2018.pdf",
        "url": "https://egov.df.gov.br/wp-content/uploads/2023/07/Lei-n.-13.709-2018-%E2%80%93-Lei-Geral-de-Protecao-de-Dados-Pessoais-LGPD.pdf",
        "title": "Lei 13.709/2018 — Lei Geral de Proteção de Dados Pessoais (LGPD)",
    },
    {
        "filename": "anpd-guia-agentes-de-tratamento.pdf",
        "url" : "https://www.gov.br/anpd/pt-br/centrais-de-conteudo/materiais-educativos-e-publicacoes/2021.05.27GuiaAgentesdeTratamento_Final.pdf/@@display-file/file",
        "title": "ANPD — Guia Orientativo: Agentes de Tratamento e Encarregado (2021)",
    },
    {
        "filename": "anpd-guia-seguranca-da-informacao.pdf",
        "url": "https://www.gov.br/anpd/pt-br/centrais-de-conteudo/materiais-educativos-e-publicacoes/guia_seguranca_da_informacao_para_atpps___defeso_eleitoral.pdf/@@display-file/file",
        "title": "ANPD — Guia Orientativo: Segurança da Informação para Agentes de Tratamento (2021)",
    },
]

_USER_AGENT = "lgpd-compliance-rag/1.0"
_MAX_SIZE_MB = 25


def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_one(entry: dict, out_dir: Path) -> Path:
    dest = out_dir / entry["filename"]
    if dest.exists():
        _log.info("ja existe: %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)
        return dest
    _log.info("baixando: %s", entry["title"])
    req = Request(entry["url"], headers={"User-Agent": _USER_AGENT})
    with urlopen(req, timeout=60, context=_SSL_CTX) as resp:
        data = resp.read()
    # Valida pelos bytes mágicos do PDF (mais confiável que Content-Type em sites gov.br)
    if not data.startswith(b"%PDF"):
        _log.warning(
            "AVISO: %s retornou HTML em vez de PDF (URL desatualizada?). "
            "Pulando — adicione o arquivo manualmente se necessário.",
            entry["filename"],
        )
        return dest
    size_mb = len(data) / 1e6
    if size_mb > _MAX_SIZE_MB:
        raise RuntimeError(f"{entry['filename']}: {size_mb:.1f} MB > limite {_MAX_SIZE_MB} MB")
    dest.write_bytes(data)
    _log.info("  ok: %.1f MB — %s", size_mb, dest.name)
    return dest


def _load_sums(sums_path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in sums_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(maxsplit=1)
        if len(parts) == 2:
            out[parts[1].strip()] = parts[0].strip()
    return out


_LGPD_PDF_FILENAME = "lgpd-lei-13709-2018.pdf"
_ARTICLES_OUTPUT = Path("data/lgpd_articles.json")

# Padrão que identifica início de artigo: "Art. 1º", "Art. 55-A", "Art. 2o", etc.
_ART_PATTERN = re.compile(r"Art\.\s*(\d+(?:-[A-Z])?)[ºo°]?[\s\.]")


def _extract_lgpd_articles(pdf_path: Path, output_path: Path) -> int:
    """Extrai artigos do PDF da LGPD e salva em output_path como JSON.

    Retorna o número de artigos extraídos.
    """
    _log.info("extraindo artigos de %s ...", pdf_path.name)

    # Concatena texto de todas as páginas com normalização mínima
    full_text = ""
    for page in PdfReader(str(pdf_path)).pages:
        raw = page.extract_text() or ""
        raw = re.sub(r"-\n(\w)", r"\1", raw)       # rehifenização
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        raw = re.sub(r"[ \t]{2,}", " ", raw)
        full_text += raw + "\n"

    # Encontra todas as posições de início de artigo
    matches = list(_ART_PATTERN.finditer(full_text))
    if not matches:
        _log.warning("nenhum artigo encontrado em %s", pdf_path.name)
        return 0

    articles: dict[str, str] = {}
    for i, match in enumerate(matches):
        art_num = match.group(1)          # ex: "1", "55", "55-A"
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        text = full_text[start:end].strip()
        # Guarda apenas a primeira ocorrência (evita capturar referências cruzadas)
        if art_num not in articles:
            articles[art_num] = text

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(articles, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _log.info("  %d artigos extraídos → %s", len(articles), output_path)
    return len(articles)


if __name__ == "__main__":
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)

    for _entry in _CORPUS:
        _download_one(_entry, CORPUS_DIR)

    # Extrai artigos do PDF da LGPD logo após o download
    _lgpd_pdf = CORPUS_DIR / _LGPD_PDF_FILENAME
    if _lgpd_pdf.exists():
        _extract_lgpd_articles(_lgpd_pdf, _ARTICLES_OUTPUT)

    _sums_path = next(
        (p for p in (Path("datasets/SHA256SUMS"), Path("../datasets/SHA256SUMS")) if p.exists()),
        None,
    )
    if _sums_path:
        _existing = _load_sums(_sums_path)
        for _entry in _CORPUS:
            _dest = CORPUS_DIR / _entry["filename"]
            if _dest.exists() and _entry["filename"] in _existing:
                if _sha256_of(_dest) != _existing[_entry["filename"]]:
                    raise RuntimeError(f"SHA256 divergente: {_entry['filename']}")
        _log.info("SHA256SUMS verificado.")
    else:
        _log.info("SHA256SUMS nao encontrado — verificacao ignorada (normal no Colab).")

    _count = len(list(CORPUS_DIR.glob("*.pdf")))
    print(f"\ncorpus: {_count} PDFs em {CORPUS_DIR}/")
    for p in sorted(CORPUS_DIR.glob("*.pdf")):
        print(f"  {p.name}  ({p.stat().st_size / 1e6:.1f} MB)")
