"""Catálogo canónico de áreas/tratamentos do Instituto Areluna.

Fonte de verdade única. Usado por:
- `graph/nodes/triage.py`     : valida `qualification_state.area_interesse`
- `graph/nodes/router.py`     : pista para encerramento educado fora-do-âmbito
- `graph/nodes/specialist.py` : mapeamento área → namespace RAG (KB)

O LLM pode escrever a área de várias formas ("alinhador invisível" → ortodontia).
`normalize_area` faz best-effort para colapsar para uma chave canónica.
Devolve None quando NADA do catálogo bate — sinal para o caller recusar.
"""

from __future__ import annotations

import re
import unicodedata


# Chaves canónicas (singular, snake_case, sem acentos).
AREAS_VALIDAS: frozenset[str] = frozenset({
    # Dentária
    "implantes",
    "all_on",
    "protese",
    "ortodontia",
    "facetas",
    "branqueamento",
    "periodontia",
    "endodontia",
    "cirurgia_oral",
    # Estética facial
    "botox",
    "hialuronico",
    "harmonizacao_facial",
    # Capilar
    "transplante_capilar",
    "fue",
    # Pacote
    "turismo_dentario",
})


# Cada entrada: regex compilado → chave canónica.
# Ordem importa: padrões mais específicos primeiro para evitar falsos positivos
# (ex.: "all-on-4" antes de "implante", "harmonização facial" antes de "facetas").
_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # All-on (variantes: all on 4, all-on-6, allon, all on x)
    (re.compile(r"\ball[\s\-]?on[\s\-]?\d*\b"), "all_on"),
    # Implantes (cobre "implante dentário", "implantes")
    (re.compile(r"\bimplante?s?\b"), "implantes"),
    # Próteses (cobre "prótese fixa/removível", "dentadura")
    (re.compile(r"\b(prote(s|c)e?s?|dentadura)\b"), "protese"),
    # Ortodontia (alinhadores, invisalign, aparelho)
    (re.compile(r"\b(ortodontia|ortodont(ic|ist)\w*|alinhador(es)?|invisalign|aparelho)\b"), "ortodontia"),
    # Facetas / lentes de contacto dentárias
    (re.compile(r"\b(facetas?|lentes?\s+de\s+contacto)\b"), "facetas"),
    # Branqueamento / clareamento
    (re.compile(r"\b(branqueamento|clareamento)\b"), "branqueamento"),
    # Periodontia (gengivas)
    (re.compile(r"\b(periodontia|gengiv\w*)\b"), "periodontia"),
    # Endodontia / canal
    (re.compile(r"\b(endodontia|canal|desvitaliza\w*)\b"), "endodontia"),
    # Cirurgia oral
    (re.compile(r"\b(cirurgia\s+oral|extra(c|ç)ao|siso)\b"), "cirurgia_oral"),
    # Botox / toxina botulínica
    (re.compile(r"\b(botox|toxina\s+botulinic\w*)\b"), "botox"),
    # Ácido hialurónico / preenchimento facial
    (re.compile(r"\b(hialuronic\w*|preenchimento\s+facial)\b"), "hialuronico"),
    # Harmonização facial
    (re.compile(r"\b(harmoniza\w*\s+facial|harmoniza\w*)\b"), "harmonizacao_facial"),
    # FUE (técnica de transplante capilar)
    (re.compile(r"\bfue\b"), "fue"),
    # Transplante capilar / alopecia / queda cabelo / calva
    (re.compile(r"\b(transplante\s+capilar|capilar|alopecia|queda\s+(de\s+)?cabelo|calv\w*)\b"), "transplante_capilar"),
    # Turismo dentário (paciente do estrangeiro)
    (re.compile(r"\b(turismo\s+dentari\w*|pacote\s+dentari\w*)\b"), "turismo_dentario"),
]


def _strip_accents(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )


def normalize_area(raw: str | None) -> str | None:
    """Mapeia uma string livre para uma chave do catálogo, ou None se não houver match.

    Aceita case-insensitive, com/sem acentos, formato com espaços ou underscores.
    Devolve None para:
      - input vazio ou None
      - serviços fora do âmbito do Instituto Areluna
        (urologia, cirurgia plástica geral, podologia, etc.)
    """
    if not raw:
        return None

    norm = _strip_accents(raw.lower().strip().replace("_", " "))
    if not norm:
        return None

    # Match directo se já vier no formato canónico (com underscore restaurado).
    canonical_candidate = raw.lower().strip().replace(" ", "_").replace("-", "_")
    if canonical_candidate in AREAS_VALIDAS:
        return canonical_candidate

    # Procura por padrões na string normalizada.
    for pattern, key in _PATTERNS:
        if pattern.search(norm):
            return key

    return None
