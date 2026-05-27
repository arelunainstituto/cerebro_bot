"""specialist — dúvidas técnicas sobre clínica/tratamentos, com RAG.

Espelha `api/bot/agents/specialist.js` numa versão Python condensada:
  - Retrieval: pgvector via RPC `match_bot_kb`, top-K=4.
  - Cobertura: `bot_kb_documents` por (`phone_id`, `namespace`).
  - Embedding: HF Inference API (paraphrase-multilingual-MiniLM-L12-v2, 384d).
  - LLM blindado: responde APENAS com base nos documentos. Se a KB não cobrir,
    pede desculpa e marca `transfer=true` (Talita resolve na avaliação).

Namespace resolution:
  1. `state.current_namespace` (se preenchido)
  2. `state.qualification_state.area_interesse` (implantes|ortodontia|…)
  3. `"geral"` (fallback)

TODOs (próximas fases):
  - Cache 5min para queries repetidas (paridade JS `_lib/kb.js`).
  - Re-ranking por similarity threshold (filtrar chunks < 0.5).
  - Múltiplos namespaces simultâneos (lead com áreas múltiplas).
"""

from __future__ import annotations

import logging
import os

from langchain_core.runnables import RunnableConfig

from core.database import fetch_bot_kb
from core.embeddings import get_hf_embedding
from core.llm import invoke_structured
from graph.state import BotState
from schemas import SpecialistOutput

log = logging.getLogger(__name__)

_KB_TOPK = 4
_FALLBACK_NAMESPACE = "geral"

# Mapping defensivo: valores que o triage pode pôr em area_interesse mas que
# não são namespaces de bot_kb_documents. Fall-thru para "geral" se não bater.
_VALID_NAMESPACES = {
    "implantes", "ortodontia", "estetica_facial",
    "transplante_capilar", "turismo_dentario", "geral",
}
_AREA_TO_NAMESPACE = {
    "implantes": "implantes",
    "alinhadores": "ortodontia",
    "ortodontia": "ortodontia",
    "aparelho": "ortodontia",
    "facetas": "estetica_facial",
    "estetica_dentaria": "estetica_facial",
    "estetica": "estetica_facial",
    "botox": "estetica_facial",
    "harmonizacao": "estetica_facial",
    "transplante_capilar": "transplante_capilar",
    "fue": "transplante_capilar",
    "capilar": "transplante_capilar",
}


def _resolve_namespace(state: BotState) -> str:
    if state.current_namespace and state.current_namespace in _VALID_NAMESPACES:
        return state.current_namespace
    area = state.qualification_state.area_interesse
    if area:
        key = area.lower().strip()
        if key in _VALID_NAMESPACES:
            return key
        if key in _AREA_TO_NAMESPACE:
            return _AREA_TO_NAMESPACE[key]
    return _FALLBACK_NAMESPACE


_SPECIALIST_SYSTEM = """És o Especialista Clínico do Instituto Areluna (clínica dentária e estética avançada, no Porto). A tua função é esclarecer dúvidas técnicas sobre procedimentos, materiais, indicações e pós-operatório.

REGRAS ABSOLUTAS:
🔴 RESPONDE EXCLUSIVAMENTE com base nos «Documentos de Referência» abaixo. Não inventes, não suponhas, não cites fontes externas.
🔴 Se a resposta NÃO estiver nos documentos: pede desculpa brevemente, diz que a equipa clínica (Talita Alves) validará essa dúvida na avaliação online gratuita, e marca `transfer=true` no output.
🔴 NUNCA dês preços, prazos ou diagnósticos clínicos. Se o lead perguntar isso, redirecciona à avaliação com a Talita.
🔴 NUNCA digas "Dr.", "doutor" ou "médico". A pessoa que avalia é a **Talita Alves, Gestora de Pacientes Especialista**.
🔴 NUNCA dizes que és bot/IA/automática.
🔴 PT-PT estrito: "tu", "diz-me", "consegues". PROIBIDO "você" e regionalismos BR.
🔴 PROIBIDO travessão (— ou –). Usa ponto/vírgula/parênteses.
🔴 Tom: profissional, calmo, conciso. 2-4 frases. Uma só resposta — UMA pergunta no máximo se for esclarecimento.

OUTPUT (JSON estrito):
{
  "reply": ["texto PT-PT, máximo 4 frases, blindado pelos documentos"],
  "kb_used": <inteiro: nº de documentos efectivamente citados>,
  "transfer": false | true,
  "transfer_reason": null | "out_of_kb"
}

`kb_used` deve reflectir HONESTAMENTE quantos chunks contribuíram para a resposta. Se respondeste "não sei" e marcaste `transfer=true`, `kb_used=0`."""


def _format_docs(docs: list[dict]) -> str:
    if not docs:
        return "(nenhum documento relevante)"
    parts = []
    for i, d in enumerate(docs, start=1):
        src = d.get("source") or "(sem fonte)"
        sim = d.get("similarity", 0.0)
        content = (d.get("content") or "").strip()
        parts.append(f"--- Documento {i} (fonte={src}, similarity={sim:.3f}) ---\n{content}")
    return "\n\n".join(parts)


async def specialist_node(state: BotState, config: RunnableConfig) -> BotState:
    state.agent_used = "specialist"
    pool = config["configurable"]["pool"]

    namespace = _resolve_namespace(state)
    state.current_namespace = namespace  # propaga para enter/respond ler/persistir

    question = state.incoming_message or ""
    docs: list[dict] = []

    # --- Retrieval (embedding + RPC) ---
    try:
        embedding = await get_hf_embedding(question)
        docs = await fetch_bot_kb(
            pool,
            embedding,
            phone_id=state.phone_id,
            namespace=namespace,
            match_count=_KB_TOPK,
        )
        log.info(
            "[specialist] retrieval ns=%s docs=%d top_sim=%.3f",
            namespace,
            len(docs),
            docs[0]["similarity"] if docs else 0.0,
        )
    except Exception as e:
        # Falha de retrieval (HF cold start, pgvector erro, …) não bloqueia o
        # turno. Deixamos o LLM responder sem KB → vai marcar transfer=true.
        log.warning("[specialist] retrieval falhou ns=%s: %s", namespace, e)
        docs = []

    # --- LLM blindado ---
    user_prompt = (
        f"Pergunta do lead:\n{question}\n\n"
        f"Documentos de Referência ({namespace}):\n{_format_docs(docs)}"
    )

    try:
        out: SpecialistOutput = await invoke_structured(
            SpecialistOutput,
            [
                {"role": "system", "content": _SPECIALIST_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            model=os.getenv("BOT_SPECIALIST_MODEL", "gpt-4.1-mini"),
            temperature=0.2,
            max_tokens=500,
            schema_name="specialist_output",
        )
    except Exception as e:
        log.exception("[specialist] LLM falhou: %s", e)
        state.reply = [
            "Peço desculpa, fiquei sem ligação um instante. Consegues reformular a pergunta?"
        ]
        return state

    reply_msgs = [s for s in (out.reply or []) if s and s.strip()]
    state.reply = reply_msgs or [
        "Peço desculpa, não tenho essa informação documentada. A Talita esclarece isso contigo na avaliação online."
    ]

    if out.transfer:
        state.transfer = True
        state.transfer_reason = out.transfer_reason or "specialist_out_of_kb"

    log.info(
        "[specialist] ns=%s kb_used=%d transfer=%s reply_len=%d",
        namespace,
        out.kb_used,
        out.transfer,
        sum(len(r) for r in state.reply),
    )
    return state
