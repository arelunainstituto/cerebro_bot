"""bypass — atalhos antes do router.

Ordem rigorosa:
  1. Opt-out lookup em `bot_opt_outs` (hard opt-out → silêncio absoluto).
  2. Fast-path regex (`detect_bypass`) com 6 grupos (emergency, human_request,
     reception_service, wrong_number, no_interest, satisfied).
     - Override: `reception_service` mas mensagem cita tratamento high-value
       cai através (lead pede caso composto, segue POP).
     - Mapping `satisfied → already_patient` (paridade JS bypassNode).
  3. Smart silence (atalho extra do Python): emoji isolado ou ack curto.
     ⚠️ Reply suppression real fica para próxima fase (precisa de respond.py).
  4. Skip LLM se `wordCount ≤ 2` e há fluxo activo (poupa custo/latência).
  5. LLM classifier (`IntentClassification`, gpt-4o-mini default).
  6. Guard `already_patient` quando `qualification_state.ja_e_paciente='não'`.
  7. Dispatch só se `confidence ≥ 0.65` e tipo em white-list.

Sinais de saída:
  - `state.bypass_reason`     → ordena ao edge para saltar router e ir a respond
  - `state.transfer`          → True para emergency/human_request
  - `state.transfer_reason`   → motivo (string)
  - `state.state_extras["_pending_opt_out"]` → para futura persistência
  - `state.state_extras["_smart_silence"]`   → para futura supressão de reply

TODOs (fora de escopo desta fase):
  - Sticky `_reception_only` flag handling.
  - Reply suppression real para smart_silence / hard_opt_out (1 linha em respond).
  - Persistência em bot_opt_outs via `upsert_opt_out` (helper já criado).
"""

from __future__ import annotations

import logging
import os
import re

from langchain_core.runnables import RunnableConfig

from core.database import load_opt_out
from core.llm import invoke_structured
from graph.state import BotState
from schemas import IntentClassification

log = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Regex (paridade api/bot/_lib/handoff.js)
# ----------------------------------------------------------------------------


_EMERGENCY_KEYWORDS = [
    "dor forte", "dor muito forte", "sangramento", "sangrando",
    "urgente", "urgência", "emergencia", "emergência",
    "inchado", "inchaço",
]

_HUMAN_KEYWORDS = [
    "falar com humano", "falar com pessoa", "falar com alguém", "falar com alguem",
    "atendente", "recepcionista", "rececionista", "secretária", "secretaria",
    "não és humano", "nao es humano", "és um robô", "es um robo",
    "quero falar com", "pode me ligar", "liga para mim", "me liga",
]

_SERVICE_KEYWORD = r"(?:limp(?:e[zs]a|ea)|profilaxia|destartar(?:iza[çc][aã]o)?|restaura[çc][aã]o|restaur(?:ar|ação)?|t[áa]rtaro)"
_INTENT_VERBS = r"(?:s[oó]|apenas|gostari?a|queri?a|quero|preciso|venho|fazer)"

_RECEPTION_SERVICE_PATTERNS = [
    re.compile(rf"\b{_INTENT_VERBS}\b[^.!?]{{0,40}}\b{_SERVICE_KEYWORD}\b", re.IGNORECASE),
    re.compile(rf"^\s*{_SERVICE_KEYWORD}\s*[!.?]*$", re.IGNORECASE),
    re.compile(r"\b(?:tirar|fazer)\s+(?:o\s+)?t[áa]rtaro\b", re.IGNORECASE),
    re.compile(r"\bclareamento\s+(?:simples|de\s+rotina|dent[áa]rio)\b", re.IGNORECASE),
]

# Se mensagem cita tratamento high-value, `reception_service` cai através.
_RECEPTION_HIGH_VALUE = re.compile(
    r"\b(implant|pr[oó]tese|all[\s-]?on|alinhad|invisalign|ortodont|faceta|porcelana|"
    r"lente\s+de?\s+contact|est[eé]tic\w*\s+(?:facial|dent)|botox|hialur|harmoniza|"
    r"transplante\s+capilar|fue\b|alopecia|enxerto)",
    re.IGNORECASE,
)

_WRONG_NUMBER_PATTERNS = [
    re.compile(r"\bn[aã]o\s+(?:sou|s[oôó])\s+(?:[ao]\s+)?\w+", re.IGNORECASE),
    re.compile(r"\bn[aã]o\s+conhe[çc]o\s+(?:essa\s+)?(?:pessoa|ningu[eé]m)", re.IGNORECASE),
    re.compile(r"\bn[uú]mero\s+(?:errad[oa]|incorreto)", re.IGNORECASE),
    re.compile(r"\benga(?:no|nei)\s+(?:de\s+n[uú]mero)?", re.IGNORECASE),
    re.compile(r"\b(?:isto|isso|essa\s+mensagem)\s+(?:n[aã]o\s+)?(?:é\s+)?(?:para\s+mim|n[aã]o\s+é\s+(?:para\s+)?mim)", re.IGNORECASE),
    re.compile(r"\b(?:n[aã]o\s+é|nao\s+e)\s+comigo", re.IGNORECASE),
    re.compile(r"\bse\s+(?:enganaram|enganou)\b", re.IGNORECASE),
    re.compile(r"\bnunca\s+(?:fui|fiz)\s+(?:paciente|cliente)", re.IGNORECASE),
]

_NO_INTEREST_PATTERNS = [
    re.compile(r"\bn[aã]o\s+(?:me\s+)?(?:liguem|contactem|mandem|enviem|escrevam)\s+mais\b", re.IGNORECASE),
    re.compile(r"\bn[aã]o\s+quero\s+(?:receber|mais|ser\s+contact)", re.IGNORECASE),
    re.compile(r"\bpar[ae]m?\s+de\s+(?:me\s+)?(?:mandar|enviar|ligar|contactar|incomodar)", re.IGNORECASE),
    re.compile(r"\bdeixem.?me\s+em\s+paz", re.IGNORECASE),
    re.compile(r"\bn[aã]o\s+(?:tenho|me)\s+interesse(?:m)?\b", re.IGNORECASE),
    re.compile(r"\bremov(?:am|er|a)(?:-me)?\s+(?:o\s+meu\s+)?(?:n[uú]mero|contacto)?(?:\s+da\s+lista)?", re.IGNORECASE),
    re.compile(r"\bcancel(?:em|ar)\s+(?:as\s+)?(?:mensagens|envios|campanha)", re.IGNORECASE),
    re.compile(r"\bsair\s+da\s+(?:lista|base)", re.IGNORECASE),
    re.compile(r"\bunsubscribe\b", re.IGNORECASE),
    re.compile(r"\bspam\b", re.IGNORECASE),
]

_SATISFIED_PATTERNS = [
    re.compile(r"n[aã]o\s+(?:tem|h[áa])\s+necessidade", re.IGNORECASE),
    re.compile(r"estou\s+satisfeit[oa]", re.IGNORECASE),
    re.compile(r"est[áa]\s+tudo\s+(?:bem|perfeito|ok)", re.IGNORECASE),
    re.compile(r"tudo\s+(?:perfeito|certo|bem)\s*[!.😊☺️🙏🏻]*$", re.IGNORECASE),
    re.compile(r"n[aã]o\s+precis[oa]\s+(?:de\s+)?(?:nada|marca[rç])", re.IGNORECASE),
    re.compile(r"obrigad[oa]\s+na\s+mesma", re.IGNORECASE),
    re.compile(r"j[aá]\s+sou\s+(?:vossa|vosso|paciente|cliente)", re.IGNORECASE),
    re.compile(r"j[aá]\s+(?:estou|fa[çc]o|fa[çc]os)\s+(?:em\s+)?tratamento", re.IGNORECASE),
]


def detect_bypass(text: str) -> dict[str, str] | None:
    """Devolve `{type, matched}` ou `None`. Paridade `handoff.js#detectBypass`.

    Ordem importa: emergency > human_request > reception_service > wrong_number
    > no_interest > satisfied. `reception_service` antes de wrong/no_interest
    porque "só quero uma limpeza" é interesse comercial legítimo.
    """
    if not text:
        return None
    lower = text.lower()
    for k in _EMERGENCY_KEYWORDS:
        if k in lower:
            return {"type": "emergency", "matched": k}
    for k in _HUMAN_KEYWORDS:
        if k in lower:
            return {"type": "human_request", "matched": k}
    for p in _RECEPTION_SERVICE_PATTERNS:
        m = p.search(text)
        if m:
            return {"type": "reception_service", "matched": m.group(0)}
    for p in _WRONG_NUMBER_PATTERNS:
        m = p.search(text)
        if m:
            return {"type": "wrong_number", "matched": m.group(0)}
    for p in _NO_INTEREST_PATTERNS:
        m = p.search(text)
        if m:
            return {"type": "no_interest", "matched": m.group(0)}
    for p in _SATISFIED_PATTERNS:
        m = p.search(text)
        if m:
            return {"type": "satisfied", "matched": m.group(0)}
    return None


# ----------------------------------------------------------------------------
# Smart silence
# ----------------------------------------------------------------------------


_EMOJI_ONLY = re.compile(r"^\s*(?:👍|👌|🙏|✅|❤️|😊|🙂|👋|🆗|🙆|🙋)+\s*$")
_SHORT_ACK = re.compile(r"^\s*(?:ok+|okay|k+|sim|yes|ya|claro|certo|combinado)\s*[!.😊🙏👍]*\s*$", re.IGNORECASE)


def _is_smart_silence(text: str) -> bool:
    return bool(_EMOJI_ONLY.match(text)) or bool(_SHORT_ACK.match(text))


# ----------------------------------------------------------------------------
# LLM intent classifier
# ----------------------------------------------------------------------------


_INTENT_SYSTEM = """És um classificador de intenção do bot do Instituto Areluna (clínica dentária e estética avançada, no Porto). A tua única tarefa é classificar a mensagem do lead numa de 8 categorias.

CATEGORIAS:
- "emergency"          → urgência clínica AGORA (dor forte AGORA, sangramento activo, infecção, inchaço).
  ⚠️ NÃO classificar: "tenho dor há anos", "dor antiga", "tive uma dor", "tenho dor de vez em quando" (dor crónica/intermitente NÃO é emergency, é new_lead a procurar tratamento).
- "human_request"      → pede EXPLICITAMENTE para falar com PESSOA REAL (atendente, rececionista, "quero falar com alguém", "não quero IA").
  ⚠️ NÃO classificar: "podemos falar"/"vamos conversar"/"falamos depois" (continuar esta conversa, NÃO pedir humano).
- "wrong_number"       → não é a pessoa esperada, número errado, engano, "não é comigo", "não conheço".
- "no_interest"        → recusa de prospecção CLARA E DIRECTA ("não quero", "removam da lista", "parem de mandar", "unsubscribe", "spam").
  ⚠️ NÃO classificar: lead pergunta preço/tratamento (é interesse), "obrigado" isolado (cortesia), "não moro em PT" (objecção logística — turismo dentário).
- "already_patient"    → diz EXPLICITAMENTE que já é/foi paciente do Instituto Areluna.
  ⚠️ NÃO classificar: "tenho aparelho"/"fiz implante" (refere-se a NOUTRA clínica, é new_lead).
- "reception_service"  → quer APENAS rotina (limpeza, profilaxia, destartarização, branqueamento de rotina). NÃO inclui implantes/alinhadores/facetas/estética facial/transplante capilar.
- "new_lead"           → tudo o que NÃO é nenhum dos casos acima. Default seguro.
- "unknown"            → mensagem realmente ambígua.

REGRAS:
1. Default é "new_lead" — só sai dessa categoria quando há sinal CLARO.
2. Tolerante a typos PT-PT e PT-BR.
3. Considera o histórico curto e qualification_state — ajudam a desambiguar.
4. confidence ≥0.85 sinal muito claro; 0.6-0.85 razoável; <0.6 → "unknown".

Devolve APENAS JSON: {"type": "...", "confidence": 0.0-1.0, "reason": "explicação curta em PT-PT"}"""


async def _classify_intent(state: BotState) -> IntentClassification | None:
    mem_tail = state.short_memory[-4:] if state.short_memory else []
    qual = state.qualification_state.model_dump(exclude_none=True)
    user_parts = [f'Mensagem actual do lead: "{state.incoming_message}"']
    if qual:
        user_parts.append(f"Dados conhecidos do lead: {qual}")
    if mem_tail:
        hist = "\n".join(f"{'Lead' if t.role == 'user' else 'Bot'}: {t.content}" for t in mem_tail)
        user_parts.append(f"Conversa recente:\n{hist}")
    user_prompt = "\n\n".join(user_parts)
    try:
        return await invoke_structured(
            IntentClassification,
            [
                {"role": "system", "content": _INTENT_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            model=os.getenv("BOT_INTENT_MODEL", "gpt-4o-mini"),
            temperature=0.1,
            max_tokens=300,
            schema_name="intent_classification",
        )
    except Exception as e:
        log.warning("[bypass] intent classifier failed: %s", e)
        return None


# ----------------------------------------------------------------------------
# Nó
# ----------------------------------------------------------------------------


_OPT_OUTABLE = {"emergency", "human_request", "wrong_number", "no_interest", "already_patient", "reception_service"}
_TRANSFER_TYPES = {"emergency", "human_request"}
_PERSIST_OPT_OUT_TYPES = {"wrong_number", "no_interest", "already_patient"}


def _is_ja_paciente_no(state: BotState) -> bool:
    """Guard JS: lead já confirmou 'primeira vez'."""
    raw = state.qualification_state.ja_e_paciente
    if raw is False:
        return True
    if isinstance(raw, str):
        return raw.lower().strip() in {"não", "nao", "no"}
    return False


def _in_active_flow(state: BotState) -> bool:
    if state.qualification_state.nome:
        return True
    extras = state.state_extras or {}
    return bool(extras.get("_proposed_slots") or extras.get("_pending_confirm"))


async def bypass_node(state: BotState, config: RunnableConfig) -> BotState:
    pool = config["configurable"]["pool"]
    text = state.incoming_message or ""

    # 1) Opt-out lookup global por contact_phone
    opt_out = await load_opt_out(pool, state.contact_phone)
    if opt_out in ("wrong_number", "no_interest"):
        # Hard opt-out: silêncio absoluto. Reply ainda passa por respond stub
        # (suppression real fica para próxima fase).
        state.bypass_reason = f"hard_opt_out:{opt_out}"
        state.reply = []
        log.info("[bypass] hard_opt_out=%s contact=%s — silêncio", opt_out, state.contact_phone)
        return state
    if opt_out == "already_patient":
        state.bypass_reason = "opt_out:already_patient"
        state.state_extras["_opt_out_returning"] = "already_patient"
        log.info("[bypass] opt_out=already_patient contact=%s — short-circuit", state.contact_phone)
        return state

    # 2) Fast-path regex
    regex_hit = detect_bypass(text)
    if regex_hit:
        rtype = regex_hit["type"]
        matched = regex_hit["matched"]

        # Override: reception_service mas mensagem cita high-value → cai através
        if rtype == "reception_service" and _RECEPTION_HIGH_VALUE.search(text):
            log.info("[bypass] reception_service detectado mas high-value mencionado — cai através")
        else:
            # Mapping satisfied → already_patient (paridade JS)
            dispatched = "already_patient" if rtype == "satisfied" else rtype
            return _dispatch(state, dispatched, matched, source="regex")

    # 3) Smart silence (apenas se não houve regex hit)
    if _is_smart_silence(text):
        state.state_extras["_smart_silence"] = True
        state.bypass_reason = "smart_silence"
        log.info("[bypass] smart_silence: %r", text.strip())
        return state

    # 4) Skip LLM se mensagem ultra-curta dentro de fluxo activo (paridade JS)
    word_count = len([w for w in text.strip().split() if w])
    if word_count <= 2 and _in_active_flow(state):
        log.info("[bypass] skip LLM (wordCount=%d, in_active_flow=True)", word_count)
        return state

    # 5) LLM classifier
    intent = await _classify_intent(state)
    if intent is None:
        return state  # API falhou — segue ao router

    # 6) Guard already_patient quando lead já confirmou não-paciente
    if intent.type == "already_patient" and _is_ja_paciente_no(state):
        log.info("[bypass] guard: already_patient bloqueado (ja_e_paciente='não')")
        return state

    # 7) Dispatch só com confidence suficiente e tipo white-listado
    if intent.confidence >= 0.65 and intent.type in _OPT_OUTABLE:
        return _dispatch(
            state, intent.type, intent.reason or intent.type, source=f"llm:{intent.confidence:.2f}"
        )

    log.info(
        "[bypass] LLM intent=%s conf=%.2f (sem dispatch) — segue ao router",
        intent.type, intent.confidence,
    )
    return state


def _dispatch(state: BotState, intent_type: str, matched: str, *, source: str) -> BotState:
    """Atalha o turno e prepara state para o nó `respond`."""
    state.bypass_reason = intent_type

    if intent_type in _PERSIST_OPT_OUT_TYPES:
        # Flag para fase seguinte persistir em bot_opt_outs via respond
        state.state_extras["_pending_opt_out"] = {
            "type": intent_type,
            "matched_text": matched,
            "source": source,
        }

    if intent_type in _TRANSFER_TYPES:
        state.transfer = True
        state.transfer_reason = intent_type

    log.info("[bypass] dispatch type=%s source=%s matched=%r", intent_type, source, matched[:80])
    return state
