"""scheduling — máquina de estados de calendário (available → held → booked).

Espelha `api/bot/agents/scheduling.js` em versão condensada para Python:
  - Fast-path inline affirmation: se há `_pending_confirm` e a mensagem é uma
    afirmação clara (`ok / sim / pode marcar / combinado`) → book direto SEM LLM.
  - Fast-path negativa: se há `_pending_confirm` e a mensagem é claramente
    negativa → libertar hold + cair através para LLM com slots frescos.
  - LLM decide action ∈ {propose, confirm, booked, nothing} via
    `SchedulingOutput` (Pydantic structured output).
  - Hold race: se outro lead apanhou o slot no milissegundo anterior, fallback
    determinístico ("apanhado por outra pessoa, eis outros").
  - Booking atómico: transacção UPDATE slot + INSERT appointment.

TODOs documentados em [core/calendar.py](core/calendar.py):
  - Integração real Google Calendar Service Account.
"""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from langchain_core.runnables import RunnableConfig

from core.calendar import create_gcal_event
from core.database import (
    book_slot_db,
    fetch_available_slots,
    hold_slot,
    release_hold,
)
from core.errors import SlotUnavailableError
from core.llm import invoke_structured
from graph.state import BotState
from schemas import SchedulingOutput

log = logging.getLogger(__name__)

_TZ = ZoneInfo("Europe/Lisbon")
_HOLD_TTL_MIN = 30
_FETCH_DAYS = 7
_FETCH_LIMIT = 12

# Paridade api/bot/agents/scheduling.js linhas 167-174.
_INLINE_AFFIRMATION_RE = re.compile(
    r"\b(ok+|okay|sim|pode\s+(?:marcar|ser|reservar|agendar)|confirmo|confirma|"
    r"certo|fechad[oa]|fechou|combinado|aceito|perfeit[oa]|[oó]tim[oa]|feito|"
    r"cool|seguir|por\s+favor|please|isso\s+mesmo|exato|certinho|t[áa]\s+bom|"
    r"t[áa]\s+ok)\b",
    re.IGNORECASE,
)
_QUESTION_WORDS_RE = re.compile(
    r"\b(quanto|qual\s+(?:o\s+)?valor|pre[çc]o|custa|custo|talvez|"
    r"quem\s+sabe|n[ãa]o\s+sei)\b",
    re.IGNORECASE,
)
# Paridade api/bot/agents/scheduling.js linha 137.
_NEGATIVE_RE = re.compile(
    r"\b(n[ãa]o|cancela|cancelar|esquece|deixa\s*pra\s*l[áa]|outro\s*hor[áa]rio|"
    r"outra\s*hora|outro\s*dia|nada\s*disso|melhor\s*n[ãa]o)\b",
    re.IGNORECASE,
)


def _has_inline_affirmation(msg: str) -> bool:
    if not msg:
        return False
    t = msg.strip()
    if t.endswith("?"):
        return False
    if _QUESTION_WORDS_RE.search(t):
        return False
    return bool(_INLINE_AFFIRMATION_RE.search(t))


def _is_negative(msg: str) -> bool:
    return bool(msg and _NEGATIVE_RE.search(msg))


# ----------------------------------------------------------------------------
# Formatação PT-PT manual (locale-independente; macOS pode não ter pt_PT)
# ----------------------------------------------------------------------------


_WEEKDAYS_PT = [
    "Segunda-feira", "Terça-feira", "Quarta-feira", "Quinta-feira",
    "Sexta-feira", "Sábado", "Domingo",
]
_MONTHS_PT = [
    "janeiro", "fevereiro", "março", "abril", "maio", "junho",
    "julho", "agosto", "setembro", "outubro", "novembro", "dezembro",
]


def _format_dt_pt(dt: datetime) -> str:
    """Ex.: 'Quarta-feira 28 de maio, 14:00'."""
    local = dt.astimezone(_TZ)
    return (
        f"{_WEEKDAYS_PT[local.weekday()]} {local.day} de {_MONTHS_PT[local.month - 1]}, "
        f"{local.strftime('%H:%M')}"
    )


def _format_slots_pt(slots: list[dict[str, Any]]) -> str:
    if not slots:
        return "(sem horários disponíveis nos próximos dias — pede para tentar mais tarde)"
    lines = []
    for i, s in enumerate(slots, start=1):
        when = _format_dt_pt(s["slot_start_dt"])
        iso = s["slot_start_iso"]
        lines.append(f"{i}. {when}  (ISO={iso})")
    return "\n".join(lines)


def _format_slots_pt_sms(slots: list[dict[str, Any]], max_items: int = 3) -> str:
    """Variante curta para SMS — até 3 opções, sem ISO, sem hifens longos."""
    if not slots:
        return "(sem horários disponíveis nos próximos dias)"
    lines = []
    for i, s in enumerate(slots[:max_items], start=1):
        iso = s["slot_start_iso"]
        # Curto: sem dia da semana extenso. Ex.: "28 maio, 14:00"
        local = s["slot_start_dt"].astimezone(_TZ)
        short_when = f"{local.day} {_MONTHS_PT[local.month - 1]}, {local.strftime('%H:%M')}"
        lines.append(f"{i}. {short_when} (ISO={iso})")
    return "\n".join(lines)


def _format_iso_pt(iso: str) -> str:
    """Fallback quando só temos o ISO (sem objecto datetime já parseado)."""
    try:
        return _format_dt_pt(datetime.fromisoformat(iso))
    except Exception:
        return iso


def _valid_iso(iso: str | None, slots: list[dict[str, Any]]) -> bool:
    """Defesa contra alucinação: aceita só ISOs presentes na lista oferecida."""
    if not iso:
        return False
    valid = {s["slot_start_iso"] for s in slots}
    return iso in valid


# ----------------------------------------------------------------------------
# LLM
# ----------------------------------------------------------------------------


_SCHEDULING_SYSTEM = """És a assistente de agendamento do Instituto Areluna. O teu único objectivo é marcar uma AVALIAÇÃO ONLINE (videochamada WhatsApp, ~30 min, totalmente GRATUITA) com a **Talita Alves**, Gestora de Pacientes Especialista.

REGRAS ABSOLUTAS:
🔴 USA APENAS os HORÁRIOS DISPONÍVEIS abaixo. Nunca inventes outras horas. `slot_iso` no output TEM de ser EXACTAMENTE um dos ISOs listados.
🔴 Se o lead aceita um horário CLARAMENTE ("sim quarta às 14h", "ok pode marcar", "fechado para amanhã") → action="confirm" + slot_iso desse horário.
🔴 Se já existe `_pending_confirm` (slot proposto à espera) e o lead diz "sim/ok/combinado" → action="confirm" com slot_iso = pending.
🔴 Se sugeres um horário (e o lead ainda não confirmou explicitamente) → action="propose" + slot_iso.
🔴 Se o lead recusa o slot pendente ("não, prefiro outra hora") → action="propose" com um slot diferente da lista.
🔴 Se o lead faz uma pergunta lateral (preço, processo, técnica) → action="nothing" + reply curto. O preço a Talita esclarece NA videochamada.
🔴 NUNCA "Dr.", "doutor", "médico". A pessoa que avalia é a **Talita Alves**.
🔴 PT-PT estrito ("tu", "consegues", "diz-me"). PROIBIDO "você", regionalismos BR, travessões (— ou –).
🔴 1-2 frases por mensagem. Tom directo, amável, profissional. NUNCA inventes confirmações que ainda não aconteceram.

OUTPUT JSON estrito (sem texto fora do JSON):
{
  "reply": ["mensagem PT-PT curta"],
  "action": "propose"|"confirm"|"booked"|"nothing",
  "slot_iso": "<ISO 8601 com TZ>" | null,
  "calendar_event_id": null,
  "transfer": false,
  "transfer_reason": null
}"""


def _build_user_prompt(
    state: BotState,
    slots: list[dict[str, Any]],
    pending: str | None,
) -> str:
    now_pt = datetime.now(_TZ).strftime("%A, %d de %B de %Y, %H:%M")
    qual = state.qualification_state.model_dump(exclude_none=True)
    mem_tail = state.short_memory[-4:] if state.short_memory else []
    is_sms = state.channel == "sms"

    slots_block = _format_slots_pt_sms(slots) if is_sms else _format_slots_pt(slots)
    slots_header = (
        f"HORÁRIOS DISPONÍVEIS (até 3, próximos {_FETCH_DAYS} dias):"
        if is_sms
        else f"HORÁRIOS DISPONÍVEIS (próximos {_FETCH_DAYS} dias):"
    )

    parts = [
        f"AGORA (Europe/Lisbon): {now_pt}",
        f"Canal: {state.channel}",
        f"Nome do lead: {qual.get('nome') or '(desconhecido)'}",
        f"Área de interesse: {qual.get('area_interesse') or '(não definida)'}",
        f"Pendente de confirmação: {pending or '(nenhum)'}",
        "",
        slots_header,
        slots_block,
        "",
        f'Mensagem do lead: "{state.incoming_message}"',
    ]
    if is_sms:
        parts.extend([
            "",
            "🟡 CANAL = SMS — REGRAS ADICIONAIS:",
            "- Máximo 160 caracteres no reply.",
            "- Sem emojis, sem markdown, sem travessões, sem listas numeradas longas.",
            "- Propõe no máximo 2 horários em texto corrido. Ex.: 'Posso marcar 28 maio 14:00 ou 29 maio 10:30. Qual preferes?'",
        ])
    if mem_tail:
        hist = "\n".join(
            f"{'Lead' if t.role == 'user' else 'Bot'}: {t.content}" for t in mem_tail
        )
        parts += ["", f"Conversa recente:\n{hist}"]
    return "\n".join(parts)


async def _invoke_llm(
    state: BotState,
    slots: list[dict[str, Any]],
    pending: str | None,
) -> SchedulingOutput:
    user_prompt = _build_user_prompt(state, slots, pending)
    return await invoke_structured(
        SchedulingOutput,
        [
            {"role": "system", "content": _SCHEDULING_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        model=os.getenv("BOT_SCHEDULING_MODEL", "gpt-4o-mini"),
        temperature=0.2,
        max_tokens=400,
        schema_name="scheduling_output",
    )


# ----------------------------------------------------------------------------
# Booking
# ----------------------------------------------------------------------------


async def _do_book(
    state: BotState,
    pool,
    slot_iso: str,
    *,
    fallback_reply: list[str] | None = None,
) -> BotState:
    """Reserva atómica + gcal mock + reply de confirmação cordial."""
    name = state.qualification_state.nome or "Lead"
    treatment = state.qualification_state.area_interesse or "avaliação online"
    try:
        gcal_id = await create_gcal_event(
            slot_start_iso=slot_iso,
            slot_end_iso=None,
            contact_name=name,
            treatment=treatment,
        )
        appt_id = await book_slot_db(
            pool,
            slot_start_iso=slot_iso,
            phone_id=state.phone_id,
            contact_phone=state.contact_phone,
            contact_name=name,
            treatment=treatment,
            gcal_event_id=gcal_id,
        )
    except SlotUnavailableError:
        state.state_extras.pop("_pending_confirm", None)
        state.reply = fallback_reply or [
            "Esse horário deixou de estar disponível enquanto confirmavas. Vou propor-te outros."
        ]
        log.info("[scheduling] slot %s indisponível na hora do book", slot_iso)
        return state
    except Exception as e:
        log.exception("[scheduling] book falhou: %s", e)
        state.transfer = True
        state.transfer_reason = "scheduling_book_error"
        state.reply = [
            "Peço desculpa, tive um problema técnico a confirmar a marcação. Vou pedir à Talita para te contactar diretamente."
        ]
        return state

    state.state_extras.pop("_pending_confirm", None)
    state.state_extras["_booked"] = {
        "slot_iso": slot_iso,
        "appt_id": appt_id,
        "gcal_id": gcal_id,
    }
    state.pop_step = 9  # booked
    when_pt = _format_iso_pt(slot_iso)
    first = name.split()[0] if name else None
    opener = f"Perfeito, {first}!" if first else "Perfeito!"
    state.reply = [
        f"{opener} A tua videochamada está agendada para {when_pt} (horário de Portugal). "
        f"A Talita Alves vai ligar-te neste mesmo número de WhatsApp à hora marcada para a tua avaliação online, totalmente gratuita. Até lá!"
    ]
    log.info(
        "[scheduling] booked slot=%s appt_id=%s gcal=%s contact=%s",
        slot_iso, appt_id, gcal_id, state.contact_phone,
    )
    return state


# ----------------------------------------------------------------------------
# Nó principal
# ----------------------------------------------------------------------------


async def scheduling_node(state: BotState, config: RunnableConfig) -> BotState:
    state.agent_used = "scheduling"
    pool = config["configurable"]["pool"]
    msg = state.incoming_message or ""

    # 1) Carregar slots disponíveis
    slots = await fetch_available_slots(pool, days=_FETCH_DAYS, limit=_FETCH_LIMIT)
    pending = state.state_extras.get("_pending_confirm") if state.state_extras else None

    # 2) Fast-path inline affirmation → book direto (sem LLM)
    if pending and _has_inline_affirmation(msg) and not _is_negative(msg):
        log.info("[scheduling] fast-path: inline_affirmation com pending=%s", pending)
        return await _do_book(state, pool, pending)

    # 3) Fast-path negativa: liberta hold + segue para LLM
    if pending and _is_negative(msg):
        log.info("[scheduling] fast-path: negative — release hold %s", pending)
        await release_hold(pool, pending, state.phone_id)
        state.state_extras.pop("_pending_confirm", None)
        pending = None

    # 4) LLM decide
    try:
        out: SchedulingOutput = await _invoke_llm(state, slots, pending)
    except Exception as e:
        log.exception("[scheduling] LLM falhou: %s", e)
        state.reply = [
            "Peço desculpa, fiquei sem rede um instante. Tenta de novo dizer-me a hora que preferes."
        ]
        return state

    # 5) Despachar acção
    fmt_slots = _format_slots_pt_sms if state.channel == "sms" else _format_slots_pt
    if out.action == "propose":
        if not _valid_iso(out.slot_iso, slots):
            log.warning("[scheduling] LLM propôs slot inválido: %r — fallback re-propor", out.slot_iso)
            state.reply = [
                "Olha, estes são os horários que tenho disponíveis. Diz-me qual te dá jeito:",
                fmt_slots(slots),
            ]
            return state
        ok = await hold_slot(pool, out.slot_iso, state.phone_id, ttl_minutes=_HOLD_TTL_MIN)
        if not ok:
            log.info("[scheduling] hold perdido — race")
            fresh = await fetch_available_slots(pool, days=_FETCH_DAYS, limit=_FETCH_LIMIT)
            state.reply = [
                "Esse horário acabou de ser apanhado por outra pessoa. Estes ficaram disponíveis:",
                fmt_slots(fresh),
            ]
            return state
        state.state_extras["_pending_confirm"] = out.slot_iso
        state.pop_step = 8  # confirmacao
        state.reply = out.reply or [f"Vou reservar-te {_format_iso_pt(out.slot_iso)}. Confirmas? (responde 'sim')"]
        log.info("[scheduling] proposed slot=%s held OK", out.slot_iso)

    elif out.action == "confirm":
        # Aceita slot_iso do LLM OU usa o pending guardado.
        slot_iso = out.slot_iso if _valid_iso(out.slot_iso, slots) else pending
        if not slot_iso:
            # LLM disse confirm mas não há slot identificável — pede esclarecimento
            state.reply = [
                "Qual desses horários preferes? Diz-me o número ou a hora exacta."
            ]
            log.warning("[scheduling] confirm sem slot — pediu clarificação")
            return state
        return await _do_book(state, pool, slot_iso, fallback_reply=out.reply)

    elif out.action == "booked":
        # Marcação já fechada anteriormente. Só ecoa reply.
        state.reply = out.reply or ["A tua avaliação já está marcada. A Talita ligar-te-á à hora combinada."]

    else:  # "nothing" — pergunta lateral, sem mudar estado de calendário
        state.reply = out.reply or ["Posso ajudar-te a marcar a videochamada com a Talita?"]

    if out.transfer:
        state.transfer = True
        state.transfer_reason = out.transfer_reason or "scheduling_transfer"

    return state
