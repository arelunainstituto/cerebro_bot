"""respond — nó terminal. Persiste sessão, regista turno e liberta o lock.

Garantias:
  - UPSERT de `bot_sessions` SEMPRE (qualification + state_extras + last_agent).
  - **Silêncio absoluto** quando bypass marcar:
        state_extras["_smart_silence"] == True
     OR bypass_reason.startswith("hard_opt_out:")
    Nesses casos: NÃO insere turno do assistant, devolve reply=[].
  - **Persiste opt-out** em `bot_opt_outs` se `state_extras["_pending_opt_out"]`
    existir (paridade JS `addOptOut`).
  - INSERT em `bot_turns` com 1 row do assistant (juntando splits com '\\n')
    apenas quando há reply real (não silenciado).
  - `release_lock` corre num `finally` — invariante: o lock nunca fica órfão.
  - **Empatia em transfer/falha**: se reply chegar vazio fora de silêncio,
    preenche com mensagem empática que convida à videochamada com a Talita
    (variante curta para SMS). Nunca devolve string de debug para o lead.
  - **Guard de horas inventadas**: se `agent_used != "scheduling"` e o reply
    contém um horário concreto (regex), substitui por nota neutra e
    sinaliza `_force_scheduling` para o próximo turno ir directo ao scheduling.
"""

from __future__ import annotations

import logging
import re

from langchain_core.runnables import RunnableConfig

from core.database import insert_turn, release_lock, upsert_opt_out, upsert_session
from graph.state import BotState

log = logging.getLogger(__name__)


# Detecta horas concretas tipo "17:30", "17h30", "17h", "às 9", "17 horas".
# Usado para impedir que triage/specialist proponham horários (só scheduling pode).
_TIME_LIKE_RE = re.compile(
    r"\b\d{1,2}\s*(?:[:h]\s*\d{2}|h(?!\w)|\s+horas?\b)",
    re.IGNORECASE,
)


def _should_silence(state: BotState) -> bool:
    """True quando o turno deve sair sem reply.

    3 gatilhos (paridade spec do user):
      - smart silence (👍/ok/sim)
      - hard opt-out (row já existente em bot_opt_outs)
      - pending opt-out (acabou de ser detectado nesta mensagem → bot já cala)
    """
    if state.state_extras.get("_smart_silence") is True:
        return True
    if state.bypass_reason and state.bypass_reason.startswith("hard_opt_out:"):
        return True
    if state.state_extras.get("_pending_opt_out"):
        return True
    return False


def _empathic_fallback(state: BotState) -> list[str]:
    """Mensagem decente quando o grafo termina sem reply útil.

    Adapta por `transfer_reason` e por canal (SMS ≤160 chars).
    """
    reason = (state.transfer_reason or "").lower()
    sms = state.channel == "sms"

    if reason == "emergency":
        if sms:
            return [
                "Lamento. Se for urgencia aguda liga 112. Vou pedir a Talita Alves para te ligar agora numa videochamada."
            ]
        return [
            "Lamento ouvir isso. Se for uma urgência aguda, liga já 112. "
            "Vou pedir à Talita Alves, a nossa Gestora de Pacientes, para te ligar numa videochamada para te ajudar. Aguarda alguns minutos."
        ]

    if reason == "human_request":
        if sms:
            return [
                "Claro. A Talita Alves vai falar contigo numa videochamada. Preferes hoje ou amanha?"
            ]
        return [
            "Claro. A Talita Alves, a nossa Gestora de Pacientes, vai falar contigo numa videochamada gratuita. Preferes hoje ou amanhã?"
        ]

    # default — book/triage transfer ou outro caminho sem reply
    if sms:
        return [
            "Vou pedir a Talita Alves para te ligar numa videochamada e ajudar. Aguarda um momento."
        ]
    return [
        "Vou pedir à Talita Alves, a nossa Gestora de Pacientes, para te ligar diretamente numa videochamada e ajudar. Aguarda um momento."
    ]


def _strip_invented_time(state: BotState) -> None:
    """Se um agente que não é scheduling propôs hora concreta, neutraliza.

    Acontece quando o triage alucina ("hoje às 17:30") em vez de entregar
    ao scheduling. Marca `_force_scheduling=True` para o próximo turno ir
    directo ao scheduling (real slots do DB).
    """
    if state.agent_used == "scheduling":
        return
    if not state.reply:
        return
    if not any(_TIME_LIKE_RE.search(r or "") for r in state.reply):
        return
    log.warning(
        "[respond] guard: agent=%s tentou propor hora concreta — neutralizo e forço scheduling no próximo turno: %r",
        state.agent_used,
        state.reply,
    )
    state.reply = [
        "Deixa-me ver a agenda da Talita e digo-te os horários disponíveis."
    ]
    state.state_extras["_force_scheduling"] = True


async def respond_node(state: BotState, config: RunnableConfig) -> BotState:
    pool = config["configurable"]["pool"]
    silence = _should_silence(state)

    if silence:
        # Garante reply vazio; nenhum stub a inflar.
        state.reply = []
    else:
        # Guard determinístico contra horas inventadas por agentes não-scheduling.
        _strip_invented_time(state)
        if not state.reply:
            # Nenhum agente preencheu reply. Em vez de string de debug,
            # devolvemos mensagem empática + marcamos transfer para alerta interno.
            log.error(
                "[respond] reply vazio fora de silêncio — fallback empático. agent=%s transfer=%s reason=%s",
                state.agent_used,
                state.transfer,
                state.transfer_reason,
            )
            state.reply = _empathic_fallback(state)
            state.transfer = True
            if not state.transfer_reason:
                state.transfer_reason = "empty_reply_fallback"

    try:
        # 1) Persistir sessão SEMPRE (qualification + flags internas + último agente)
        await upsert_session(
            pool,
            state.phone_id,
            state.contact_phone,
            qualification_state=state.qualification_state.model_dump(exclude_none=True),
            state=state.state_extras or {},
            current_namespace=state.current_namespace,
            last_agent=state.agent_used,
        )

        # 2) Persistir opt-out global se bypass detectou (paridade JS addOptOut)
        pending = state.state_extras.get("_pending_opt_out") if state.state_extras else None
        if pending and isinstance(pending, dict) and pending.get("type"):
            opt_out_type = pending["type"]
            # Só persistimos os 3 tipos válidos da tabela bot_opt_outs.
            if opt_out_type in {"wrong_number", "no_interest", "already_patient"}:
                # `source` na tabela tem CHECK IN ('bot','manual','cron') —
                # qualquer detecção automática (regex|llm) é "bot". O detalhe
                # da origem fica no matched_text e no _pending_opt_out.source.
                trigger_source = pending.get("source", "bot")
                try:
                    await upsert_opt_out(
                        pool,
                        state.contact_phone,
                        opt_out_type,
                        reason=f"{state.bypass_reason or opt_out_type} (via {trigger_source})",
                        matched_text=pending.get("matched_text"),
                        source="bot",
                    )
                    log.info(
                        "[respond] opt-out persistido contact=%s type=%s trigger=%s",
                        state.contact_phone, opt_out_type, trigger_source,
                    )
                except Exception as e:
                    # Não-fatal: log e segue (lock + session já tratados).
                    log.warning("[respond] upsert_opt_out falhou: %s", e)

        # 3) Inserir turno do assistant SOMENTE se houver reply real
        if not silence:
            full_reply = "\n".join(state.reply).strip()
            if full_reply:
                await insert_turn(
                    pool,
                    state.phone_id,
                    state.contact_phone,
                    role="assistant",
                    content=full_reply,
                    agent_used=state.agent_used,
                    tools_called=None,
                    latency_ms=None,
                )
    finally:
        # 4) Libertar lock SEMPRE — invariante para evitar lock pendurado.
        await release_lock(pool, state.phone_id, state.contact_phone)

    log.info(
        "[respond] persisted agent=%s reply_msgs=%d transfer=%s silence=%s",
        state.agent_used, len(state.reply), state.transfer, silence,
    )
    return state
