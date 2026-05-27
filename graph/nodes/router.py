"""router — decide o agente terminal (triage | specialist | scheduling).

Espelha `api/bot/agents/router.js` mas simplificado para 3 agentes (sem
`transfer_to_human` — em Python isso vem como `state.transfer=True`
propagado por bypass/transfer_guard).

Regras principais embebidas no system prompt:
  - 🔴 Preço/valor/orçamento → SEMPRE "triage".
  - Dúvida técnica sobre tratamento (não preço) → "specialist".
  - Marcar / pedir horário / aceitar slot → "scheduling".
  - Default → "triage".
  - Pivô de área: `area_interesse` actual é sinal forte, menção colateral
    NÃO sobrescreve sem mudança drástica.

Fallback em qualquer falha → "triage" (lado seguro: continua qualificação).
TODOs (próximas fases):
  - Fast-paths determinísticos (slots pendentes, dataIntent).
  - Pivô de área aplicado ao qualification_state quando há mudança drástica.
"""

from __future__ import annotations

import logging
import os

from core.llm import invoke_structured
from graph.state import BotState
from schemas import RouterDecision

log = logging.getLogger(__name__)


_ROUTER_SYSTEM = """És um router silencioso do bot do Instituto Areluna (clínica dentária e estética avançada, no Porto). A tua única tarefa é escolher qual dos 3 agentes responde ao lead. Não falas com o lead.

AGENTES DISPONÍVEIS:
- "triage": qualificação inicial (POP), recolher dados em falta (nome, área de interesse, motivação), responder de forma geral à pré-consulta.
- "specialist": pergunta clínica/técnica específica sobre tratamento (processo, duração, técnica, indicações). Usa RAG.
- "scheduling": lead quer MARCAR a avaliação, pede horários, ou escolhe/confirma um slot.

REGRAS DE DECISÃO (ordem de precedência):

🔴 REGRA DO PREÇO (precedência máxima):
- Pergunta sobre PREÇO / VALOR / ORÇAMENTO / "QUANTO CUSTA" / "PARTE FINANCEIRA" / PAGAMENTO → SEMPRE "triage".
- O bot NUNCA dá preço. Quem mostra valores é a Talita NA VIDEOCHAMADA gratuita.
- Mesmo que o lead também mencione agendamento na mesma frase → primeiro qualificação, depois Talita.

REGRA DE MARCAÇÃO:
- Lead diz explicitamente "quero marcar", "vamos agendar", "tem horário", "que dias têm" → "scheduling".
- Lead aceita um slot ("ok pode ser", "fechado", "confirmo a quarta") → "scheduling".
- Lead pede hora/data concreta ("quarta às 14h", "amanhã de tarde") → "scheduling".

REGRA DE DÚVIDA TÉCNICA:
- Pergunta sobre PROCESSO / DURAÇÃO / TÉCNICA / INDICAÇÕES / PASSOS de um tratamento específico (sem ser preço) → "specialist".
- Ex: "como funciona o implante?", "quanto tempo demora a faceta?", "que técnica usam para o transplante?".

REGRA DE DEFAULT:
- Primeira mensagem sem histórico → "triage".
- Qualquer outra coisa (saudação, pergunta genérica, queixa, partilha de informação) → "triage".

PIVÔ DE ÁREA:
- A `area_interesse` actual (do histórico ou do template de origem) é um sinal forte.
- Menção colateral a outra área ("ah, também tenho aparelho") NÃO chega para mudar agente nem área. Continua na área principal.
- Só pivota se o lead disser claramente "deixa lá os implantes, agora quero ortodontia".

OUTPUT (JSON estrito):
{
  "agent": "triage" | "specialist" | "scheduling",
  "confidence": 0.0-1.0,
  "reason": "explicação curta em PT-PT"
}"""


async def router_node(state: BotState) -> BotState:
    qual = state.qualification_state.model_dump(exclude_none=True)
    mem_tail = state.short_memory[-4:] if state.short_memory else []

    user_parts = [
        f'Mensagem do lead: "{state.incoming_message}"',
        f"Estado de qualificação: {qual or '(vazio)'}",
        f"Área de interesse actual: {qual.get('area_interesse') or '(nenhuma)'}",
    ]
    if mem_tail:
        hist = "\n".join(f"{'Lead' if t.role == 'user' else 'Bot'}: {t.content}" for t in mem_tail)
        user_parts.append(f"Conversa recente:\n{hist}")

    user_prompt = "\n\n".join(user_parts)

    try:
        decision: RouterDecision = await invoke_structured(
            RouterDecision,
            [
                {"role": "system", "content": _ROUTER_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            model=os.getenv("BOT_ROUTER_MODEL", "gpt-4o-mini"),
            temperature=0.1,
            max_tokens=300,
            schema_name="router_decision",
        )
        state.next_agent = decision.agent
        log.info(
            "[router] decision agent=%s confidence=%.2f reason=%r",
            decision.agent, decision.confidence, decision.reason[:80],
        )
    except Exception as e:
        # Fallback seguro: triage continua qualificação sem fechar nada.
        log.warning("[router] LLM falhou (%s) — fallback agent=triage", e)
        state.next_agent = "triage"

    return state
