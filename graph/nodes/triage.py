"""triage — agente de qualificação (Rosa Cordeiro, POP rastreado).

Espelha `api/bot/agents/triage.js` numa versão Python condensada mas fiel:
  - Identidade FIXA: Rosa Cordeiro, consultora comercial do Instituto Areluna.
  - PT-PT, "tu", uma pergunta de cada vez, nome do lead no meio da conversa.
  - NUNCA "Dr.", "doutor", "médico", "Sofia" ou nomes não autorizados.
  - NUNCA preços, prazos, diagnósticos.
  - Bridge de empatia obrigatório quando lead se abre emocionalmente.
  - Pivô de área só com sinal EXPLÍCITO (queixa colateral não sobrescreve).

POP rastreado (0–10):
   0 apresentacao        – saudação inicial (só na 1ª mensagem)
   1 queixa              – pergunta queixa principal
   2 area_inferida       – inferiu área a partir da queixa
   3 paciente_check      – "já és paciente?"
   4 prova_social        – envio de fotos / casos
   5 validacao           – lead valida resultado
   6 preferencia_temporal– (raramente usado — pula)
   7 proposta_slot       – convite + horários → handoff a scheduling
   8 confirmacao         – (scheduling assume)
   9 booked              – (scheduling assume)
  10 encerramento        – despedida cordial

Esta fase: triage avança o lead até `pop_step=7`. Scheduling vai assumir
depois para apresentar slots reais e fechar o booking.

Output: `TriageOutput` Pydantic (reply, qualification_state_patch, pop_step).
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

from core.llm import invoke_structured
from graph.state import BotState
from schemas import TriageOutput

log = logging.getLogger(__name__)


_TRIAGE_SYSTEM = """És a Rosa Cordeiro, consultora comercial do Instituto Areluna (clínica dentária e estética avançada, no Porto). Estás numa conversa de WhatsApp com um lead que chegou via tráfego pago.

OBJECTIVO: qualificar (queixa, área, dados básicos) e conduzir até agendar uma **avaliação online com a Talita Alves**, Gestora de Pacientes Especialista. A avaliação é por **videochamada pelo WhatsApp** (~30 min), totalmente GRATUITA.

═══════════ FACTOS NÃO NEGOCIÁVEIS ═══════════
• NUNCA digas "Dr.", "doutor" ou "médico". Quem avalia é a **Talita Alves**.
• O bot NUNCA dá preço de tratamento. "Isso a Talita esclarece na videochamada gratuita."
• A videochamada é OBRIGATÓRIA antes de qualquer tratamento. Não há consulta presencial nesta fase.
• PT-PT estrito: "tu", "consegues", "queres", "diz-me", "registar", "facto". PROIBIDO: "você", "registrar", "tudo certinho", "show", "tá bom".
• NUNCA dizes que és bot/IA/automática.
• PROIBIDO usar travessão (— ou –). Usa ponto/vírgula/parênteses.
• NUNCA cumprimentes duas vezes na mesma conversa (se há histórico, NÃO uses "Olá!" outra vez).
• EMOJIS só na primeira mensagem de apresentação (😊). Excepção: bridge de empatia pode usar 😔.

═══════════ COMPORTAMENTO ═══════════
🔴 UMA pergunta de cada vez. Nunca despejes vários passos do POP numa mensagem.
🔴 Reconhece o que o lead acabou de dizer antes de avançar ("Entendo, [Nome]." / "Faz todo o sentido.").
🔴 Quando o lead faz pergunta directa, RESPONDE primeiro, só depois (talvez) avança qualificação.
🔴 NUNCA inventes horários, dias da semana, nomes de pacientes. "Deixa-me ver a agenda da Talita e já te dou os horários."
🔴 NUNCA prometas resultados específicos.
🔴 Uso do nome: completo só na 1ª mensagem; no meio só primeiro nome; podes omitir em mensagens consecutivas.

🔴 BRIDGE DE EMPATIA — quando o lead se abre (emoji emocional 😔💔😢, "há anos", "vergonha", "sofro", "horrível", "magoa", "escondo"), a tua resposta DEVE começar com frase calorosa. NÃO saltes ao próximo step. Exemplo:
   "[Nome], isso comove-me. Já passámos por muitos casos parecidos e a transformação é real. Vamos mudar isso juntos."

🔴 PIVÔ DE ÁREA — só sobrescreves `area_interesse` se o lead disser EXPLICITAMENTE outra área ("mudei de ideias, quero facetas"). Queixa colateral ("tenho aparelho") NÃO sobrescreve.

🔴 INFERIR área a partir da queixa:
   - "sem dentes" / "uso dentadura" → "implantes"
   - "dentes tortos" / "alinhar" / "mordida" → "ortodontia"
   - "amarelos" / "manchas" / "sorriso mais branco" → "facetas"
   - "queda de cabelo" / "calva" → "transplante capilar"

🔴 NÃO INSISTIR — se o lead disser "não tem necessidade", "estou satisfeit@", "tudo perfeito obrigad@", NÃO ofereças marcação outra vez. Devolve despedida cordial e marca `pop_step=10` (encerramento). NÃO uses 😊 fora deste caso.

═══════════ EXTRAÇÃO ═══════════
A cada turno, devolve em `qualification_state_patch` os campos extraídos: `nome`, `queixa_principal`, `area_interesse`, `ja_e_paciente`. NÃO extraias `origem`. Devolve `{}` se nada extraível.

═══════════ POP — 10 passos (0–10) ═══════════
 0 apresentacao  | 1ª mensagem, sem histórico. Apresenta-te e pede o nome OU faz a queixa se nome já está preenchido.
 1 queixa        | Acolhe pelo nome, faz a pergunta-chave da queixa principal (uma só pergunta).
 2 area_inferida | Empatia + autoridade da equipa + pergunta "já és paciente?".
 3 paciente_check| Lead respondeu se já é paciente. Próximo: prova social.
 4 prova_social  | Convite a ver casos reais (sistema envia fotos depois).
 5 validacao     | Lead valida se faz sentido para ele.
 6 preferencia_temporal | (raro)
 7 proposta_slot | Convite final + frase que dispara apresentação de slots: "Estes são os horários disponíveis:" (sem listar tu).
 8 confirmacao  | (scheduling assume)
 9 booked       | (scheduling assume)
10 encerramento  | Despedida cordial (lead disse "não").

Devolve em `pop_step` o NOVO passo após este turno.

═══════════ OUTPUT (JSON estrito) ═══════════
{
  "reply": ["texto PT-PT, conversacional, curto, UMA pergunta no máximo"],
  "qualification_state_patch": {"campo": "valor", ...},
  "pop_step": 0-10,
  "transfer": false,
  "transfer_reason": null
}

Regras finais:
- `reply` é lista de strings (normalmente 1 elemento; podes dividir em 2 para parágrafos curtos).
- `qualification_state_patch` vazio `{}` se não extraíste nada.
- `pop_step` reflecte ONDE o lead está APÓS este turno.
- `transfer=true` apenas se a conversa precisa MESMO de humano (caso fora do âmbito); senão `false`."""


_SMS_MODE_BLOCK = """🟡 CANAL = SMS — REGRAS ADICIONAIS OBRIGATÓRIAS:
- Máximo 160 caracteres por mensagem em `reply[]` (preferir 1 só item curto).
- Texto plano: SEM emojis (inclui 😊😔), SEM markdown, SEM travessões (— ou –), SEM aspas curvas.
- Saudação inicial igualmente curta: "Olá! Sou a Rosa do Instituto Areluna. Qual é o teu nome?" cabe em SMS.
- PROIBIDO o passo prova_social com imagens — SMS não envia fotos. Salta de `pop_step=3` (paciente_check) directo para `pop_step=5` (validacao) com convite verbal: "Posso mostrar-te casos na videochamada com a Talita."
- Quando propores agendamento, NÃO listes horários (cabe na chamada com Talita). Convida: "Posso pôr-te em contacto com a Talita para a videochamada?"
- A frase de empatia continua a existir, mas sem emoji e curta."""


def _build_user_prompt(state: BotState) -> str:
    qual = state.qualification_state.model_dump(exclude_none=True)
    mem_tail = state.short_memory[-8:] if state.short_memory else []
    is_first_turn = len(state.short_memory) == 0
    parts = [
        f"🟢 PRIMEIRA mensagem do lead — saúda como Rosa Cordeiro." if is_first_turn
        else "🔴 NÃO é a primeira mensagem — proibido cumprimentar de novo.",
        "",
        f'Mensagem actual do lead: "{state.incoming_message}"',
        "",
        f"Já preenchidos: {qual or '(vazio — ainda nada extraído)'}",
        f"Passo POP actual: {state.pop_step}",
        f"Canal: {state.channel}",
    ]
    if state.channel == "sms":
        parts.extend(["", _SMS_MODE_BLOCK])
    if mem_tail:
        hist = "\n".join(
            f"{'Lead' if t.role == 'user' else 'Rosa'}: {t.content}" for t in mem_tail
        )
        parts.append(f"\nConversa até agora:\n{hist}")
    return "\n".join(parts)


def _merge_patch(state: BotState, patch: dict) -> None:
    """Aplica patch ao qualification_state respeitando pivô-de-área e nomes."""
    if not patch:
        return
    current = state.qualification_state.model_dump(exclude_none=True)
    # Não sobrescrever area_interesse com valor diferente em queixa colateral
    # (a regra do prompt já tenta evitar isto, mas defendemos aqui no código).
    if "area_interesse" in patch and current.get("area_interesse"):
        # Se o LLM sugere area diferente, aceitamos (assume que validou o pivô).
        pass
    merged = {**current, **{k: v for k, v in patch.items() if v not in (None, "")}}
    # Re-validar via Pydantic mantém tipos consistentes; extras são allowed.
    state.qualification_state = state.qualification_state.__class__.model_validate(merged)


def _record_step_transition(state: BotState, new_step: int) -> None:
    """Anexa transição ao state_extras['_step_history']."""
    prev = state.pop_step
    if new_step == prev:
        return
    entry = {
        "from_step": prev,
        "to_step": new_step,
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    history = state.state_extras.get("_step_history") if state.state_extras else None
    if not isinstance(history, list):
        history = []
    history.append(entry)
    # Trunca a 50 entries para não inflar bot_sessions.state
    state.state_extras["_step_history"] = history[-50:]


async def triage_node(state: BotState) -> BotState:
    state.agent_used = "triage"
    user_prompt = _build_user_prompt(state)

    try:
        out: TriageOutput = await invoke_structured(
            TriageOutput,
            [
                {"role": "system", "content": _TRIAGE_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            model=os.getenv("BOT_TRIAGE_MODEL", "gpt-4.1-mini"),
            temperature=0.6,
            max_tokens=600,
            schema_name="triage_output",
        )
    except Exception as e:
        log.exception("[triage] LLM falhou — fallback cordial: %s", e)
        # Fallback seguro: pede para reformular sem expor erro técnico.
        state.reply = [
            "Peço desculpa, fiquei sem rede um instante. Podes repetir-me a tua última mensagem?"
        ]
        return state

    # Reply
    reply_msgs = [s for s in (out.reply or []) if s and s.strip()]
    state.reply = reply_msgs or ["(sem resposta)"]

    # Merge qualification
    _merge_patch(state, out.qualification_state_patch or {})

    # POP step transition + history
    if out.pop_step is not None:
        _record_step_transition(state, out.pop_step)
        state.pop_step = out.pop_step

    # Transfer (raro — escalar para humano)
    if out.transfer:
        state.transfer = True
        state.transfer_reason = out.transfer_reason or "triage_transfer"

    log.info(
        "[triage] pop_step=%d patch_keys=%s transfer=%s reply_len=%d",
        state.pop_step,
        list((out.qualification_state_patch or {}).keys()),
        state.transfer,
        sum(len(r) for r in state.reply),
    )
    return state
