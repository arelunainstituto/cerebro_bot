# cérebro_python

Serviço HTTP isolado que expõe a lógica conversacional do bot (LangGraph + FastAPI),
para ser consumido por um Gateway externo (hoje o Vercel/Node.js — `/api/bot/*` —
chama este cérebro via HTTP em vez de executar o grafo localmente).

## Regra de Ouro

Nada fora desta pasta pode ser modificado. O código JS em produção
(`/api/bot/*`, `/api/webhook.js`, etc.) continua a funcionar tal como está.
Este serviço é puramente aditivo.

## Quickstart

```bash
cd cerebro_python
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # depois edita com credenciais reais
uvicorn main:app --reload --port 8001
```

Testes rápidos:

```bash
curl http://localhost:8001/healthz
curl -X POST http://localhost:8001/v1/chat/invoke \
  -H 'Content-Type: application/json' \
  -d '{"phone_id":"763656903507884","contact_phone":"351900000000","message":"olá"}'
```

## Estrutura

```
cerebro_python/
├── main.py            # FastAPI app + endpoint /v1/chat/invoke
├── graph/
│   ├── state.py       # BotState (Pydantic) — estado do grafo
│   ├── graph.py       # build_graph() — monta StateGraph
│   ├── edges.py       # transições condicionais entre nós
│   └── nodes/         # 8 nós: enter, bypass, router, transfer_guard,
│                      #         triage, specialist, scheduling, respond
└── schemas/           # Pydantic models (mirror dos Zod schemas JS)
```

## Status actual

Esqueleto. Nenhum nó tem lógica real ainda — todos retornam o estado intacto.
A integração com Supabase, LLMs e ferramentas vem em fases seguintes.
