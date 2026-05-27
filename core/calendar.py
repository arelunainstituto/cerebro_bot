"""Google Calendar — integração real via Service Account.

Espelha [api/bot/_lib/calendar.js](../../api/bot/_lib/calendar.js) em Python.

A Service Account `bot-areluna-calendar@expanded-aria-496116-v2.iam.gserviceaccount.com`
foi partilhada com permissões de edição no calendário da clínica. Credenciais
suportadas em duas formas (a primeira que estiver disponível ganha):

  1. **Ficheiro JSON**: `GOOGLE_CREDS_FILE=core/google_creds.json` (caminho
     relativo ao cwd ou absoluto). Recomendado para dev local.
  2. **Env inline**: `GOOGLE_SERVICE_ACCOUNT_JSON='{"type":"service_account",...}'`.
     Recomendado para Vercel/cloud (sem ficheiros em disco).

Arquitetura crítica
-------------------
A biblioteca `googleapiclient` é **síncrona** (bloqueia o event loop). Todas
as chamadas que tocam a rede correm dentro de `loop.run_in_executor(None, ...)`
para não bloquear o FastAPI/LangGraph.

O `service` é construído UMA vez por processo (lazy + cache thread-safe via
asyncio.Lock) — `events().insert()` reusa a mesma sessão HTTPS.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

log = logging.getLogger(__name__)

_SCOPES = ["https://www.googleapis.com/auth/calendar"]
_DEFAULT_TZ = "Europe/Lisbon"
_DEFAULT_DURATION_MIN = 30

# Cache module-level + lock para não criar 2 services em paralelo
_service = None
_service_lock = asyncio.Lock()


def _load_credentials() -> service_account.Credentials:
    """Carrega creds da SA. Ficheiro tem prioridade sobre env inline."""
    file_path = os.getenv("GOOGLE_CREDS_FILE")
    if file_path:
        p = Path(file_path)
        if not p.is_absolute():
            # Resolve relativo ao cwd actual (uvicorn arranca a partir de cerebro_python/)
            p = Path.cwd() / p
        if p.exists():
            return service_account.Credentials.from_service_account_file(
                str(p), scopes=_SCOPES,
            )
        log.warning("[gcal] GOOGLE_CREDS_FILE=%s não existe — a tentar inline", p)

    inline = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
    if inline:
        data = json.loads(inline)
        return service_account.Credentials.from_service_account_info(
            data, scopes=_SCOPES,
        )

    raise RuntimeError(
        "[gcal] credenciais não encontradas — define GOOGLE_CREDS_FILE "
        "(caminho para JSON) OU GOOGLE_SERVICE_ACCOUNT_JSON (JSON inline)."
    )


async def _get_service():
    """Lazy + cached. Constrói o cliente da Calendar API uma só vez."""
    global _service
    if _service is not None:
        return _service
    async with _service_lock:
        if _service is not None:
            return _service
        loop = asyncio.get_running_loop()
        creds = await loop.run_in_executor(None, _load_credentials)
        # `cache_discovery=False` evita warning sobre `oauth2client` e ficheiros
        # cache em /tmp; insignificante em latência (1 fetch JSON na 1ª chamada).
        _service = await loop.run_in_executor(
            None, lambda: build("calendar", "v3", credentials=creds, cache_discovery=False),
        )
        log.info("[gcal] Calendar service inicializado (SA=%s)", creds.service_account_email)
        return _service


def _parse_to_aware(iso: str) -> datetime:
    """ISO 8601 → `datetime` aware. Se vier naive, assume Europe/Lisbon."""
    dt = datetime.fromisoformat(iso)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ZoneInfo(_DEFAULT_TZ))
    return dt


def _insert_sync(
    service,
    *,
    calendar_id: str,
    start_iso: str,
    end_iso: str,
    timezone: str,
    contact_name: str,
    treatment: str,
) -> dict[str, Any]:
    """Chamada SÍNCRONA — corre dentro de `run_in_executor`."""
    body = {
        "summary": f"Avaliação online — {contact_name} ({treatment})",
        "description": (
            "Videochamada WhatsApp com a Talita Alves, Gestora de Pacientes Especialista. "
            "Duração estimada: 30 minutos. Marcação criada automaticamente pela Rosa Cordeiro."
        ),
        "start": {"dateTime": start_iso, "timeZone": timezone},
        "end": {"dateTime": end_iso, "timeZone": timezone},
        "reminders": {"useDefault": True},
    }
    return service.events().insert(calendarId=calendar_id, body=body).execute()


async def create_gcal_event(
    *,
    slot_start_iso: str,
    slot_end_iso: str | None = None,
    contact_name: str,
    treatment: str,
) -> str:
    """Cria o evento no Google Calendar e devolve o `event_id` real.

    Mantém a assinatura kwargs-only para preservar paridade com o nó
    [scheduling.py](../graph/nodes/scheduling.py) que já a invoca assim.
    Se `slot_end_iso` não vier, é calculado como `slot_start + 30 min`.

    O timezone enviado à API é `GOOGLE_CALENDAR_TZ` (default `Europe/Lisbon`)
    para garantir display correcto na agenda da Talita. Os `dateTime` podem
    vir em qualquer offset — a Google converte para o timezone do evento.

    Lança `RuntimeError` em falha de auth/credenciais (fatal — não retry).
    Propaga `HttpError` em falhas transientes da API (scheduler já tem
    `try/except` que converte em `transfer=True` se necessário).
    """
    calendar_id = os.environ["GOOGLE_CALENDAR_ID"]
    timezone = os.getenv("GOOGLE_CALENDAR_TZ", _DEFAULT_TZ)
    duration_min = int(os.getenv("GCAL_EVENT_DURATION_MIN", _DEFAULT_DURATION_MIN))

    start_dt = _parse_to_aware(slot_start_iso)
    if slot_end_iso:
        end_dt = _parse_to_aware(slot_end_iso)
    else:
        end_dt = start_dt + timedelta(minutes=duration_min)

    start_iso = start_dt.isoformat()
    end_iso = end_dt.isoformat()

    service = await _get_service()
    loop = asyncio.get_running_loop()
    try:
        event = await loop.run_in_executor(
            None,
            lambda: _insert_sync(
                service,
                calendar_id=calendar_id,
                start_iso=start_iso,
                end_iso=end_iso,
                timezone=timezone,
                contact_name=contact_name,
                treatment=treatment,
            ),
        )
    except HttpError as e:
        log.error("[gcal] HttpError ao criar evento: status=%s detail=%s", e.resp.status, e)
        raise

    event_id = event.get("id")
    if not event_id:
        raise RuntimeError(f"[gcal] resposta sem id: {event!r}")

    log.info(
        "[gcal] event_id=%s start=%s end=%s contact=%r treatment=%r",
        event_id, start_iso, end_iso, contact_name, treatment,
    )
    return event_id
