from __future__ import annotations
from datetime import datetime, timezone
from typing import Any
from app.persistence.supabase import store

async def record_decision(symbol: str, payload: dict[str, Any]):
    return await store.insert('decision_records', {'id': payload['record_id'], 'symbol': symbol, 'payload': payload, 'created_at': datetime.now(timezone.utc).isoformat()})

async def record_outcome(decision_id: str, payload: dict[str, Any]):
    return await store.insert('decision_outcomes', {'decision_id': decision_id, 'payload': payload, 'created_at': datetime.now(timezone.utc).isoformat()})

async def record_event(event_type: str, payload: dict[str, Any]):
    return await store.insert('system_events', {'event_type': event_type, 'payload': payload, 'created_at': datetime.now(timezone.utc).isoformat()})


async def upsert_position_state(exchange: str, symbol: str, side: str, payload: dict[str, Any]):
    return await store.upsert('position_states', {
        'exchange': exchange, 'symbol': symbol, 'side': side, 'payload': payload,
        'updated_at': datetime.now(timezone.utc).isoformat(),
    }, 'exchange,symbol,side')

async def load_position_states():
    return await store.select('position_states', {'select':'*'})
