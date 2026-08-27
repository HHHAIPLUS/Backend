from __future__ import annotations
import httpx
from typing import Any
from app.core.config import settings

class SupabaseStore:
    def __init__(self):
        self.url = settings.supabase_url.rstrip('/') if settings.supabase_url else ''
        self.key = settings.supabase_service_role_key

    @property
    def configured(self) -> bool:
        return bool(self.url and self.key)

    def _headers(self):
        return {'apikey': self.key, 'Authorization': f'Bearer {self.key}', 'Content-Type': 'application/json', 'Prefer': 'return=representation'}

    async def insert(self, table: str, row: dict[str, Any]):
        if not self.configured: raise RuntimeError('Supabase is not configured')
        async with httpx.AsyncClient(timeout=8) as client:
            r = await client.post(f'{self.url}/rest/v1/{table}', headers=self._headers(), json=row)
            r.raise_for_status(); return r.json()

    async def upsert(self, table: str, row: dict[str, Any], on_conflict: str):
        if not self.configured: raise RuntimeError('Supabase is not configured')
        headers = self._headers(); headers['Prefer'] = f'resolution=merge-duplicates,return=representation'
        async with httpx.AsyncClient(timeout=8) as client:
            r = await client.post(f'{self.url}/rest/v1/{table}?on_conflict={on_conflict}', headers=headers, json=row)
            r.raise_for_status(); return r.json()

    async def select(self, table: str, params: dict[str, str] | None = None):
        if not self.configured: raise RuntimeError('Supabase is not configured')
        async with httpx.AsyncClient(timeout=8) as client:
            r = await client.get(f'{self.url}/rest/v1/{table}', headers=self._headers(), params=params or {'select':'*'})
            r.raise_for_status(); return r.json()

    async def latest(self, table: str, params: dict[str, str] | None = None):
        rows = await self.select(table, params or {'select':'*','order':'created_at.desc','limit':'1'})
        return rows[0] if rows else None

store = SupabaseStore()
