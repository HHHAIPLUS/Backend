create table if not exists public.model_artifacts (
  name text not null,
  version text primary key,
  artifact jsonb not null,
  metrics jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);
grant select, insert, update, delete on public.model_artifacts to service_role;


create table if not exists public.position_states (
  exchange text not null,
  symbol text not null,
  side text not null,
  payload jsonb not null default '{}'::jsonb,
  updated_at timestamptz not null default now(),
  primary key(exchange, symbol, side)
);
grant select, insert, update, delete on public.position_states to service_role;
