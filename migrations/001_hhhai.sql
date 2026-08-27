create extension if not exists pgcrypto;
create table if not exists public.decision_records (
  id uuid primary key default gen_random_uuid(),
  symbol text not null,
  payload jsonb not null,
  created_at timestamptz not null default now()
);
create table if not exists public.decision_outcomes (
  id uuid primary key default gen_random_uuid(),
  decision_id uuid not null references public.decision_records(id) on delete cascade,
  payload jsonb not null,
  created_at timestamptz not null default now()
);
create table if not exists public.system_events (
  id uuid primary key default gen_random_uuid(),
  event_type text not null,
  payload jsonb not null,
  created_at timestamptz not null default now()
);
create table if not exists public.model_registry (
  name text not null,
  version text not null,
  status text not null,
  metrics jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  primary key(name, version)
);
create table if not exists public.learning_examples (
  id uuid primary key default gen_random_uuid(),
  symbol text not null,
  observed_at timestamptz not null,
  features jsonb not null,
  label integer not null check(label in (-1,0,1)),
  outcome_return double precision,
  created_at timestamptz not null default now()
);
grant select, insert, update, delete on public.decision_records, public.decision_outcomes, public.system_events, public.model_registry, public.learning_examples to service_role;
