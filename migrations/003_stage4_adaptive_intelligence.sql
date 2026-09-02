create table if not exists adaptive_observations (
    id uuid primary key,
    symbol text not null,
    model_version text not null,
    action text not null,
    confidence double precision not null check (confidence >= 0 and confidence <= 1),
    realized_return double precision not null,
    observed_at timestamptz not null,
    regime text not null default 'unknown',
    horizon integer not null default 6,
    expected_probability double precision,
    features jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now()
);
create index if not exists adaptive_observations_model_time_idx on adaptive_observations(model_version, observed_at desc);
create index if not exists adaptive_observations_regime_time_idx on adaptive_observations(regime, observed_at desc);

create table if not exists adaptive_candidates (
    id uuid primary key,
    champion_version text not null,
    challenger_version text not null,
    status text not null check (status in ('quarantined','promotion_eligible','rejected')),
    reason text not null,
    evidence jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    evaluated_at timestamptz
);
create index if not exists adaptive_candidates_status_idx on adaptive_candidates(status, created_at desc);

alter table adaptive_observations enable row level security;
alter table adaptive_candidates enable row level security;

revoke all on adaptive_observations from anon, authenticated;
revoke all on adaptive_candidates from anon, authenticated;
