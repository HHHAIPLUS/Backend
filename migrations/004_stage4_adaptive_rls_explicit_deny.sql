create policy adaptive_observations_deny_anon on public.adaptive_observations as restrictive for all to anon using (false) with check (false);
create policy adaptive_observations_deny_authenticated on public.adaptive_observations as restrictive for all to authenticated using (false) with check (false);
create policy adaptive_candidates_deny_anon on public.adaptive_candidates as restrictive for all to anon using (false) with check (false);
create policy adaptive_candidates_deny_authenticated on public.adaptive_candidates as restrictive for all to authenticated using (false) with check (false);
