from ai.capital_guard import KillSwitch

# One authoritative process-local kill switch for all backend control paths.
# Persistence is intentionally a separate concern and remains required for production.
kill_switch = KillSwitch()
