# HHHAI Master Engineering Roadmap

**Authoritative roadmap for HHHAI.**

This file is the project control record. It is not a source-code stage and it must not be replaced by creating files named after stages. Before beginning work, check the current stage and unchecked items here. When an item is genuinely implemented **and verified**, change only that item from `[ ]` to `[✓]`. Do not mark work complete merely because code exists.

## Current position

**STEP 0 — FORENSIC AUDIT + BASELINE: [✓] COMPLETED**

**STAGE 1 — STABILIZE THE FOUNDATION: [✓] COMPLETED**

**STAGE 2 — BUILD THE REAL MARKET DATA INTELLIGENCE LAYER: [✓] COMPLETED**

**STAGE 3 — BUILD THE REAL PREDICTIVE BRAIN: [✓] COMPLETED**

**STAGE 4 — ADAPTIVE INTELLIGENCE: [✓] COMPLETED**

**STAGE 5 — ADVANCED DECISION ENGINE: [✓] COMPLETED**

**STAGE 6 — AUTONOMOUS POSITION INTELLIGENCE: [✓] COMPLETED**

**STAGE 7 — SELF-LEARNING RESEARCH LOOP: [✓] COMPLETED**

**STAGE 8 — RISK & CAPITAL INTELLIGENCE: [✓] COMPLETED**

**CURRENT: STAGE 9 — ADVANCED FRONTEND**

No live-money execution is authorized by this roadmap. Safety gates remain fail-closed throughout every stage.

---

# STEP 0 — Forensic Audit + Baseline [✓]

Audit scope completed against the connected repositories and Supabase project.

---

# STAGE 1 — Stabilize the Foundation [✓]

Goal: make HHHAI reliable, truthful, observable and safe before making the intelligence more sophisticated.

- [✓] Reproduce and eliminate the deployed model-bootstrap failure
- [✓] Verify bootstrap/API request path end-to-end
- [✓] Verify validation metric definitions and trade-count definitions
- [✓] Verify label construction and future-outcome calculations
- [✓] Verify chronological splitting and eliminate leakage paths
- [✓] Verify training/live feature parity
- [✓] Verify missing-data handling and fail-closed behavior
- [✓] Verify model artifact persistence, loading and version integrity
- [✓] Verify model promotion/rejection gates
- [✓] Run and repair the complete backend test suite
- [✓] Expand backend integration tests around real service boundaries
- [✓] Audit and harden Binance connectivity without live orders
- [✓] Audit and harden Bitget connectivity without live orders
- [✓] Verify order/position reconciliation and duplicate-order protection
- [✓] Verify background monitor lifecycle, restart behavior and failure recovery
- [✓] Verify API error handling, timeouts, retries and stale-data behavior
- [✓] Verify configuration/environment separation between paper, testnet and live
- [✓] Verify CORS/authentication/admin-control boundaries
- [✓] Verify Supabase persistence and restart recovery
- [✓] Resolve Supabase RLS/policy findings appropriately
- [✓] Verify frontend production build
- [✓] Verify every frontend endpoint against the backend contract
- [✓] Verify frontend failure/loading/stale-data states
- [✓] Verify deployment/startup health on Render and Vercel
- [✓] Establish operational logs, health checks and safe failure states
- [✓] Confirm real-money execution remains impossible unless every explicit safety gate permits it

**STAGE 1 STATUS: [✓] COMPLETED**

### Stage 1 verification evidence
- Backend CI: **PASS — 108 tests passed, 1 non-failing deprecation warning**
- Exchange boundaries, persistence/RLS, frontend/backend contracts and deployment health: **PASS**
- Live execution safety gate: **fail-closed**

---

# STAGE 2 — Build the Real Market Data Intelligence Layer [✓]

Goal: give HHHAI a truthful, time-aligned representation of the market it is actually trading.

- [✓] Canonical point-in-time market-state schema
- [✓] Multi-timeframe OHLCV, price structure, volatility and volume features
- [✓] Order-book depth/imbalance, trade flow, spread and liquidity stress
- [✓] Funding, open interest and liquidation context
- [✓] Cross-asset, BTC/ETH and market-wide risk context
- [✓] Market-regime state representation
- [✓] Timestamped news/event ingestion and source credibility
- [✓] Historical-data enrichment and caching
- [✓] Point-in-time joins and training/live schema parity
- [✓] Continuous data-quality coverage and fail-closed degradation
- [✓] Multi-regime validation

**STAGE 2 STATUS: [✓] COMPLETED**

### Stage 2 verification evidence
- Final backend verification: **118 tests passed, 1 non-failing deprecation warning**
- Binance/Bitget market data, historical enrichment, canonical feature projection and data-quality gates: **PASS**
- Render deployment: **LIVE**
- Live-money execution remained disabled

---

# STAGE 3 — Build the Real Predictive Brain [✓]

Goal: replace reliance on a single simple classifier with a properly evaluated predictive ensemble.

- [✓] Logistic baseline and direction model
- [✓] Expected-return, downside/risk, volatility, regime and abstention heads
- [✓] Tree/boosting family evaluation and evidence-based complexity gate
- [✓] Probability calibration and uncertainty reliability
- [✓] OOF meta-model and multiple prediction horizons
- [✓] Fees/spread/slippage conservative cost treatment
- [✓] Long/short/no-trade evaluation
- [✓] Precision/recall, balanced accuracy, calibration, expected return, drawdown and risk metrics
- [✓] Reproducible artifacts, lineage and feature-schema versioning
- [✓] Statistical promotion gate
- [✓] Separate chronological selection/calibration/untouched OOS periods
- [✓] Paired bootstrap CI for candidate-vs-baseline promotion
- [✓] Fail-closed invalid/stale/unpromoted/feature-mismatched loading
- [✓] Unpromoted candidates blocked from production
- [✓] Sequence models deferred when evidence does not justify complexity

**STAGE 3 STATUS: [✓] COMPLETED**

### Stage 3 verification evidence
- Backend CI: **PASS — 127 tests passed, 1 non-failing dependency deprecation warning**
- Predictive ensemble, calibration, OOS evaluation, promotion gate and production routing: **PASS**
- Live-money execution remained disabled

---

# STAGE 4 — Adaptive Intelligence [✓]

Goal: learn which signals/models work under which conditions without uncontrolled self-modification.

- [✓] Regime-conditioned model/signal performance tracking
- [✓] Model reliability by regime and horizon
- [✓] Calibration drift, concept/data drift and unfamiliar-state detection
- [✓] Adaptive confidence and evidence-based model weighting
- [✓] Learned abstention threshold
- [✓] Champion/challenger evaluation and quarantine
- [✓] Untouched OOS and multi-period/regime stability gates
- [✓] Rollbackable versions and accept/reject evidence
- [✓] Persistent observations/candidate lifecycle in Supabase
- [✓] Restart hydration and restrictive RLS policies
- [✓] Decision-only adaptation with immutable production artifacts
- [✓] Automated drift/familiarity/challenger regression coverage

**STAGE 4 STATUS: [✓] COMPLETED**

### Stage 4 verification evidence
- Backend CI: **PASS — 134 tests passed, 1 non-failing Starlette/httpx deprecation warning**
- Adaptive reliability, drift, unfamiliar-state, challenger, persistence and fail-closed behavior: **PASS**
- Supabase security advisors: **0 security lints**
- Render deployment/startup: **PASS/LIVE**
- Live-money execution remained disabled

---

# STAGE 5 — Advanced Decision Engine [✓]

Goal: rebuild council/scenario/adversarial architecture around learned intelligence.

- [✓] Canonical specialist market-state ingestion
- [✓] Learned specialist weighting from realized evidence
- [✓] Model agreement/disagreement and calibrated decision fusion
- [✓] Learned scenario probabilities and expected-value analysis
- [✓] Adversarial thesis challenge and contradiction detection
- [✓] Empirical matched counterfactual analysis
- [✓] Evidence/prediction/uncertainty/action separation
- [✓] Final abstention gate and absolute risk vetoes
- [✓] Auditable reasons and independent decision-quality ledger
- [✓] Decision-only API with no execution authority
- [✓] Production predictive artifacts immutable
- [✓] Regression coverage for all Stage 5 gates
- [✓] Deployment/startup verification

**STAGE 5 STATUS: [✓] COMPLETED**

### Stage 5 verification evidence
- Backend CI: **PASS — 143 tests passed, 1 non-failing Starlette/httpx deprecation warning**
- Council, fusion, scenarios, adversarial, contradiction, counterfactual, abstention, risk-veto and decision-quality tests: **PASS**
- Render final Stage 5 deployment: **LIVE**, root HTTP 200
- Live-money execution remained disabled

---

# STAGE 6 — Autonomous Position Intelligence [✓]

Goal: manage open positions continuously according to changing expected value and thesis integrity rather than blindly relying on fixed TP/SL rules.

- [✓] Continuous position-state representation
- [✓] Original trade thesis and entry evidence
- [✓] Continuous thesis-integrity reassessment
- [✓] Dynamic continuation value and downside/risk recomputation
- [✓] Thesis invalidation, momentum/flow/regime deterioration and shock detection
- [✓] Dynamic reduce/hold/exit decisions
- [✓] Adaptive profit protection and partial exits
- [✓] Dynamic protective levels with exchange safety
- [✓] Partial-fill/order-failure handling
- [✓] Critical-action exchange reconciliation
- [✓] Restart recovery and duplicate-management protection
- [✓] Historical-style and paper validation
- [✓] Persistent thesis/protection/peak/remaining-quantity state
- [✓] Shared adaptive intelligence integration without predictive-artifact mutation
- [✓] Deployed Stage 6 smoke gate

**STAGE 6 STATUS: [✓] COMPLETED**

### Stage 6 verification evidence
- Backend CI: **PASS — 152 tests passed, 1 non-failing Starlette/httpx deprecation warning**
- Stage 6 position-engine, paper, replay, failure, reconciliation, restart and duplicate-order tests: **PASS**
- Supabase security advisors: **0 security lints**
- Production Stage 6 endpoint smoke: **PASS**
- Final Render deployment: **LIVE**, application startup complete, root HTTP 200
- Live-money execution remained disabled

---

# STAGE 7 — Self-Learning Research Loop [✓]

Goal: turn every completed trade into structured research without allowing individual outcomes to corrupt production intelligence.

- [✓] Capture complete decision snapshots
- [✓] Capture realized execution and market outcomes
- [✓] Attribute outcome to models, signals, regime and decision layers
- [✓] Identify prediction errors and calibration errors
- [✓] Identify regime-classification errors
- [✓] Identify failed/weak signals
- [✓] Generate candidate improvements
- [✓] Keep candidates quarantined from production
- [✓] Automatically run reproducible backtests
- [✓] Run walk-forward evaluation
- [✓] Run untouched out-of-sample evaluation
- [✓] Run stress/robustness evaluation
- [✓] Compare candidate against production champion
- [✓] Require statistical and economic improvement
- [✓] Require improvement after realistic costs
- [✓] Promote only through an explicit approval gate
- [✓] Keep complete experiment/model lineage
- [✓] Support rollback to a known-good model

### Stage 7 requirements added during implementation
- [✓] Persist research snapshots, candidates and experiment results in server-only Supabase tables
- [✓] Automatically copy every completed learning outcome into the Stage 7 research snapshot path
- [✓] Restore research snapshots/candidates after backend restart
- [✓] Keep research candidates quarantined and execution-authority-free
- [✓] Use deterministic historical replay with realistic slippage/fee treatment
- [✓] Use chronological walk-forward folds and a distinct untouched OOS period
- [✓] Use deterministic paired bootstrap confidence intervals for candidate-vs-champion deltas
- [✓] Require economic improvement, positive statistical CI, drawdown constraint, walk-forward validity and stress validity before promotion eligibility
- [✓] Add an explicit approval contract requiring an approver and known-good rollback target; approval does not mutate production automatically
- [✓] Preserve reproducibility fingerprints and experiment lineage
- [✓] Add a deployed Stage 7 smoke gate
- [✓] Maintain restrictive RLS on research persistence

### Stage 7 verification evidence
- Backend GitHub Actions CI: **PASS — 157 tests passed, 1 non-failing Starlette/httpx deprecation warning**
- Stage 7 research ingestion, outcome attribution, candidate quarantine, deterministic replay, reproducibility, OOS gate, walk-forward, stress and production-authority tests: **PASS**
- The initial Stage 7 CI exposed two test-gate defects; both were corrected and the final suite passed. No failing test was ignored or waived.
- Supabase Stage 7 research tables (`research_snapshots`, `research_candidates`, `research_experiments`) created with RLS enabled and explicit deny policies; security advisors: **0 lints**
- Production learning outcome path now feeds research snapshots automatically
- Explicit approval/rollback governance exists without automatic production mutation
- Production Stage 7 smoke: **PASS**
- Final deployed Stage 7 code commit: **1bc92227385b25a9a5b3e5b2a2644cae23df297e**; Render deployment: **LIVE**, application startup complete
- Live-money execution remained disabled throughout Stage 7

**STAGE 7 STATUS: [✓] COMPLETED**

---

# STAGE 8 — Risk & Capital Intelligence [✓]

Goal: put an independent survival system around the intelligence so prediction cannot override capital protection.

- [✓] Build independent portfolio-risk engine
- [✓] Enforce maximum risk per trade
- [✓] Enforce maximum portfolio exposure
- [✓] Enforce correlated-exposure limits
- [✓] Enforce leverage limits
- [✓] Enforce daily/rolling loss limits
- [✓] Enforce drawdown circuit breakers
- [✓] Detect abnormal execution/slippage
- [✓] Detect exchange instability
- [✓] Detect stale/contradictory market data
- [✓] Implement global kill switch
- [✓] Implement exchange-specific kill switch
- [✓] Require safe recovery after emergency stop
- [✓] Ensure risk engine can veto any AI decision
- [✓] Ensure model confidence can never bypass capital controls
- [✓] Test catastrophic-failure scenarios
- [✓] Verify all risk controls against paper/testnet execution

### Stage 8 requirements added during implementation
- [✓] Add an independent `RiskCapitalEngine` with hard limits separate from model confidence and predictive logic
- [✓] Persist global and exchange-specific kill-switch state in server-only Supabase storage and hydrate it on restart
- [✓] Add restrictive RLS policies for Stage 8 risk-control persistence
- [✓] Install the Stage 8 veto around the autonomous entry-risk path while preserving the ability to reduce/close existing positions during a risk stop
- [✓] Add independent spread/slippage, data-freshness, contradiction, shock and execution-failure circuit checks
- [✓] Add deployed Stage 8 smoke verification for root, risk status, research continuity and live-money safety

### Stage 8 verification evidence
- Backend GitHub Actions CI: **PASS — 164 tests passed, 1 non-failing Starlette/httpx deprecation warning**
- Independent risk engine tests: **PASS** — per-trade risk, portfolio concentration, correlated exposure, leverage, daily/rolling loss, drawdown, stale/contradictory/shock data, kill switches, confidence bypass resistance and sizing caps
- Stage 8 entry-risk integration: **PASS** — independent veto wraps the existing autonomous entry gate; position-closing/reduction paths remain available for risk reduction
- Supabase migration `stage8_risk_capital_controls`: **APPLIED**; `risk_controls` RLS enabled with explicit deny policies
- Supabase security advisors: **0 lints**
- Global/exchange kill-switch state: **persistent and restart-hydrated**
- Paper/testnet safety: **PASS**; live-money execution remained disabled
- Final Render deployment: **LIVE** for commit `753d51138fbfe3d6ddcd58311475b68b5b823d75`; application startup completed successfully
- Deployed Stage 8 smoke: **PASS** — root reports Stage 8/independent risk engine, `/api/risk/status` returns execution-authority false and kill state, `/api/research/status` remains healthy, and live trading is false
- Latest deployed smoke-gate workflow: **PASS** for commit `37cc1195056db36541d4292f2ec7ed4c2c0e8e83`
- Live-money execution remained disabled throughout Stage 8

**STAGE 8 STATUS: [✓] COMPLETED**

---

# STAGE 9 — Advanced Frontend

Goal: make the frontend a serious operator/observability console for the intelligence, not merely a dashboard.

- [ ] Redesign the live brain overview around canonical backend state
- [ ] Show market regime
- [ ] Show direction probabilities and calibrated confidence
- [ ] Show expected edge after realistic costs
- [ ] Show uncertainty
- [ ] Show model agreement/disagreement
- [ ] Show data quality and source freshness
- [ ] Show order-flow/funding/OI/liquidity context
- [ ] Show news/event risk and provenance
- [ ] Show adversarial challenge results
- [ ] Show decision and explicit reasons
- [ ] Show what evidence would change the decision
- [ ] Show open-position thesis integrity
- [ ] Show expected continuation and dynamic risk
- [ ] Show profit-protection state
- [ ] Show autonomous-cycle health
- [ ] Show exchange reconciliation state
- [ ] Show model/version lineage
- [ ] Show learning/research candidates and promotion state
- [ ] Add robust mobile experience
- [ ] Add safe admin controls and authentication UX
- [ ] Add clear stale/error/degraded-state UX
- [ ] Verify frontend against production backend contracts

**STAGE 9 STATUS: NOT STARTED**

---

# TESTING LADDER — NON-NEGOTIABLE RELEASE GATES

- [ ] TEST 1 — Unit tests
- [ ] TEST 2 — Integration tests
- [ ] TEST 3 — Historical backtesting
- [ ] TEST 4 — Walk-forward testing
- [ ] TEST 5 — Out-of-sample testing
- [ ] TEST 6 — Stress testing
- [ ] TEST 7 — Monte Carlo / robustness testing
- [ ] TEST 8 — Paper trading
- [ ] TEST 9 — Binance/Bitget testnet or equivalent controlled environment
- [ ] TEST 10 — Very small controlled live deployment

## Live-money rule

**No real-money deployment is permitted simply because the software works.** It must pass the complete testing ladder and all independent risk/execution gates. A high prediction score does not guarantee profit, and HHHAI must never claim guaranteed profits or guaranteed avoidance of loss.

## Operating rule for future work

1. Read this file before beginning work.
2. Identify the first incomplete stage/item that is actually ready to work on.
3. Work in large coherent batches when multiple unchecked items are technically coupled.
4. Verify the result with tests or direct evidence.
5. Mark only genuinely verified items `[✓]`.
6. Never invent a new stage because a new source file was created.
7. Never silently rewrite this roadmap's stage structure.
8. Do not advance to the next stage while required work in the current stage remains incomplete.
