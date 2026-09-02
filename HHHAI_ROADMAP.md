# HHHAI Master Engineering Roadmap

**Authoritative roadmap for HHHAI.**

This file is the project control record. It is not a source-code stage and it must not be replaced by creating files named after stages. Before beginning work, check the current stage and unchecked items here. When an item is genuinely implemented **and verified**, change only that item from `[ ]` to `[✓]`. Do not mark work complete merely because code exists.

## Current position

**STEP 0 — FORENSIC AUDIT + BASELINE: [✓] COMPLETED**

**STAGE 1 — STABILIZE THE FOUNDATION: [✓] COMPLETED**

**STAGE 2 — BUILD THE REAL MARKET DATA INTELLIGENCE LAYER: [✓] COMPLETED**

**STAGE 3 — BUILD THE REAL PREDICTIVE BRAIN: [✓] COMPLETED**

**CURRENT: STAGE 4 — ADAPTIVE INTELLIGENCE**

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

- Backend GitHub Actions CI: **PASS**
- Backend test suite: **108 passed, 1 non-failing deprecation warning**
- Exchange boundary tests: **PASS**
- Frontend/backend contract tests: **PASS**
- Supabase RLS remediation: **PASS**
- Render production deployment: **LIVE**
- Vercel production deployment: **READY**
- Live execution safety gate: **fail-closed**

---

# STAGE 2 — Build the Real Market Data Intelligence Layer [✓]

Goal: give HHHAI a truthful, time-aligned representation of the market it is actually trading.

- [✓] Define the canonical point-in-time market-state schema
- [✓] Implement multi-timeframe OHLCV representation
- [✓] Implement robust price-structure features
- [✓] Implement volatility and volatility-regime features
- [✓] Implement volume/volume-profile features where data supports them
- [✓] Implement order-book depth and imbalance features
- [✓] Implement trade-flow/aggressor-pressure features where reliably available
- [✓] Implement funding-rate history
- [✓] Implement open-interest history and change features
- [✓] Implement liquidation data where a trustworthy historical source is available
- [✓] Implement spread, depth and liquidity-stress features
- [✓] Implement cross-asset/cross-market relationships
- [✓] Implement BTC/ETH/market-wide risk context
- [✓] Implement market-regime state representation
- [✓] Implement news-event ingestion with timestamps and provenance
- [✓] Implement richer sentiment/event classification with source credibility
- [✓] Implement historical-data enrichment and caching
- [✓] Enforce point-in-time joins with no future information
- [✓] Measure feature coverage and data quality continuously
- [✓] Make training and live feature schemas identical
- [✓] Add data-quality degradation/abstention behavior
- [✓] Validate the resulting dataset across multiple market regimes

**STAGE 2 STATUS: [✓] COMPLETED**

### Stage 2 verification evidence

- Canonical market-state schema and point-in-time join tests: **PASS**
- Multi-timeframe OHLCV, structure, volatility and volume tests: **PASS**
- Binance/Bitget market-data provider integration: **PASS**
- Order-book, spread, trade-flow and liquidation context: **PASS**
- Funding and open-interest history/change support: **PASS**
- BTC/ETH cross-asset correlation and market-risk context: **PASS**
- Timestamped news ingestion, sentiment, relevance, impact and source credibility: **PASS**
- Historical market-data caching and deterministic replay: **PASS**
- Canonical training/live feature projection: **PASS**
- Data-quality threshold and fail-closed degradation: **PASS**
- Cross-asset/regime validation tests: **PASS**
- Final backend verification: **118 tests passed, 1 non-failing deprecation warning**
- Render deployment of final Stage 2 commit: **LIVE**
- Live-money execution remained disabled throughout Stage 2

---

# STAGE 3 — Build the Real Predictive Brain [✓]

Goal: replace reliance on a single simple classifier with a properly evaluated predictive ensemble.

- [✓] Preserve Logistic Regression as an honest baseline
- [✓] Build a direction model
- [✓] Build an expected-return model
- [✓] Build a downside/risk model
- [✓] Build a volatility model
- [✓] Build a market-regime model
- [✓] Build a no-trade/abstention model
- [✓] Evaluate tree/boosting and other appropriate model families
- [✓] Evaluate sequence/temporal models where justified by data volume
- [✓] Prevent model complexity from being adopted without baseline improvement
- [✓] Calibrate predictive probabilities
- [✓] Measure uncertainty and confidence reliability
- [✓] Build out-of-sample ensemble/meta-model evaluation
- [✓] Include fees, spread and slippage in objective/evaluation using a conservative combined execution-cost assumption
- [✓] Evaluate multiple prediction horizons
- [✓] Evaluate long, short and no-trade separately
- [✓] Measure precision/recall, balanced accuracy, calibration and expected return
- [✓] Measure drawdown and risk-adjusted performance
- [✓] Create reproducible training runs and artifacts
- [✓] Add model lineage and feature-schema versioning
- [✓] Establish a statistically defensible model-promotion gate

### Stage 3 requirements added during implementation

- [✓] Use separate chronological model-selection, calibration and untouched OOS test periods to prevent selection/calibration leakage
- [✓] Use paired bootstrap confidence intervals on matched OOS net returns for candidate-vs-baseline promotion
- [✓] Fail closed on invalid, stale, unpromoted or feature-mismatched predictive artifacts
- [✓] Keep unpromoted multi-head candidates out of production decisions; production uses only the promoted brain or validated baseline fallback
- [✓] Record explicit feature fingerprint, artifact schema, model family, cost assumption, horizons and promotion evidence in the model manifest
- [✓] Keep sequence models deferred when current data coverage/sample volume does not justify their added complexity

### Stage 3 verification evidence

- Backend CI: **PASS — 127 tests passed, 1 non-failing dependency deprecation warning**
- Logistic baseline, multi-head ensemble and tree/boosting family coverage: **PASS**
- Expected return, downside, volatility, regime and abstention heads: **PASS**
- Temporal probability calibration using a disjoint calibration period: **PASS**
- OOF meta-model construction: **PASS**
- Untouched chronological OOS evaluation: **PASS**
- Fees/spread/slippage conservative cost treatment: **PASS**
- Long/short/no-trade and calibration metrics: **PASS**
- Paired bootstrap statistical promotion gate: **PASS**
- Artifact schema/feature fingerprint/fail-closed loading: **PASS**
- Production predictor routing: **promoted brain first; validated Logistic baseline only if no promoted brain is available**
- Sequence-model decision: **deferred by evidence, not omitted**
- Render deployment pipeline for final Stage 3 code: **VERIFIED through deployment lifecycle; final commit deployment updating**
- Live-money execution remained disabled throughout Stage 3

**STAGE 3 STATUS: [✓] COMPLETED**

---

# STAGE 4 — Adaptive Intelligence

Goal: make HHHAI learn which signals/models work under which conditions without uncontrolled self-modification.

- [ ] Build regime-conditioned model performance tracking
- [ ] Learn signal reliability from realized outcomes
- [ ] Learn model reliability by regime and horizon
- [ ] Track prediction calibration drift
- [ ] Detect concept/data drift
- [ ] Detect when the system enters an unfamiliar market state
- [ ] Adapt confidence based on historical reliability
- [ ] Adapt model weighting from validated evidence rather than fixed weights
- [ ] Learn abstention thresholds from validation data
- [ ] Build champion/challenger model evaluation
- [ ] Keep production models immutable until promotion criteria are met
- [ ] Quarantine candidate improvements
- [ ] Evaluate candidates on untouched out-of-sample data
- [ ] Require stability across multiple periods/regimes
- [ ] Maintain rollbackable model versions
- [ ] Record why an adaptive change was accepted or rejected

**STAGE 4 STATUS: NOT STARTED**

---

# STAGE 5 — Advanced Decision Engine

Goal: rebuild the current council/scenario/adversarial architecture around real learned intelligence.

- [ ] Replace hand-coded specialist weighting where learned evidence is available
- [ ] Make specialist agents consume the canonical market state
- [ ] Add model-agreement/disagreement intelligence
- [ ] Build calibrated decision fusion
- [ ] Build learned scenario probabilities/distributions
- [ ] Build expected-value scenario analysis
- [ ] Upgrade adversarial challenge to test the actual trade thesis
- [ ] Add contradiction detection across models and market evidence
- [ ] Upgrade counterfactual analysis using empirical/learned outcomes
- [ ] Separate evidence, prediction, uncertainty and action
- [ ] Implement final trade gate with explicit abstention
- [ ] Produce an auditable reason for every decision
- [ ] Track decision quality independently from trade P/L
- [ ] Ensure safety/risk vetoes cannot be overridden by model confidence

**STAGE 5 STATUS: NOT STARTED**

---

# STAGE 6 — Autonomous Position Intelligence

Goal: allow HHHAI to manage open positions continuously according to changing expected value and thesis integrity rather than blindly relying on fixed TP/SL rules.

- [ ] Build continuous position-state representation
- [ ] Record the original trade thesis and evidence
- [ ] Continuously reassess thesis integrity
- [ ] Recompute expected continuation value
- [ ] Recompute downside/risk dynamically
- [ ] Detect thesis invalidation
- [ ] Detect momentum/flow/regime deterioration
- [ ] Detect adverse news and market-wide shocks
- [ ] Implement dynamic reduce/hold/exit decisions
- [ ] Implement adaptive profit protection
- [ ] Implement partial-exit logic where justified
- [ ] Implement dynamic protective levels while preserving exchange safety
- [ ] Handle partial fills and order failures safely
- [ ] Reconcile exchange position state after every critical action
- [ ] Recover open-position state after backend restart
- [ ] Prevent position-management loops from creating duplicate orders
- [ ] Validate autonomous management in historical and paper environments

**STAGE 6 STATUS: NOT STARTED**

---

# STAGE 7 — Self-Learning Research Loop

Goal: turn every completed trade into structured research without allowing individual outcomes to corrupt production intelligence.

- [ ] Capture complete decision snapshots
- [ ] Capture realized execution and market outcomes
- [ ] Attribute outcome to models, signals, regime and decision layers
- [ ] Identify prediction errors and calibration errors
- [ ] Identify regime-classification errors
- [ ] Identify failed/weak signals
- [ ] Generate candidate improvements
- [ ] Keep candidates quarantined from production
- [ ] Automatically run reproducible backtests
- [ ] Run walk-forward evaluation
- [ ] Run untouched out-of-sample evaluation
- [ ] Run stress/robustness evaluation
- [ ] Compare candidate against production champion
- [ ] Require statistical and economic improvement
- [ ] Require improvement after realistic costs
- [ ] Promote only through an explicit approval gate
- [ ] Keep complete experiment/model lineage
- [ ] Support rollback to a known-good model

**STAGE 7 STATUS: NOT STARTED**

---

# STAGE 8 — Risk & Capital Intelligence

Goal: put an independent survival system around the intelligence so prediction cannot override capital protection.

- [ ] Build independent portfolio-risk engine
- [ ] Enforce maximum risk per trade
- [ ] Enforce maximum portfolio exposure
- [ ] Enforce correlated-exposure limits
- [ ] Enforce leverage limits
- [ ] Enforce daily/rolling loss limits
- [ ] Enforce drawdown circuit breakers
- [ ] Detect abnormal execution/slippage
- [ ] Detect exchange instability
- [ ] Detect stale/contradictory market data
- [ ] Implement global kill switch
- [ ] Implement exchange-specific kill switch
- [ ] Require safe recovery after emergency stop
- [ ] Ensure risk engine can veto any AI decision
- [ ] Ensure model confidence can never bypass capital controls
- [ ] Test catastrophic-failure scenarios
- [ ] Verify all risk controls against paper/testnet execution

**STAGE 8 STATUS: NOT STARTED**

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
