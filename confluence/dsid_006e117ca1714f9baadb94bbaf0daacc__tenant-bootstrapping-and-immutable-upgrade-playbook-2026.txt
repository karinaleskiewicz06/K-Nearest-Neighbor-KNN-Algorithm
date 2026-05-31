Tenant bootstrapping and immutable upgrade playbook

Summary

This playbook documents the end-to-end tenant bootstrapping procedure and an immutable upgrade strategy for private (VPC and on-prem) Redwood deployments. It is written for field engineers, site reliability engineers, and customer platform owners responsible for onboarding new tenants and performing non-disruptive, auditable upgrades to runtime components.

Scope

- Applies to private deployments using the Redwood control-plane with tenant isolation via service subnets and reserved GPU pools.
- Covers: tenant bootstrap (network, identity, storage, model-store sync), immutable upgrade flow (operator-driven image rollouts), canary validations, and safe rollback/cutover.
- Excludes: hardware procurement and low-level firmware tasks (see onprem-packaging/ephemeral-boot-and-attestation-playbook).

Goals

- Fast, repeatable tenant bootstrapping with automated checks to reach steady state within a predictable window.
- Immutable upgrades that minimize blast radius and provide a deterministic rollback path.
- Clear ownership, checkpoints, and telemetry to prove functional parity after upgrades.

Key components referenced

- redwood-control-plane: central control services (API, orchestrator, and telemetry ingest).
- redwood-operator: cluster operator that manages node pools, model-store mounts, and immutable image rotations.
- node-agent: per-host agent responsible for KV cache warmup, health probe registration, and telemetry shipping.
- runtime-proxy: request routing layer that enforces per-tenant rate limits and routing policies.
- model-store: object-store-backed repository for model artifacts and quantized variants.

Prerequisites (must be satisfied before bootstrapping)

- Network: BGP/VPC peering established and tenant CIDR allocated. Firewall rules allow control-plane management ranges and telemetry egress.
- KMS: Customer KMS key(s) provisioned and accessible by control-plane service account.
- Storage: S3/GCS-compatible model-store bucket created with lifecycle policy; credentials provisioned.
- Images: Immutable images for redwood-operator and runtime components pushed to customer registry (signature verified).
- Capacity: GPU node pool with min capacity >= requested tenant concurrency, verified in capacity planner.

Ownership matrix

- Field engineer: co-ordinates network, KMS, and registry checks; performs the initial bootstrap run.
- Customer platform owner: approves tenant acceptance criteria and provides production credentials.
- Redwood SRE: responsible for control-plane readiness, telemetry ingestion, and escalation during canary.

Compatibility matrix (quick reference)

| Component | Minimum version | Required config flags | Notes |
|---|---:|---|---|
| redwood-operator | v1.18.0 | immutable_rollout=true | Adds image-swap gating and canary labels |
| node-agent | v2.4.0 | kv_cache_prefetch=true | Enables KV pre-warm during boot |
| runtime-proxy | v3.2.1 | tenant-routing=dedicated | Required for per-tenant routing policies |
| model-store | N/A | s3_encryption=customer_kms | Object-store must support SSE-C/SSE-KMS

Step 1 — Pre-bootstrap checklist (15–30 minutes)

1. Confirm control-plane health: all core services return OK via /health endpoints and alerting is green.
2. Validate network ACLs and egress: run connectivity tests (curl/control-plane-agent) from a bastion into tenant subnet.
3. Verify KMS access: encrypt/decrypt a short payload using the tenant KMS key from the control-plane service account.
4. Confirm model-store sync: list bucket objects and sample-download a 10MB model shard.
5. Verify registry signatures: run cosign/registry-verifier on operator and runtime images.
6. Capacity sanity: query capacity planner for reserved GPU quota and node health.

Step 2 — Tenant bootstrap procedure (automated with operator)

1. Create tenant manifest (YAML) that includes: tenant-id, CIDR, KMS-key-arn, model-store-bucket, initial-node-pool-size, quotas. Example snippet:

   apiVersion: redwood/v1
   kind: Tenant
   metadata:
     name: tenant-acme
   spec:
     cidr: 10.60.128.0/24
     kmsKeyArn: arn:aws:kms:...:key/abcd
     modelStore: s3://acme-models
     nodePool:
       size: 4
       gpuType: a100

2. Apply manifest to redwood-operator: kubectl apply -f tenant-acme.yaml. Operator will create node pool resources and secrets.
3. Wait for node-agent boot: verify node-agent pods report status=Ready and kv_prefetch job completes.
4. Model pre-warm: redwood-operator triggers model-store warmup for pinned models defined in tenant manifest. Monitor progress via /api/v1/tenants/{id}/warmup.
5. Register tenant in runtime-proxy: runtime-proxy receives tenant routing config and updates routing tables.
6. Acceptance tests: run a scripted smoke test set (see checklist) that verifies generation endpoint, embedding endpoint, and token accounting.

Acceptance criteria (must pass before marking tenant active)

- Control-plane and runtime-proxy show tenant in state Active.
- Generation endpoint returns successful responses for 95% of requests in a 5-minute test.
- KV hit rate >= 70% for warmup models within 10 minutes.
- No critical alerts fired during bootstrap window.

Step 3 — Immutable upgrade strategy (operator-driven, canary-first)

Overview: upgrades are performed by swapping immutable images (operator/agents/runtime). We recommend immutable rollouts with staged canaries (0.5–2% traffic) then ramp to 25%/50%/100% while validating telemetry gates.

Pre-upgrade checks (run 24–72 hours before)

- Run regression test suite against staging control-plane.
- Ensure backups of KV cache (if applicable) and model-store object versioning are enabled.
- Verify node-agent auto-restart policies and cordon/drain playbooks tested in canary cluster.

Canary stages and validation gates

1. Canary deploy: update redwood-operator to point to new image tag and label canary=true for 1 host group.
2. Traffic split: direct 1% of tenant traffic to canary hosts (runtime-proxy weight).
3. Validation window: 30–60 minutes of live traffic with the following SLI checks:
   - p50/p95 generation latency within 10% of baseline
   - error rate (4xx/5xx) increase < 0.1 percentage points absolute
   - CPU/GPU utilization within expected envelope (no >15% regressions)
4. Functional parity: run deterministic prompt suite (10–20 prompts) and compute token-level BLEU/ROUGE scores vs baseline; flag divergence > 2% relative.
5. Observability check: ensure traces from canary hosts are visible and sampled at 100% for first 5 minutes.

Ramped rollout

- 1% -> 5% -> 25% -> 50% -> 100%. Each ramp requires the validation window to pass and sign-off from the on-call SRE. Typical wait: 15–60 minutes per stage depending on traffic volume.

Rollback policy

- Automatic rollback if any gate trips (latency spikes > 10% p95, error rate > 0.5% absolute, or functional parity divergence).
- Manual rollback: operator can revert image tag and re-run canary assessment. The operator tracks previous image tags for quick revert.

Immutability and stateful considerations

- Stateful services (e.g., persistent model caches) are versioned by directory layout: /models/{tenant}/{model}/v{semver}. New image writes to vNext while reads continue to vCurrent until cutover.
- Avoid in-place upgrades of on-disk KV formats; prefer side-by-side migration with feature flags.

Validation and acceptance tests (post-upgrade)

- Run full regression suite (automated) including: end-to-end latency profiles, quota accounting, security auth tests, and model fidelity checks.
- Manual QA: sample 100 prompts across tenant prompt types and flag any output regression.
- Billing/accounting verification: token accounting matches expected consumption within 0.25%.

Telemetry and observability configuration

- Minimum telemetry for every upgrade: request traces, model-id tag, node-image-tag, tenant-id, kv-hit/miss counters, GPU memory headroom, and operator-event stream.

| Metric | Alert threshold | Purpose |
|---|---:|---|
| p95_generation_latency_ms | > baseline * 1.10 for 10m | Detect latency regressions |
| generation_error_rate | > 0.5% absolute | Quick fail on breaking changes |
| kv_cache_hit_rate | < 60% for 15m | Warmup or cache regression indicator |
| node_restarts | > 5 in 30m | Unhealthy image or crashloop |

Common failure modes and remediation

- Node fails to join after image swap: Check operator logs for image pull errors and registry auth; revert image and escalate to SRE if signature mismatch.
- Model mismatch or missing shard: Verify model-store path and pre-warm logs; if missing, re-trigger model-store sync from snapshot.
- Token accounting drift: Re-run accounting job and compare token tallies; if discrepancy > 1%, open ticket and pause billing for affected tenant.

Disaster recovery and emergency rollback (fast path)

1. If a severe incident is detected (major error floods, API unavailable to >5% of users for >5 minutes), execute emergency rollback: label all suspect pods with previous image tag and drain new pods.
2. Re-enable previous routing weights immediately in runtime-proxy and set admission control to 0% for new image.
3. Notify stakeholders and create an incident with severity tags, attach timeline and telemetry snapshots.

Operational checklists and runbook links

- Bootstrapping smoke tests (automated): scripts/tenant-smoke-tests.sh — run and capture outputs.
- Upgrade runbook snippet: redwood-operator/upgrade-checklist.md.
- Rollback cookbook: runbooks/emergency-rollback.md.

Post-upgrade review and telemetry artifacts to collect

- Automated release report: include per-stage metrics, sample traces, regression suite results, and list of manual checks. Store report under release-manifests/{date}/tenant-{id}.
- Retrospective: if any gate failed, open a retro ticket with root cause and mitigation.

Appendix A — bootstrap smoke test checklist (executable)

- [ ] POST /v1/tenants/{id}/generate with sample prompt returns 200
- [ ] POST /v1/tenants/{id}/embed returns vector length matching model spec
- [ ] KV pre-warm metric reports >= 70% within T+10m
- [ ] Billing counters increment on request
- [ ] Control-plane audit entries created for tenant API keys

Appendix B — quick CLI commands

- Apply tenant manifest: kubectl apply -f tenant-acme.yaml
- Check operator status: kubectl -n redwood get pods -l app=redwood-operator
- List warmup jobs: curl -sS https://control-plane.internal/api/v1/tenants/tenant-acme/warmup
- Trigger rollback: kubectl -n redwood set image deployment/redwood-operator redwood-operator=registry/release:prev

Notes and links

- This playbook complements the private-deploy-onboarding-and-failover-runbook (see related_pages) and the post-installation telemetry checklist.
- For air-gapped or FIPS environments, follow the image signing and attestation steps in onprem-packaging/ephemeral-boot-and-attestation-playbook.

Change log

- 2025-08-12: Initial draft (A. Kapoor).
- 2026-02-03: Add immutable-state recommendations and updated acceptance thresholds (reviewers: D. Morales and C. Jensen).

Contact

For questions or emergency assistance, ping #infra-private-deploys and escalate to on-call SRE following the oncall-and-incident-response/incident-process.
