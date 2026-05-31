Known issue: Tensor-parallel hangs under sustained load

## Summary
Tensor-parallel (TP) replicas can intermittently **hang (no forward progress)** under sustained, mixed workloads (prefill + decode, long-context, bursty concurrency). This typically presents as **streaming responses stalling** and eventually **request timeouts**. In the worst case, one or more GPUs appear “wedged” until the replica is drained/restarted.

This page documents:
- Observable symptoms and common signatures
- Affected configurations (models, TP sizes, hardware)
- How to detect and confirm a TP stall
- Recommended mitigations (operational + config)
- Fix status by runtime version and feature flags

> If you are actively in an incident: jump to **Immediate mitigations** and then follow **Triage**.

---

## Impact / why we care
- **Customer impact:** Streaming completions stall; end-users see long hangs then timeouts. Enterprise Dedicated accounts are most affected due to higher sustained concurrency.
- **Operational impact:** Autoscaler thrash (p99 latency spikes trigger scale-up; wedged replicas never recover), elevated error rates, Sev1 potential.
- **Cost impact:** Stalled replicas waste GPU capacity and increase retry/duplicate compute.

---

## Symptoms (what you’ll see)
### External (API / client-facing)
- Streaming response starts and then stops emitting tokens mid-decode.
- Requests remain in-flight until client timeout (commonly 60–300s depending on route).
- Retry storms (client SDK retries or application-level retries) amplify load.

### Internal (replica-level)
- **No tokens produced** while GPU utilization may remain non-zero or drop unexpectedly.
- One TP rank appears stuck while others continue to enqueue work (or also stop).
- NCCL may not throw an explicit error; hangs can be “silent” without async error handling.

### Common failure signatures (taxonomy)
We currently bucket stalls into these high-level signatures (see `runtime.tp_stall_reason` below):
1. **collective_wait**: rank waiting for NCCL collective completion.
2. **collective_mismatch**: collective ordering divergence across ranks (often manifests as silent wait).
3. **scheduler_starvation**: collectives not scheduled fairly under prefill-heavy batching.
4. **stream_sync_gap**: CUDA stream/event ordering issue around NCCL boundary.
5. **kv_compaction_divergence**: KV cache compaction/fragmentation causes rank-side divergence.
6. **cancelled_stream_deadlock**: client disconnect/cancellation leaves ranks in inconsistent step state.

---

## Affected configurations
This issue has been observed most reliably under the following conditions.

### Workload characteristics
- Mixed prompt lengths (high variance) + streaming decode
- Long context (8k–32k) + high concurrent requests
- Bursty arrival patterns (spikes causing deep scheduler queues)
- Prefill-heavy traffic interleaved with ongoing decodes

### Deployment modes
- **Dedicated:** most common due to steady saturation and strict p99 expectations.
- **Hosted:** seen in a smaller number of high-throughput routes during peak bursts.

### Models and TP sizes (most affected)
| Category | Examples | Notes |
|---|---|---|
| Large dense, long-context | Llama-70B class, Qwen-72B class | Higher risk when TP=4/8 and context > 8k |
| MoE models | Mixtral 8x7B class | Sensitive to scheduling/fairness under mixed prefill/decode |
| Any model with aggressive batching | Any high-throughput route | Risk increases with deep queues and long prefill batches |

### Hardware / topology
| Hardware | Topologies seen | Notes |
|---|---|---|
| A100 80GB | NVLink intra-node | Most reproductions and customer reports |
| H100 80GB | NVLink intra-node | Also observed; typically at higher throughput |
| PCIe-only multi-GPU | Mixed | Less common in production; higher sensitivity to comms latency |

### Runtime versions
- Observed most often on **runtime 1.18.x** in the presence of CUDA graph capture + mixed workloads.
- Some signatures also present on early **1.19.0 canaries** prior to watchdog + ordering hardening.

---

## Detection signals (metrics, logs, traces)
### Dashboards
- See: **TP stall and timeout dashboard** (Confluence: `tp-stall-and-timeout-dashboard`).

### Key metrics (authoritative)
> Metric names reflect the current standards agreed with SRE; older versions may differ.

1) **Per-rank heartbeat**
- `runtime_tp_rank_heartbeat_seconds{replica_id, tp_rank, model, cluster}`
- Signal: heartbeats stop updating for one or more ranks.
- Rule of thumb: **gap > 10s** during active traffic is suspicious; **gap > 30s** is a likely stall.

2) **Stall reason counters**
- `runtime_tp_stall_reason_total{reason, model, tp_size, cluster}`
- Watch for spikes in `collective_wait`, `scheduler_starvation`, `cancelled_stream_deadlock`.

3) **In-flight collective counters**
- `runtime_nccl_inflight_collectives{replica_id, tp_rank, collective}`
- Signal: counter remains >0 with no progress (and heartbeat gap).

4) **Scheduler pressure indicators**
- `runtime_scheduler_queue_depth{replica_id, queue}`
- `runtime_scheduler_prefill_decode_ratio{replica_id}`
- Signal: queue depth grows while decode throughput collapses.

5) **Customer-visible indicators**
- `gateway_request_timeout_total{route, model}`
- `gateway_streaming_disconnect_total{route, model}`
- `request_p99_latency_ms{route, model}`

### Logs (what to grep)
Look for structured fields (availability depends on runtime version):
- `tp_stall_detected=true`
- `tp_rank=<int>` and `tp_size=<int>`
- `stall_reason=<enum>`
- `inflight_collectives=<int>`
- `last_progress_ms=<int>`

### NCCL debug (use with care)
Use only for time-boxed windows due to verbosity/perf impact. Coordinate with SRE (see INT: enable extra NCCL debug logging).
Suggested minimal settings for targeted capture:
- `NCCL_ASYNC_ERROR_HANDLING=1`
- `NCCL_DEBUG=WARN`
- `NCCL_DEBUG_SUBSYS=COLL`

Do NOT leave `NCCL_DEBUG=INFO` enabled broadly on production Dedicated clusters.

---

## Immediate mitigations (incident-safe)
These mitigations prioritize restoring service and preventing global GPU wedging.

### 1) Drain/replace suspect replicas
- Drain the affected replica(s) at the orchestrator level (preferred) rather than waiting for recovery.
- If multiple replicas show heartbeat gaps, drain in a staggered fashion to avoid full capacity loss.

### 2) Reduce concurrency and/or batch size (fastest risk reduction)
Recommended temporary caps (starting point; adjust per model/hardware):
- Reduce `tp.max_concurrency_per_replica` by **30–50%**
- Reduce `max_batch_prefill_tokens` (or equivalent) by **25–40%**
- Prefer limiting **prefill batch growth** over decode to preserve streaming responsiveness

### 3) Disable high-risk execution paths (if available)
Depending on runtime version and flags, consider:
- Disable CUDA graphs for TP decode on affected routes:
  - `runtime.tp.cuda_graphs_decode.enabled=false`
- Enable conservative stream/event synchronization around NCCL:
  - `runtime.tp.nccl_stream_sync_hardening=on`

### 4) Enable watchdog + safe abort (where supported)
If the watchdog feature is available in your runtime build, enabling it can prevent indefinite hangs:
- `runtime.tp_watchdog.enabled=true`
- `runtime.tp_watchdog.timeout_ms=<workload-specific>` (see below)

> Watchdog enables **controlled abort + replica drain** semantics instead of indefinite waits. This will increase error counts during true stalls but improves overall availability.

### 5) Load shedding / routing fallback (platform-level)
- If stall signals are present, enable overload protection to **reject/redirect new requests** away from the wedged replica pool.
- Prefer fallback to a compatible model variant/region if configured for Dedicated/Hosted.

---

## Recommended default mitigations (non-incident)
For clusters with frequent TP stall history, keep these guardrails enabled until runtime 1.19.2+ is fully deployed.

### Configuration guardrails
| Guardrail | Recommendation | Rationale |
|---|---|---|
| Concurrency cap | Set conservative `tp.max_concurrency_per_replica` | Reduces scheduler pressure and mixed-step divergence risk |
| Prefill batch cap | Limit `max_batch_prefill_tokens` | Prevents prefill-heavy batches from starving collectives/decodes |
| Async error handling | `NCCL_ASYNC_ERROR_HANDLING=1` | Surfaces errors sooner; reduces silent waits |
| Stream sync hardening | Enable if perf impact acceptable | Avoids event/stream ordering gaps around collectives |
| Watchdog | Enable with staged rollout | Converts indefinite hang into bounded failure + recovery |

### Watchdog timeout guidance
Use model/workload-specific defaults; avoid overly aggressive thresholds that create false positives under long-context.
- Small/medium (≤13B dense): start at **8–12s**
- Large dense (≥70B) TP=4/8: start at **15–25s**
- Long-context (≥16k): start at **25–40s**

Always stage changes:
1) Canary one cluster/route
2) Validate stall detection + error rate impact
3) Expand to Dedicated
4) Expand to Hosted

---

## Triage checklist (confirming a TP hang)
Follow the detailed steps in the playbook (Confluence: `tp-stall-triage-playbook`). Quick checklist:

1) **Confirm customer symptom**
- Streaming stuck mid-response, elevated timeouts, p99 jump.

2) **Check TP heartbeat gaps**
- Identify impacted `replica_id` and `tp_rank`.

3) **Correlate with in-flight collectives**
- `runtime_nccl_inflight_collectives` stuck >0 indicates comms boundary involvement.

4) **Check scheduler pressure**
- Queue depth rising while throughput drops suggests scheduler starvation component.

5) **Search logs for stall reason**
- Confirm `stall_reason` and `last_progress_ms`.

6) **Mitigate**
- Drain replica(s) and apply caps/flags.

7) **Capture artifacts** (only if safe)
- Minimal NCCL WARN logs
- Trace IDs for representative stuck requests
- Replica runtime logs around the stall window

---

## Fix status / roadmap
This is an active reliability initiative tracked in Linear: **ENG-4821**.

### Fixes delivered / in-flight
| Area | Status | References |
|---|---|---|
| Heartbeat + stall reason tagging | Landed | PR-21834; dashboard panels in observability pack PR-412 |
| Scheduler fairness / collective ordering guardrails | Landed | PR-21841; mitigates `scheduler_starvation` and ordering divergence |
| NCCL stream/event sync hardening | Landed (feature-flagged) | PR-21853; evaluate perf impact per model |
| Watchdog + safe abort + replica drain | Landed (feature-flagged) | PR-21860; ADR-0127 defines semantics |
| KV cache compaction safety | Landed | PR-21866; targets `kv_compaction_divergence` |
| Cancelled streaming deadlock fix | Landed | PR-21890; targets disconnect-triggered stalls |
| Improved NCCL error surfacing | In rollout | PR-21912; improves customer-facing diagnostics |
| Load shed on TP stall signal | In rollout | PR-21875 |

### Version guidance (operational)
| Runtime version | Guidance |
|---|---|
| ≤ 1.18.7 | Higher risk. Keep concurrency/prefill caps. Prefer disabling CUDA graphs for TP decode on affected routes. Drain-on-stall is primary recovery. |
| 1.19.0–1.19.1 | Use only with watchdog and stream sync hardening flags in canary first. Validate stall reasons and noise. |
| 1.19.2+ | Target stable baseline for Dedicated + high-throughput Hosted. Keep monitoring; reduce guardrails only after 2+ weeks without stall alerts on target workloads. |

### Rollout plan
- Dedicated-first rollout with explicit go/no-go gates: see Linear **ENG-4829**.
- Canary success metrics:
  - TP stall alerts reduced by **>80%** week-over-week (normalized)
  - Request timeouts attributable to TP stalls reduced to **<0.1%** of requests on canary routes
  - No sustained p99 regression > **5%** after enabling stream sync hardening/watchdog

---

## Customer communication notes (Support/CS)
When communicating with customers, avoid over-specifying a single root cause unless confirmed by artifacts. Recommended phrasing:
- “We have identified a class of TP forward-progress stalls under specific mixed workloads. We are rolling out runtime changes to improve collective ordering, stream synchronization, and add bounded-time watchdog recovery.”

If asking for customer artifacts, request:
- Affected model + TP size, GPU type, context length distribution
- Approx concurrency at time of stall
- Example request IDs / timestamps
- Whether stalls correlate with client disconnects

Support-facing summary and templated mitigations are maintained in: **customer-success-and-support/known-issues-and-workarounds/tp-timeouts-workaround-dedicated**.

---

## Owner / escalation
- Primary owner: **Eng Serving Runtime (Kernel + Scheduling)**
- Incident escalation: **#eng-sre** + **#eng-runtime** (include replica IDs, model, TP size, and stall reason metrics)
- Contacts:
  - Runtime TL: Noah Patel
  - Kernel: Hiro Tanaka / Grace Kim
  - Scheduling: Sofia Ivanova / Julian Moore
  - SRE lead: Rafael Mendes / Aisha Bello

---

## Changelog
- 2025-03-04: Updated version guidance for 1.19.2+ rollout gates; added cancelled-streaming deadlock signature and mitigation.
- 2025-02-20: Added heartbeat and stall reason metric names; linked dashboard and ADR.
- 2024-11-12: Initial publication based on Dedicated TP timeout reports and early repro findings.
