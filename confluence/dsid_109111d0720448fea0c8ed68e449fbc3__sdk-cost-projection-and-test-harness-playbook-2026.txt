SDK cost-projection and test-harness playbook

Summary

This playbook documents the standardized approach for projecting token-level cost, validating cost optimizations, and running a repeatable local test harness for Redwood SDK integrations. It is intended for SDK engineers, integration owners, and developer advocates who need to estimate operational spend and validate fallback and batching strategies before customer rollout.

Why this exists

Redwood customers frequently ask for predictable cost budgets and safe ways to test cost-saving features (batching, caching, quantization hints). Historically teams have used ad-hoc scripts with inconsistent telemetry and environment parity. This playbook provides:
- A reproducible local harness that mirrors production token accounting semantics.
- A cost-projection workflow that combines real telemetry, sampling, and synthetic traffic.
- Standardized config templates and acceptance criteria to gate product changes that affect unit economics.

Scope and audience

Scope: server-side SDK usage (Node/Python/Go), client-side bundling scenarios (lightweight adapters), and CI test-harnesses used by Developer Experience and Integration teams.
Audience: SDK engineers, integration owners, solutions engineers, and product owners.

Goals and success metrics

Primary goals:
1) Produce a credible 30/90/365 day cost projection with +/- 10% confidence for a given integration.
2) Provide a CI-friendly harness that detects regressions in token usage and latency due to SDK changes.
3) Standardize downstream telemetry mappings so chargebacks and customer-facing invoices are consistent.
Key metrics (examples):
- Token-cost-per-request baseline (tokens/request)
- 95th percentile per-request token delta after optimization changes
- False-positive rate for CI cost regressions (< 2%)

High-level workflow

1. Baseline collection
   - Instrument a short sampling window in production (7 days recommended) to capture token histogram, sequence length distribution, and call types (embedding/gen/chat).
   - Export a 10k sample trace (anonymized) to the local cost-sim harness. See "Telemetry export format" section.
2. Define projection scenario
   - Scenario inputs: expected daily active callers (DAC), avg calls/caller, percent gen vs embeddings, expected caching/hit rate, quantization profile (if applicable).
3. Run local simulation
   - Use the harness to replay the sample with configured workload scaling factors. The harness models batching, prefix caching, and tokenization behavior to compute token counts and simulated billing.
4. Validate and gate
   - If a code change reduces tokens/request by > 5% on 95th percentile without increasing latency beyond SLO, mark as a candidate optimization.
   - If tokens increase > 3% at 95th percentile, open a regression triage ticket and block release until root cause is addressed.

Telemetry export format (canonical JSON)

All local simulations accept a compact trace JSON that follows these fields. The harness will reject payloads that are missing required fields. Example:

```json
[
  {
    "trace_id": "abc123",
    "timestamp": "2026-01-01T12:12:12Z",
    "route": "/v1/generate",
    "sdk": "node",
    "call_type": "generation",
    "input_tokens": 120,
    "output_tokens": 310,
    "metadata": {"prompt_type": "assistant-note"},
    "response_ms": 85
  }
]
```

(Strict note: for this internal format, fields are lightly validated — metadata freeform is allowed, but token fields and route are mandatory.)

Local test harness quickstart (Linux / macOS)

Prereqs: Docker, python3.11 (or node 18 for JS harness), jq

1. Clone the local-harness repo (internal):
   git clone git@github.com:redwood-internal/devx-cost-sim.git
2. Create a working env and install deps:
   cd devx-cost-sim
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
3. Import a telemetry sample (example stores in data/samples):
   ./tools/import_sample.py --input ../samples/sample-prod-10k.json --name=baseline-jan-10k
4. Run a projection:
   ./bin/run_simulation --sample baseline-jan-10k --scale 10 --scenario batching-enabled,cache-50
5. Output: reports/<sample>-scale-10.json with per-route token breakdown, simulated bill, and suggested optimizations.

Example scenario configurations (YAML)

Below is a minimal scenario template used by CI and local runs. Save as scenarios/standard-batching.yaml

```yaml
name: standard-batching
scale_factor: 50
caller_profile:
  dac: 1200
  avg_calls_per_day: 4
workload_mix:
  generation: 0.7
  embeddings: 0.2
  reranking: 0.1
caching:
  prefix_cache_hit_rate: 0.45
  kv_cache_enabled: true
quantization_profile: \"q4_0\"
batched_window_ms: 8
```

CI integration and gating

- Add a simulation stage in PR pipelines that runs the harness against a canonical sample. Use the scenario that best approximates expected production traffic for the change.
- Failure conditions that should block merges by default:
  1. +>3% token increase at p95 for any route.
  2. +>5% tokens at p99 for a route labeled "high-cost" (admin-configurable).
  3. Degradation in simulated 99th latency beyond SLO.
- When a gate fails, a follow-up triage checklist must be attached to the PR, including: differential traces, stack profile, and responsible owner.

SDK integration snippets (illustrative)

Node (short):

```js
// Use the SDK to capture token metrics locally before sending to production
const Redwood = require('@redwood/sdk');
const client = new Redwood({ apiKey: process.env.RW_KEY });

async function captureAndSend(prompt) {
  const start = Date.now();
  const res = await client.generate({ model: 'redwood-1', prompt });
  const elapsed = Date.now() - start;
  // Local capture call to harness-import endpoint (dev only)
  await fetch('http://localhost:8080/_import', {
    method: 'POST',
    headers: {'content-type':'application/json'},
    body: JSON.stringify({
      trace_id: Math.random().toString(36).slice(2),
      timestamp: new Date().toISOString(),
      route: '/v1/generate',
      sdk: 'node',
      call_type: 'generation',
      input_tokens: res.usage.input_tokens,
      output_tokens: res.usage.output_tokens,
      response_ms: elapsed
    })
  });
  return res;
}
```

Python (short):

```py
from redwood import Client
import requests
client = Client(api_key=os.getenv('RW_KEY'))

def capture_and_send(prompt):
    res = client.generate(prompt=prompt, model='redwood-1')
    payload = {
        'trace_id': uuid.uuid4().hex,
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'route': '/v1/generate',
        'sdk': 'python',
        'call_type': 'generation',
        'input_tokens': res['usage']['input_tokens'],
        'output_tokens': res['usage']['output_tokens'],
        'response_ms': res['latency_ms']
    }
    requests.post('http://localhost:8080/_import', json=payload)
    return res
```

Canonical fixes and troubleshooting (top issues we've seen)

1) Unexpected token inflation after template refactor
   Symptoms: per-request tokens increased by 15% at p95 after changing prompt interpolation.
   Fix checklist:
   - Compare tokenized prompt before/after using local tokenizer (tools/tokenize.py).
   - Look for duplicated prefixes or accidental debug logs included in prompt metadata.
   - Re-run harness with identical samples to validate.

2) Batching disabled by runtime change
   Symptoms: cost jump with no change in prompt sizes.
   Fix checklist:
   - Check SDK config: verify batched_window_ms not set to 0 and that async queue is healthy.
   - Inspect runtime logs for queue backpressure or worker restarts (runbook: eng-serving-runtime/runbooks/batching-queue-failures).
   - If batching cannot be recovered, apply temporary canary config disabling large fan-out routes.

3) Telemetry mismatch between harness and billing pipeline
   Symptoms: simulated bill does not match invoice-level cost granularities.
   Fix checklist:
   - Ensure mapping table in tools/price-map.csv matches current pricing and embeddings vs gen multipliers.
   - Verify tokenization differences (byte-pair vs unicode-aware tokenizers) and align tokenizer version.
   - Re-run validation script: tools/validate_pricing_alignment.py

Operational ownership and on-call

- Owner: Developer Experience SDK team (pager: #devx-sdks)
- Support escalation: 48-hour SLA for triage, 7-day SLA for root cause analysis for blocked releases.
- Long-term tracking: open regressions go into the "SDK cost regressions" project board and must include simulation reports and counterfactual inputs.

Acceptance criteria for enabling optimizations in production

1. Automated simulation shows median and p95 token reduction for targeted routes.
2. No increase in simulated p99 latency beyond SLOs (defined in observability-standards).
3. End-to-end integration test passes in CI with cost-checks enabled.
4. Rollout plan with canary (10% traffic -> 50% -> 100%) and auto-rollback trigger if production telemetry deviates from simulation by > 10% at p95.

Appendix: useful commands and small utilities

- Import sample: ./tools/import_sample.py --input data/raw/sample.json --name my-sample
- Run a single scenario locally: ./bin/run_simulation --sample my-sample --scenario scenarios/standard-batching.yaml
- Local web UI (dev): docker-compose up (port 8080) — shows token histograms and differential trace viewer
- Validate pricing: python tools/validate_pricing_alignment.py --sample reports/my-sample.json

Related pages and references

- Runtime batching runbook: /confluence/eng-serving-runtime/runbooks/batching-queue-failures
- Observability SLOs and thresholds: /confluence/eng-platform/dashboards-and-alerts/slo-and-error-budgets
- SDK telemetry contract: /confluence/eng-developer-experience/api-examples/sdk-telemetry-telemetry-contract-debug-workflow-2026

Contact and feedback

If you find gaps in scenario coverage or want help modeling a customer-specific workload, open a ticket under the "devx-sdks" board or ping Priya Nambiar on Slack (#devx-sdks).

Change log

- 2025-08-12: Created initial draft and sample harness.
- 2025-11-04: Added CI gating rules and scenario templates.
- 2026-01-15: Expanded troubleshooting section and formalized acceptance criteria.
