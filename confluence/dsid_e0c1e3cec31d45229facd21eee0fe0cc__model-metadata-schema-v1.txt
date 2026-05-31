Model Metadata Schema v1 (Model Registry + Serving)

## Purpose
This document defines **Model Metadata Schema v1** used by the Model Registry and Control Plane to:
- Provide a single source of truth for **served model identity**, **versioning**, **compatibility**, and **deployment availability**.
- Enable automated **customer-visible release notes** generation (Console + docs) without leaking internal-only details.
- Support internal auditability (who changed what, when, and what was promoted) and safe rollout policy evaluation.

Schema v1 is intentionally scoped to information needed for **serving** (runtime/tokenizer/limits/features), **rollouts** (eligibility/risk/deprecations), and **change visibility** (public summaries + internal rationale/evidence pointers).

## Scope
### In scope
- Model + variant + quantization profile identity and relationships
- SemVer model versioning rules for served endpoints
- Compatibility semantics and constraints (tokenizer/runtime/client)
- Per-deployment availability and rollout eligibility
- Public vs internal field separation rules
- Quality signal attachment (IDs, links, guardrail outcomes) suitable for release note summaries

### Not in scope (by design)
- Full training provenance / dataset disclosure
- Full benchmark tables in the schema (link to external evidence instead)
- Customer notification subscription preferences (separate system)

## Record types
Schema v1 defines a single primary record: **ModelVersionMetadata**.

A *model* is a stable identity (`model_id`) and display name. A *model version* (`model_version`) is a SemVer release of that model in our serving environment. A model version contains one or more **profiles**, which capture quantization / runtime configuration that may vary while remaining within the same model version.

## Versioning rules
### 1) `schema_version`
- **Required.** SemVer.
- For this document: `schema_version = 1.0.0`.
- Schema changes follow SemVer:
  - MAJOR: breaking schema change requiring migration.
  - MINOR: backward-compatible new fields.
  - PATCH: clarifications / constraints that do not break existing valid payloads.

### 2) `model_version` (served model SemVer)
`model_version` is the customer-facing version used for pinning and change history.

Rules:
- **MAJOR**: breaking change for a pinned integration. Examples:
  - Tokenizer change that can materially change tokenization boundaries for typical prompts.
  - Removal of a supported feature contract (e.g., structured output mode) for a pinned route.
  - Context/window reduction.
- **MINOR**: non-breaking but behavior-changing improvements. Examples:
  - Quality/performance improvements that may change outputs but should not break protocol.
  - New supported serving features added.
- **PATCH**: bugfix / operational change intended to be behaviorally neutral. Examples:
  - Fix for streaming edge cases.
  - Throughput/latency improvements with no expected response quality shift.

Important: even PATCH changes can affect outputs in practice. Release notes must state *expected impact* and *risk*.

### 3) Immutability
- `model_id` is immutable.
- `model_version` is immutable once `promotion.status` is `promoted`.
- `upstream.revision` and `artifacts.digest` are immutable once promoted.

## Public vs internal data model
The schema includes two parallel note channels:
- `public`: safe for customer exposure (Console/docs/API). Subject to redaction and language rules.
- `internal`: for Redwood staff only (support, incident response, compliance).

**Invariant:** Any field under `internal` MUST NOT be returned by customer-facing APIs or rendered in customer UI unless the viewer has explicit internal/admin entitlements.

## Field reference (high level)
| Field | Required | Type | Notes |
|---|---:|---|---|
| `schema_version` | yes | semver string | Always `1.x.y` for v1 |
| `model_id` | yes | string | Stable identifier, DNS-like: `rw.<family>.<name>` |
| `model_version` | yes | semver string | Served version for pinning |
| `display_name` | yes | string | Human-friendly name |
| `model_family` | yes | enum | `chat`, `completion`, `embedding`, `rerank`, `multimodal` |
| `variant` | yes | enum | `base`, `instruct`, `chat`, `coder`, `vision`, `reasoning` |
| `status` | yes | enum | `active`, `deprecated`, `disabled`, `preview` |
| `upstream` | yes | object | Upstream model ref + license |
| `serving` | yes | object | Tokenizer/runtime/limits/features |
| `profiles` | yes | array | Quantization/runtime profiles; at least 1 |
| `deployment_availability` | yes | array | Hosted/Dedicated/Private + regions |
| `rollout` | yes | object | Eligibility and policy hints |
| `deprecation` | no | object | Timelines + replacements |
| `release_notes` | yes | object | Public/internal summaries + categories |
| `quality_signals` | no | array | Eval run IDs, guardrail outcomes |
| `security` | no | object | Security-impact flags and references |
| `created_at`, `updated_at` | yes | RFC3339 timestamp | Audit + ordering |
| `ownership` | yes | object | Owning teams + oncall |

## Compatibility semantics
Compatibility is defined as constraints across:
1) **Tokenizer contract** (`serving.tokenizer.*`)
2) **Runtime contract** (`serving.runtime.*`)
3) **Client contract** (minimum SDKs or API feature gates)

### Tokenizer
- `tokenizer.id` is a stable identifier (e.g., `sentencepiece:llama3` or `bpe:mistral`).
- `tokenizer.revision` MUST change when merges/vocab/config change.
- If tokenizer changes in a way that can affect typical prompts, it is at least a **MAJOR** model version bump.

### Runtime
- `runtime.engine` indicates the serving runtime family (e.g., `redwood-runtime`).
- `runtime.min_engine_version` expresses the minimal runtime version capable of serving the model with correctness.
- If a Private deployment cannot satisfy `min_engine_version`, the model version MUST NOT be marked available for Private.

### Feature compatibility
`serving.supported_features` enumerates supported feature contracts, e.g.:
- `streaming`
- `tool_calling`
- `structured_output_json`
- `logprobs`

If a customer pins a model version and a feature is listed as supported, removing it requires a **MAJOR** version bump.

## Deployment availability semantics
`deployment_availability` is an array of entries keyed by `deployment_mode`:
- `hosted`
- `dedicated`
- `private`

Each entry carries:
- Regions (or `*` for all)
- Availability stage: `preview`, `ga`, `limited`, `unavailable`
- Optional gating notes (e.g., “requires A100/H100 class GPUs”)

**Invariant:** A model version may be `ga` on Hosted but `preview` or `unavailable` on Private; release notes must not imply synchronized rollout across deployment modes.

## Rollout policy semantics
`rollout` captures safe rollout hints used by the policy engine. It does not replace per-account policies.

Key fields:
- `rollout.eligibility.canary` (boolean)
- `rollout.eligibility.ab_test` (boolean)
- `rollout.risk_level` (`low`, `medium`, `high`)
- `rollout.recommended_blast_radius_pct` (0–100)
- `rollout.rollback_guidance` (public-safe)

**Invariant:** `risk_level=high` MUST include `internal.rationale` and at least one `quality_signals` entry (or a documented exemption in `internal.exceptions`).

## Release notes fields
`release_notes` is required to ensure every promotion has customer-visible change visibility.

- `release_notes.public.summary`: 1–3 sentences, customer-safe.
- `release_notes.public.categories`: subset of:
  - `security`, `breaking`, `quality`, `performance`, `bugfix`, `deprecation`, `behavior-change`, `operational`
- `release_notes.public.severity`: `low`, `medium`, `high`
- `release_notes.public.customer_action_required`: `yes`/`no`

Internal complements:
- `release_notes.internal.summary`
- `release_notes.internal.links` (runbooks, dashboards, PRs)

## Validation rules (selected)
1) `model_id` must match: `^rw\.[a-z0-9]+(\.[a-z0-9-]+)+$`
2) `model_version` must match SemVer: `^\d+\.\d+\.\d+(-[0-9A-Za-z.-]+)?(\+[0-9A-Za-z.-]+)?$`
3) `profiles` must contain unique `profile_id` values.
4) Exactly one profile must have `is_default=true`.
5) If `status=deprecated`, then `deprecation` must be present and include `effective_date`.
6) If `deployment_mode=private` is present, `private.bundle_min_version` must be present.
7) If `release_notes.public.categories` contains `breaking`, then `release_notes.public.customer_action_required` must be `yes`.

## Examples
### Example A: Chat model (Hosted GA, Dedicated GA, Private Preview)
```json
{
  "schema_version": "1.0.0",
  "model_id": "rw.chat.llama3_1_70b_instruct",
  "model_version": "2.3.0",
  "display_name": "Llama 3.1 70B Instruct (Redwood)",
  "model_family": "chat",
  "variant": "instruct",
  "status": "active",
  "upstream": {
    "provider": "meta",
    "model_ref": "meta-llama/Llama-3.1-70B-Instruct",
    "revision": "hf:8b2c3a1",
    "license": "llama3",
    "source_url": "https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct"
  },
  "serving": {
    "tokenizer": {
      "id": "sentencepiece:llama3",
      "revision": "spm:v3.1.0"
    },
    "runtime": {
      "engine": "redwood-runtime",
      "min_engine_version": ">=0.28.0"
    },
    "limits": {
      "max_context_tokens": 131072,
      "max_output_tokens": 8192
    },
    "supported_features": [
      "streaming",
      "tool_calling",
      "structured_output_json"
    ]
  },
  "profiles": [
    {
      "profile_id": "fp16",
      "is_default": true,
      "precision": "fp16",
      "kv_cache_dtype": "fp16",
      "recommended_batching": {
        "continuous_batching": true,
        "max_batch_tokens": 8192
      }
    },
    {
      "profile_id": "int8",
      "is_default": false,
      "precision": "int8",
      "kv_cache_dtype": "fp16",
      "recommended_batching": {
        "continuous_batching": true,
        "max_batch_tokens": 12288
      }
    }
  ],
  "deployment_availability": [
    {
      "deployment_mode": "hosted",
      "stage": "ga",
      "regions": ["us-east-1", "us-west-2", "eu-west-1"]
    },
    {
      "deployment_mode": "dedicated",
      "stage": "ga",
      "regions": ["us-east-1", "eu-west-1"],
      "requirements": "A100-80GB or H100"
    },
    {
      "deployment_mode": "private",
      "stage": "preview",
      "regions": ["*"],
      "private": {
        "bundle_min_version": ">=2025.10.0"
      }
    }
  ],
  "rollout": {
    "eligibility": {"canary": true, "ab_test": true},
    "risk_level": "medium",
    "recommended_blast_radius_pct": 10,
    "rollback_guidance": "If you observe regressions, roll back to 2.2.x or pin to the prior default profile."
  },
  "release_notes": {
    "public": {
      "summary": "Improves long-context reliability and reduces tool-call formatting errors. No API changes.",
      "categories": ["quality", "bugfix", "performance"],
      "severity": "medium",
      "customer_action_required": "no",
      "links": ["https://docs.redwoodinference.com/release-notes/models/rw.chat.llama3_1_70b_instruct/2.3.0"]
    },
    "internal": {
      "summary": "Kernel selection update + constrained decoding fix for JSON schema mode; validated on nightly eval pack.",
      "links": ["dash://observability/model-change/rw.chat.llama3_1_70b_instruct/2.3.0", "pr://redwood/18441"]
    }
  },
  "quality_signals": [
    {
      "type": "eval_run",
      "id": "evalrun_01J9QK1T7MZ9",
      "baseline_model_version": "2.2.2",
      "summary_public": "No statistically significant regressions detected on Redwood chat reliability pack.",
      "guardrails": ["safety:pass", "pii:pass"]
    }
  ],
  "security": {
    "customer_impact": "none",
    "notes_internal": "No new dependencies; SBOM unchanged"
  },
  "ownership": {
    "owner_team": "Model Catalog",
    "oncall": "#oncall-model-catalog"
  },
  "created_at": "2025-10-15T21:04:11Z",
  "updated_at": "2025-10-18T02:33:09Z"
}
```

### Example B: Embedding model (Hosted GA only)
```json
{
  "schema_version": "1.0.0",
  "model_id": "rw.embedding.e5_large_v2",
  "model_version": "1.1.1",
  "display_name": "E5 Large v2 (Redwood)",
  "model_family": "embedding",
  "variant": "base",
  "status": "active",
  "upstream": {
    "provider": "microsoft",
    "model_ref": "intfloat/e5-large-v2",
    "revision": "hf:2a91f0b",
    "license": "mit",
    "source_url": "https://huggingface.co/intfloat/e5-large-v2"
  },
  "serving": {
    "tokenizer": {"id": "bpe:e5", "revision": "bpe:2024-11-19"},
    "runtime": {"engine": "redwood-runtime", "min_engine_version": ">=0.26.0"},
    "limits": {"max_context_tokens": 8192, "max_output_tokens": 0},
    "supported_features": ["batching"]
  },
  "profiles": [
    {"profile_id": "fp16", "is_default": true, "precision": "fp16"}
  ],
  "deployment_availability": [
    {"deployment_mode": "hosted", "stage": "ga", "regions": ["us-east-1", "eu-west-1"]}
  ],
  "rollout": {
    "eligibility": {"canary": true, "ab_test": false},
    "risk_level": "low",
    "recommended_blast_radius_pct": 25,
    "rollback_guidance": "Pin to 1.1.0 if you see embedding distribution drift in downstream ranking."
  },
  "release_notes": {
    "public": {
      "summary": "Fixes an edge-case padding issue affecting a small fraction of batch embedding requests.",
      "categories": ["bugfix"],
      "severity": "low",
      "customer_action_required": "no",
      "links": ["https://docs.redwoodinference.com/release-notes/models/rw.embedding.e5_large_v2/1.1.1"]
    },
    "internal": {
      "summary": "Tokenizer padding alignment fix; confirmed against retrieval regression suite.",
      "links": ["pr://redwood/18435"]
    }
  },
  "ownership": {"owner_team": "Model Catalog", "oncall": "#oncall-model-catalog"},
  "created_at": "2025-09-28T17:12:45Z",
  "updated_at": "2025-09-29T08:20:02Z"
}
```

## Operational notes
- **Promotion:** Model versions are promoted via Control Plane workflow; metadata must pass CI validation and redaction tests before promotion.
- **Schema hash:** Downstream systems should compute a canonical JSON hash (SHA-256 of normalized JSON) to link eval evidence and audit events. This hash is emitted in change events (see audit logging docs).
- **Redaction:** Any newly added internal fields must be accompanied by an explicit allowlist for customer-facing serialization.

## Appendix: JSON Schema (draft 2020-12)
See `json_schema` field in this document for the canonical schema blob used by validation tooling.
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://schemas.redwoodinference.com/model-metadata/v1/model-version-metadata.schema.json",
  "title": "Redwood Model Version Metadata (v1)",
  "type": "object",
  "additionalProperties": false,
  "required": [
    "schema_version",
    "model_id",
    "model_version",
    "display_name",
    "model_family",
    "variant",
    "status",
    "upstream",
    "serving",
    "profiles",
    "deployment_availability",
    "rollout",
    "release_notes",
    "ownership",
    "created_at",
    "updated_at"
  ],
  "properties": {
    "schema_version": { "$ref": "#/$defs/semver" },
    "model_id": {
      "type": "string",
      "pattern": "^rw\\.[a-z0-9]+(\\.[a-z0-9-]+)+$",
      "description": "Stable identifier. Immutable."
    },
    "model_version": {
      "$ref": "#/$defs/semver",
      "description": "Served version for pinning and change history. Immutable once promoted."
    },
    "display_name": { "type": "string", "minLength": 3, "maxLength": 120 },
    "model_family": {
      "type": "string",
      "enum": ["chat", "completion", "embedding", "rerank", "multimodal" ]
    },
    "variant": {
      "type": "string",
      "enum": ["base", "instruct", "chat", "coder", "vision", "reasoning"]
    },
    "status": {
      "type": "string",
      "enum": ["active", "deprecated", "disabled", "preview"]
    },
    "upstream": {
      "type": "object",
      "additionalProperties": false,
      "required": ["provider", "model_ref", "revision", "license"],
      "properties": {
        "provider": { "type": "string", "minLength": 2, "maxLength": 64 },
        "model_ref": { "type": "string", "minLength": 2, "maxLength": 200 },
        "revision": { "type": "string", "minLength": 1, "maxLength": 200 },
        "license": { "type": "string", "minLength": 1, "maxLength": 80 },
        "source_url": { "type": "string", "format": "uri" }
      }
    },
    "serving": {
      "type": "object",
      "additionalProperties": false,
      "required": ["tokenizer", "runtime", "limits", "supported_features"],
      "properties": {
        "tokenizer": {
          "type": "object",
          "additionalProperties": false,
          "required": ["id", "revision"],
          "properties": {
            "id": { "type": "string", "minLength": 3, "maxLength": 80 },
            "revision": { "type": "string", "minLength": 1, "maxLength": 80 }
          }
        },
        "runtime": {
          "type": "object",
          "additionalProperties": false,
          "required": ["engine", "min_engine_version"],
          "properties": {
            "engine": { "type": "string", "enum": ["redwood-runtime"] },
            "min_engine_version": { "type": "string", "minLength": 3, "maxLength": 40 }
          }
        },
        "limits": {
          "type": "object",
          "additionalProperties": false,
          "required": ["max_context_tokens", "max_output_tokens"],
          "properties": {
            "max_context_tokens": { "type": "integer", "minimum": 256, "maximum": 1048576 },
            "max_output_tokens": { "type": "integer", "minimum": 0, "maximum": 131072 }
          }
        },
        "supported_features": {
          "type": "array",
          "items": {
            "type": "string",
            "enum": [
              "streaming",
              "tool_calling",
              "structured_output_json",
              "logprobs",
              "batching"
            ]
          },
          "minItems": 0,
          "uniqueItems": true
        }
      }
    },
    "profiles": {
      "type": "array",
      "minItems": 1,
      "items": { "$ref": "#/$defs/profile" }
    },
    "deployment_availability": {
      "type": "array",
      "minItems": 1,
      "items": { "$ref": "#/$defs/deploymentAvailability" }
    },
    "rollout": {
      "type": "object",
      "additionalProperties": false,
      "required": ["eligibility", "risk_level", "recommended_blast_radius_pct", "rollback_guidance"],
      "properties": {
        "eligibility": {
          "type": "object",
          "additionalProperties": false,
          "required": ["canary", "ab_test"],
          "properties": {
            "canary": { "type": "boolean" },
            "ab_test": { "type": "boolean" }
          }
        },
        "risk_level": { "type": "string", "enum": ["low", "medium", "high"] },
        "recommended_blast_radius_pct": { "type": "integer", "minimum": 0, "maximum": 100 },
        "rollback_guidance": { "type": "string", "minLength": 10, "maxLength": 800 }
      }
    },
    "deprecation": {
      "type": "object",
      "additionalProperties": false,
      "required": ["state", "effective_date"],
      "properties": {
        "state": { "type": "string", "enum": ["announced", "in_effect", "completed"] },
        "effective_date": { "type": "string", "format": "date" },
        "replacement_model_id": { "type": "string" },
        "replacement_model_version": { "$ref": "#/$defs/semver" },
        "notes": { "type": "string", "maxLength": 1200 }
      }
    },
    "release_notes": {
      "type": "object",
      "additionalProperties": false,
      "required": ["public", "internal"],
      "properties": {
        "public": { "$ref": "#/$defs/releaseNotesPublic" },
        "internal": { "$ref": "#/$defs/releaseNotesInternal" }
      }
    },
    "quality_signals": {
      "type": "array",
      "items": { "$ref": "#/$defs/qualitySignal" }
    },
    "security": {
      "type": "object",
      "additionalProperties": false,
      "properties": {
        "customer_impact": { "type": "string", "enum": ["none", "low", "medium", "high"] },
        "notes_internal": { "type": "string", "maxLength": 2000 }
      }
    },
    "ownership": {
      "type": "object",
      "additionalProperties": false,
      "required": ["owner_team", "oncall"],
      "properties": {
        "owner_team": { "type": "string", "minLength": 2, "maxLength": 80 },
        "oncall": { "type": "string", "minLength": 2, "maxLength": 80 }
      }
    },
    "created_at": { "$ref": "#/$defs/rfc3339" },
    "updated_at": { "$ref": "#/$defs/rfc3339" }
  },
  "allOf": [
    {
      "description": "If status is deprecated then deprecation must be present.",
      "if": { "properties": { "status": { "const": "deprecated" } } },
      "then": { "required": ["deprecation"] }
    }
  ],
  "$defs": {
    "semver": {
      "type": "string",
      "pattern": "^\\d+\\.\\d+\\.\\d+(-[0-9A-Za-z.-]+)?(\\+[0-9A-Za-z.-]+)?$"
    },
    "rfc3339": { "type": "string", "format": "date-time" },
    "profile": {
      "type": "object",
      "additionalProperties": false,
      "required": ["profile_id", "is_default", "precision"],
      "properties": {
        "profile_id": { "type": "string", "pattern": "^[a-z0-9][a-z0-9_-]{1,32}$" },
        "is_default": { "type": "boolean" },
        "precision": { "type": "string", "enum": ["fp16", "bf16", "int8", "int4"] },
        "kv_cache_dtype": { "type": "string", "enum": ["fp16", "bf16", "int8" ] },
        "recommended_batching": {
          "type": "object",
          "additionalProperties": false,
          "properties": {
            "continuous_batching": { "type": "boolean" },
            "max_batch_tokens": { "type": "integer", "minimum": 256, "maximum": 131072 }
          }
        }
      }
    },
    "deploymentAvailability": {
      "type": "object",
      "additionalProperties": false,
      "required": ["deployment_mode", "stage", "regions"],
      "properties": {
        "deployment_mode": { "type": "string", "enum": ["hosted", "dedicated", "private"] },
        "stage": { "type": "string", "enum": ["preview", "limited", "ga", "unavailable"] },
        "regions": {
          "type": "array",
          "minItems": 1,
          "items": { "type": "string" },
          "uniqueItems": true
        },
        "requirements": { "type": "string", "maxLength": 400 },
        "private": {
          "type": "object",
          "additionalProperties": false,
          "properties": {
            "bundle_min_version": { "type": "string", "minLength": 3, "maxLength": 40 }
          }
        }
      },
      "allOf": [
        {
          "if": { "properties": { "deployment_mode": { "const": "private" } } },
          "then": { "required": ["private"] }
        }
      ]
    },
    "releaseNotesPublic": {
      "type": "object",
      "additionalProperties": false,
      "required": ["summary", "categories", "severity", "customer_action_required", "links"],
      "properties": {
        "summary": { "type": "string", "minLength": 20, "maxLength": 600 },
        "categories": {
          "type": "array",
          "minItems": 1,
          "uniqueItems": true,
          "items": {
            "type": "string",
            "enum": ["security", "breaking", "quality", "performance", "bugfix", "deprecation", "behavior-change", "operational"]
          }
        },
        "severity": { "type": "string", "enum": ["low", "medium", "high"] },
        "customer_action_required": { "type": "string", "enum": ["yes", "no"] },
        "links": {
          "type": "array",
          "minItems": 0,
          "items": { "type": "string", "maxLength": 400 }
        }
      },
      "allOf": [
        {
          "if": { "properties": { "categories": { "contains": { "const": "breaking" } } } },
          "then": { "properties": { "customer_action_required": { "const": "yes" } } }
        }
      ]
    },
    "releaseNotesInternal": {
      "type": "object",
      "additionalProperties": false,
      "required": ["summary", "links"],
      "properties": {
        "summary": { "type": "string", "minLength": 10, "maxLength": 2000 },
        "links": {
          "type": "array",
          "items": { "type": "string", "maxLength": 400 }
        }
      }
    },
    "qualitySignal": {
      "type": "object",
      "additionalProperties": false,
      "required": ["type", "id"],
      "properties": {
        "type": { "type": "string", "enum": ["eval_run", "guardrail", "benchmark"] },
        "id": { "type": "string", "minLength": 6, "maxLength": 80 },
        "baseline_model_version": { "$ref": "#/$defs/semver" },
        "summary_public": { "type": "string", "maxLength": 600 },
        "guardrails": {
          "type": "array",
          "items": { "type": "string", "maxLength": 80 },
          "uniqueItems": true
        }
      }
    }
  }
}
