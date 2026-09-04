---
name: wire-parity-migration
description: Replace a service implementation with a drop-in (new framework, new language, new server) while unmodified clients keep working. Use when migrating an HTTP/API layer (e.g. stdlib server to FastAPI), when the exit test is "existing frontend/clients run unmodified", or when adding typed contracts to a legacy wire protocol.
---

# Wire-parity migration (drop-in server replacement)

Reference implementation: `sohbench.api` (FastAPI replacing the stdlib
server; V1 frontend ran unmodified — B3).

## 1. Inventory the contract from BOTH ends

- **Server side**: every route (method, path, params), status codes per
  outcome, the exact error-body shape, exception→status mapping (it may
  differ per HTTP method!), CORS/auth header behavior, gating flags, and
  serialization flags (`ensure_ascii`, `allow_nan`, separators).
- **Client side**: every call the real clients make, which response fields
  they actually read, how they detect errors (status? an `error` field?
  regex on messages? — optimistic-concurrency retries keyed on message
  text are load-bearing), and shape invariants (echoed IDs, required
  top-level fields). Parity targets what clients OBSERVE, not what the old
  code did internally.

## 2. Reproduce semantics, not framework defaults

Frameworks fight you exactly where parity lives: default validation errors
(422 vs legacy 400), response-model filtering silently dropping fields,
encoder walks over huge payloads, lowercased header names, different JSON
whitespace. Rules that worked here:

- return raw payloads directly (bypass response-model filtering/encoding);
  attach typed models as documentation only, `extra=allow`;
- keep request models lenient (all-optional) so the service layer stays
  the validating authority and its error MESSAGES reach clients unchanged;
- centralize gates + CORS + exception mapping in one middleware that
  mirrors the legacy ordering;
- treat header-name case and JSON whitespace as non-contractual (clients
  parse JSON; HTTP headers are case-insensitive) — fix TESTS that pin
  them, not the server.

## 3. The parity lever: run the OLD test suite against BOTH

Parameterize the existing API test suite's server factory over
[legacy, new] with an autouse fixture, over a real socket. Every behavior
the suite pins is then enforced on the replacement for free. Divergences
the suite finds are the actual migration work list.

## 4. Typed contracts + drift gates

Generate client types from the new server's schema (OpenAPI → .d.ts or
equivalent) into the client tree, and add two tests: generated files match
a fresh generation (drift), and the schema covers every endpoint the
clients call (coverage). Strip framework-documented behaviors the server
never exhibits (e.g. auto-422s) from the published schema.

## 5. Exit test

Run the real client against the new server and exercise a full workflow,
not just reads. Keep the legacy implementation in-tree until the parity
suite has run green in CI on both for a while.


## Review note (2026-09-03, after the kernel extraction)

Module map since ADR-001 Wave 2: label safety, protocols (now including `forward`), the runner, report statistics, adaptation, fold-local feature pipelines and the API facade live in `sohkernel` (`sohkernel.api`, `sohkernel.feature_safety`, `sohkernel.inference_gate`, `sohkernel.pipelines`, `sohkernel.records`, `sohkernel.datasheets`); the candidate model, search engine, sandbox, evaluator client and claim flow live in `sohagent` (`sohagent.candidates`, `sohagent.search`, `sohagent.sandbox`, `sohagent.evaluator`, `sohagent.claims`). `sohbench` is the frozen workbench and only re-exports the kernel. Any `sohbench.<module>` reference above is historical.
