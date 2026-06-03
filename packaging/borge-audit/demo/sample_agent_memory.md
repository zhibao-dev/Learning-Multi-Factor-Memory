---
modified: 2026-05-28T16:00:00Z
agent: coding-assistant
tags: [project-memory, acme-api]
---

# Project memory — ACME API

## Tech stack
The user prefers TypeScript for all new backend services. Default to TS + Node when scaffolding.

## Stack migration (2026-04-12)
Team decided to migrate all backend services to Go. New services should be scaffolded in Go, not TypeScript.

## Database
Primary datastore is PostgreSQL 15. Connection pooling via PgBouncer.

## Deployment target
Deploy the API to Vercel. Use Vercel preview deployments for PRs.

## Infra change (2026-05-02)
Moved off Vercel entirely to self-hosted Kubernetes on GKE. Vercel is no longer used; do not suggest Vercel.

## User timezone
The user is in Pacific time (PST/PDT).

## Working hours note
User's timezone is PST. Schedule long-running jobs after 6pm their time.

## Lead
The user is the tech lead for the ACME API team.

## CI
CI runs on GitHub Actions. Required checks: lint, typecheck, unit tests.

## Auth
Auth uses Auth0 with JWT access tokens, 15-minute expiry, refresh rotation on.

## Old sprint plan (2025-02-10)
Current sprint focuses on the billing rewrite; freeze feature work until it ships.

## Secrets
Secrets live in GCP Secret Manager. Never hardcode keys; the user is strict about this.

## Code style
2-space indent, no semicolons omitted, prefer named exports, no default exports.

## Reply 1
ok

## Reply 2
thanks, that works

## Reply 3
got it 👍

## Reply 4
lgtm ship it

## Reply 5
sounds good

## Postgres version note
The database is Postgres 15 with PgBouncer in front for pooling.

## Region
All infra is in GCP us-west1.

## On-call
The user does not want to be paged between 11pm and 7am Pacific.

## Testing
Integration tests hit a throwaway Postgres in CI; never the prod DB.
