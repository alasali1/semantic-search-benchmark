# Lumen HTTP API Reference

All endpoints are served under `/v1` and accept and return JSON.

## Authentication

### API Keys

Every request must send an API key in the `Authorization` header using the Bearer scheme.
Keys are created in the admin console and are scoped to a project. A key can be read-only
or read-write; read-only keys receive HTTP 403 on any write endpoint.

```http
GET /v1/collections HTTP/1.1
Host: lumen.example.com
Authorization: Bearer lmn_live_3f9a...
```

Never embed keys in client-side code. Store them in a secrets manager and inject them
through environment variables.

### Key Rotation

Each project can hold two active keys at once so credentials can be rotated without
downtime: create a new key, deploy it to all clients, then revoke the old one. Revoked keys
stop working within 60 seconds across all nodes.

## Rate Limiting

Requests are limited per API key using a token bucket. The default allowance is 50 queries
per second with bursts of up to 100. When the limit is exceeded the server answers with
HTTP 429 and a `Retry-After` header giving the number of seconds to wait.

Every response includes `X-RateLimit-Limit`, `X-RateLimit-Remaining` and `X-RateLimit-Reset`
headers. Clients should back off exponentially with jitter instead of retrying immediately.

## Querying

### Search Endpoint

`POST /v1/collections/{name}/query` returns the nearest neighbours of a query vector.

```json
{
  "vector": [0.12, -0.03, 0.88],
  "top_k": 10,
  "filter": {"lang": "en"},
  "include": ["metadata", "distance"]
}
```

`top_k` may be at most 1000. The optional `filter` restricts results to points whose
metadata matches all given key-value pairs; filtering is applied during graph traversal,
so very selective filters can reduce recall.

### Pagination

List endpoints return at most `limit` items (default 100, maximum 1000) together with a
`next_cursor` value. Pass that cursor back as the `cursor` parameter to fetch the next page.
Cursors are opaque, expire after 10 minutes, and must not be parsed by clients.

## Errors

Errors use standard HTTP status codes and a JSON body with a machine-readable `code` and a
human-readable `message`.

| Status | Code                 | Meaning                                      |
|--------|----------------------|----------------------------------------------|
| 400    | `invalid_request`    | Malformed JSON or a missing required field   |
| 401    | `unauthenticated`    | Missing or invalid API key                   |
| 404    | `not_found`          | Collection or point does not exist           |
| 409    | `dimension_mismatch` | Vector length differs from collection schema |
| 429    | `rate_limited`       | Too many requests, see Retry-After           |

## Webhooks

### Event Types

Lumen can notify your services about `collection.created`, `snapshot.completed` and
`index.rebuild_finished` events. Webhook endpoints are registered per project.

### Verifying Signatures

Each webhook request carries an `X-Lumen-Signature` header containing an HMAC-SHA256 of the
raw request body, keyed with the endpoint's signing secret. Compute the HMAC yourself and
compare it in constant time before trusting the payload.

```python
import hashlib
import hmac

def is_valid(body: bytes, signature: str, secret: str) -> bool:
    # Constant-time comparison avoids timing attacks
    expected = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)
```

Reject requests whose `X-Lumen-Timestamp` is more than five minutes old to prevent replays.
