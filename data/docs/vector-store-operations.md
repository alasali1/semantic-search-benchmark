# Lumen Vector Store Operations Guide

Lumen is a self-hosted vector database for similarity search over embeddings. This guide
covers installing, configuring, backing up and troubleshooting a single Lumen node.

## Installation

Lumen ships as a static binary for Linux and macOS and as an official container image.
A node needs at least 2 CPU cores and 4 GiB of RAM; production nodes should have enough
memory to hold every HNSW graph they serve.

### Installing on Linux

Download the release archive, verify its checksum and place the binary on your `PATH`:

```bash
# Download and verify the release
curl -LO https://downloads.lumen.dev/lumen-2.4.1-linux-amd64.tar.gz
sha256sum -c lumen-2.4.1-linux-amd64.tar.gz.sha256

# Install the binary system-wide
tar -xzf lumen-2.4.1-linux-amd64.tar.gz
sudo install -m 0755 lumen /usr/local/bin/lumen
```

Create a dedicated `lumen` system user and a data directory owned by it. Running the
server as root is not supported and the process refuses to start if `uid == 0`.

### Running with Docker

The container image exposes the HTTP API on port 7700 and stores data under `/var/lib/lumen`.
Mount a volume there, otherwise all collections are lost when the container is removed.

```bash
docker run -d --name lumen \
  -p 7700:7700 \
  -v lumen-data:/var/lib/lumen \
  -e LUMEN_API_KEY=change-me \
  lumen/lumen:2.4.1
```

## Configuration

Lumen reads `lumen.toml` from the working directory, or the path given with `--config`.
Every setting can also be overridden with an environment variable prefixed by `LUMEN_`,
for example `LUMEN_SERVER__PORT=8800`.

### Index Parameters

Each collection builds an HNSW graph. Two parameters control the trade-off between recall,
memory and build time:

- `m` - the number of neighbours each node keeps. Higher values improve recall and
  increase memory roughly linearly. The default of 16 suits most embedding models.
- `ef_construction` - the size of the candidate list while inserting. Raising it from the
  default 128 to 256 improves graph quality at the cost of slower ingestion.

At query time, `ef_search` sets how many candidates are explored. It must be at least the
number of results requested; a value between 64 and 200 is typical.

```toml
[collections.defaults.hnsw]
m = 16
ef_construction = 128
ef_search = 96
distance = "cosine"   # one of: cosine, dot, l2
```

### Memory Limits

`max_memory_mb` caps the memory used by vector data and graphs. When the cap is reached,
Lumen evicts the least recently queried collections to disk and reloads them on demand,
which adds latency to the first query after eviction. Set the limit to about 75% of the
container or host memory so the operating system page cache has room left.

## Backups

### Snapshots

A snapshot is a consistent, point-in-time copy of a collection. Snapshots are taken online
without blocking writes: Lumen freezes the current segment list and copies immutable files.

```bash
# Create a snapshot of the "articles" collection
lumen snapshot create --collection articles --dest s3://backups/lumen/
```

Schedule snapshots with cron or your orchestrator. Keep at least seven daily snapshots and
test restoring one every month.

### Restoring a Collection

Restoring replaces a collection with the contents of a snapshot. The target collection is
taken offline during the restore, and queries against it return HTTP 503 until it finishes.

```bash
lumen snapshot restore --collection articles \
  --source s3://backups/lumen/articles-2026-10-30.snap
```

Restoring into a node running an older minor version is not supported.

## Troubleshooting

### High Query Latency

Slow queries usually come from one of three causes. First, `ef_search` may be set much
higher than needed; halving it often halves latency with little recall loss. Second, the
collection may have been evicted because of the memory limit, so the first query reloads it
from disk. Third, heavy concurrent ingestion competes for CPU with search; schedule bulk
imports outside peak hours or use a dedicated ingest node.

### Out of Memory Errors

If the process is killed by the kernel OOM killer, lower `max_memory_mb`, reduce `m` for
large collections, or move to a node with more RAM. Scalar quantization (`quantization =
"int8"`) reduces vector memory by roughly 4x with a small recall cost.
