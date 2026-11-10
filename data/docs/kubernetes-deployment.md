# Deploying Lumen on Kubernetes

This guide runs a replicated Lumen cluster on Kubernetes using the official Helm chart.

## Helm Chart

Add the chart repository and install a release into its own namespace:

```bash
helm repo add lumen https://charts.lumen.dev
helm repo update
# Install three replicas with persistent storage
helm install lumen lumen/lumen \
  --namespace lumen --create-namespace \
  --set replicaCount=3 \
  --set persistence.size=100Gi
```

The chart deploys a StatefulSet so every replica keeps a stable network identity and its
own PersistentVolumeClaim. Do not switch it to a Deployment: replicas would lose their data
on rescheduling.

## Scaling

### Horizontal Scaling

Collections are split into shards that are spread across replicas. Adding replicas
increases query throughput, and Lumen rebalances shards automatically after a new pod
becomes ready. Rebalancing copies data over the network, so scale out before peak traffic
rather than during it.

### Autoscaling

A HorizontalPodAutoscaler can scale replicas on CPU utilisation. Scale-in should be
conservative because removing a replica forces its shards to move.

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: lumen
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: StatefulSet
    name: lumen
  minReplicas: 3
  maxReplicas: 12
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 900
```

## Health Checks

Lumen exposes `/healthz` for liveness and `/readyz` for readiness. A pod reports ready only
after it has loaded its shards, so traffic is not routed to a replica that would answer
with empty results. Give the readiness probe a generous `initialDelaySeconds` on nodes with
large collections, because loading HNSW graphs from disk can take several minutes.

```yaml
livenessProbe:
  httpGet: {path: /healthz, port: 7700}
  periodSeconds: 10
readinessProbe:
  httpGet: {path: /readyz, port: 7700}
  initialDelaySeconds: 30
  periodSeconds: 5
```

## Observability

### Metrics

Prometheus metrics are served on port 9090 at `/metrics`. The most useful series are
`lumen_query_duration_seconds` (a histogram of search latency), `lumen_index_vectors_total`
and `lumen_memory_bytes`. Alert when the p99 of query duration exceeds your latency budget
for more than five minutes.

### Logging

Logs are written to stdout as JSON lines, one event per line, ready for Fluent Bit or Vector.
Set `LUMEN_LOG_LEVEL=debug` temporarily to trace slow queries; debug logging is verbose and
should not be left enabled in production.

## Upgrades

Upgrade one minor version at a time. The StatefulSet uses a `RollingUpdate` strategy with
`partition` support, so you can upgrade a single canary pod first:

```bash
# Upgrade only the highest-ordinal pod first
helm upgrade lumen lumen/lumen --namespace lumen \
  --set image.tag=2.5.0 --set updateStrategy.rollingUpdate.partition=2
```

Once the canary is healthy, lower the partition to 0 to roll out to the remaining pods.
Take a snapshot of every collection before upgrading, since downgrades are not supported.
