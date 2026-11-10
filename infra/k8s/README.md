# Kubernetes manifests (reference)

These manifests describe how the corpus platform is meant to run on a cluster. **They have
not been applied to a cluster from this repository** — they were written against the
Kubernetes API reference and checked only for YAML validity. Treat them as a starting point:
review resource requests, storage classes, ingress and network policies for your environment.

| file | what it creates |
|---|---|
| `namespace-config.yaml` | namespace, non-secret configuration, the names of the secrets that must exist |
| `api.yaml` | online API deployment, service, HPA, PodDisruptionBudget |
| `workers.yaml` | change-event indexer and batch workers (separate from the API on purpose) |
| `cronjobs.yaml` | scheduled sync, data-quality, consistency, drift and evaluation jobs |
| `vllm.yaml` | GPU model server (vLLM, OpenAI-compatible) with node selection and tolerations |
| `networkpolicy.yaml` | default-deny ingress; only the API accepts traffic, only it and workers reach the model server |

PostgreSQL, Elasticsearch/OpenSearch, Redis, Kafka and object storage are expected as
managed services or operator-managed clusters; they are not defined here.

Secrets are referenced by name (`inspect-rag-secrets`) and never stored in this repository:

    kubectl -n inspect-rag create secret generic inspect-rag-secrets \
      --from-literal=database-url=... --from-literal=elasticsearch-api-key=... \
      --from-literal=api-keys-json=... --from-literal=llm-api-key=...

Rollout order: database migration job → index build (`corpus index build`) → API → workers.
Rollback: `kubectl rollout undo deployment/corpus-api`; index rollback is `corpus index rollback`.
