# Mappatura CLI → Byte API

## Modalità corpus

| CLI originale | Campo multipart API |
|---|---|
| `corpus` | endpoint `/api/v1/byte/corpus` |
| `--file` | `file` |
| `--tier` | `tier` |
| `--organization-id` | `organization_id` |
| `--user-id` | `user_id` |
| `--ontology-code` | `ontology_code` |
| `--ontology-label` | `ontology_label` |
| `--area` | `area` |
| `--subarea` | `subarea` |
| `--classification` | `classification` |
| `--pipeline-version` | `pipeline_version` |
| `--corpus-version` | `corpus_version` |
| `--embedding-model` | `embedding_model` |
| `--mime-type` | `mime_type` |

## Modalità evidence

| CLI originale | Campo multipart API |
|---|---|
| `evidence` | endpoint `/api/v1/byte/evidence` |
| `--file` | `file` |
| `--organization-id` | `organization_id` |
| `--user-id` | `user_id` |
| `--assessment-id` | `assessment_id` |
| `--response-id` | `response_id` |
| `--encryption-required` | `encryption_required=true` |
| `--no-encryption-required` | `encryption_required=false` |
| `--mime-type` | `mime_type` |

## Differenza tecnica del parametro file

Nella CLI `--file` era un percorso locale letto dal processo Python.

Nell'API il campo multipart `file` contiene i byte effettivi del documento. Il server non deve conoscere né poter accedere al percorso del client.
