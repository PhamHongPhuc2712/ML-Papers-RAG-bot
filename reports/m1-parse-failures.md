# Parse failures held for later

Generated 2026-09-16 09:27 UTC from the live corpus. The run is still
in progress, so this is a snapshot — the authoritative list is always:

```sql
select v.name, p.publication_year, pv.parse_status, p.title
from paper_versions pv join papers p on p.id = pv.paper_id
left join venues v on v.id = p.venue_id where pv.parse_status <> 'parsed';
```

Every paper listed here **kept its source PDF**: the venue sweep deletes a PDF
only once its text is stored, so each of these can be retried without
re-downloading. Their abstracts are already searchable; only full text is missing.

Parse outcomes so far: `corrupt` 1, `not_pdf` 1, `oversized` 2, `parsed` 7448

| Venue | Year | Status | Title | Checksum |
|---|---|---|---|---|
| IJCAI | 2023 | `corrupt` | Multi-Task Learning via Time-Aware Neural ODE | `5db7772afd3c` |
| JMLR | 2023 | `not_pdf` | SQLFlow: An Extensible Toolkit Integrating DB and AI | `e3b0c44298fc` |
| WACV | 2024 | `oversized` | Real-Time Polyp Detection in Colonoscopy Using Lightweight T | `dc0ab42db3ce` |
| WACV | 2026 | `oversized` | Similarity-aware Probabilistic Embeddings Modeling for Video | `7a3bedff56e7` |

## What each status needs

- **`oversized`** — the PDF is above `parser.max_pdf_bytes` (50 MB). Raising the cap
  and re-parsing is enough; the file is still on disk.
- **`corrupt`** — pypdf could not read the structure. Worth inspecting by hand before
  deciding whether the mirror copy is damaged or the parser needs work.
- **`not_pdf`** — the mirrored file is not a PDF at all. SQLFlow's is 0 bytes, so the
  shard itself is missing that paper; only the upstream publisher can supply it.
