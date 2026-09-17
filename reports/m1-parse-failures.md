# Parse failures held for later

Generated 2026-09-17 02:48 UTC from the live corpus (62,070 papers ingested; the run was still in progress). Regenerate with:

```sql
select v.name, p.publication_year, pv.parse_status, p.title
from paper_versions pv join papers p on p.id = pv.paper_id
left join venues v on v.id = p.venue_id where pv.parse_status <> 'parsed';
```

Every paper listed here **kept its source PDF**: a venue-year's sweep deletes a
PDF only once its text is stored, so each of these can be retried without
re-downloading. Their abstracts are already searchable; only full text is missing.

Parse outcomes across the corpus: `corrupt` 3, `empty_text` 7, `missing` 1, `not_pdf` 1, `oversized` 18, `parsed` 62,033, `pending` 7

| Venue | Year | Status | Title | Checksum |
|---|---|---|---|---|
| AAAI | 2023 | `corrupt` | Deep Spiking Neural Networks with High Representation Simi | `3f9666887369` |
| AAAI | 2023 | `corrupt` | Back to the Future: Toward a Hybrid Architecture for Ad Ho | `3f9666887369` |
| IJCAI | 2023 | `corrupt` | Multi-Task Learning via Time-Aware Neural ODE | `5db7772afd3c` |
| ECCV | 2024 | `empty_text` | View-Consistent Hierarchical 3D Segmentation Using Ultrame | `8ed97185edfc` |
| ECCV | 2024 | `empty_text` | Reinforcement Learning via Auxillary Task Distillation | `fe71eb33fc19` |
| ICLR | 2023 | `empty_text` | Timing is Everything: Learning to Act Selectively with Cos | `33f41f0b7568` |
| ICLR | 2023 | `empty_text` | Information-Theoretic Diffusion | `e0789572accd` |
| ICLR | 2024 | `empty_text` | Efficient Heterogeneous Meta-Learning via Channel Shufflin | `ddae1d69e779` |
| NeurIPS | 2023 | `empty_text` | Sharp Calibrated Gaussian Processes | `c47e3d7db0f2` |
| NeurIPS | 2023 | `empty_text` | ReDS: Offline RL With Heteroskedastic Datasets via Support | `cfd78bc586ae` |
| CVPR | 2025 | `missing` | StarVector: Generating Scalable Vector Graphics Code from  | `37d0839c4db2` |
| JMLR | 2023 | `not_pdf` | SQLFlow: An Extensible Toolkit Integrating DB and AI | `e3b0c44298fc` |
| CVPR | 2023 | `oversized` | Rethinking Federated Learning With Domain Shift: A Prototy | `ba5ff79c8514` |
| CVPR | 2023 | `oversized` | MOT: Masked Optimal Transport for Partial Domain Adaptatio | `a5c153239784` |
| CVPR | 2023 | `oversized` | Bayesian Posterior Approximation With Stochastic Ensembles | `26c04f8c683a` |
| CVPR | 2024 | `oversized` | CaKDP: Category-aware Knowledge Distillation and Pruning F | `866d22a72ac6` |
| CVPR | 2025 | `oversized` | Weakly Supervised Contrastive Adversarial Training for Lea | `e4f8371091d3` |
| ECCV | 2024 | `oversized` | PointRegGPT: Boosting 3D Point Cloud Registration using Ge | `8eb92df27533` |
| ECCV | 2024 | `oversized` | Towards Image Ambient Lighting Normalization | `42e588a62670` |
| ECCV | 2024 | `oversized` | Improving Diffusion Models for Authentic Virtual Try-on in | `403b586fe202` |
| ECCV | 2024 | `oversized` | COIN-Matting: Confounder Intervention for Image Matting | `082a2cae059e` |
| ECCV | 2024 | `oversized` | MMVR: Millimeter-wave Multi-View Radar Dataset and Benchma | `0e77f026c237` |
| ECCV | 2024 | `oversized` | Un-EVIMO: Unsupervised Event-based Independent Motion Segm | `ccf1c94f19c8` |
| ECCV | 2024 | `oversized` | Topo4D: Topology-Preserving Gaussian Splatting for High-Fi | `5cb6f2521c91` |
| ECCV | 2024 | `oversized` | Bridging the Gap: Studio-like Avatar Creation from a Monoc | `4bc7c72f24f3` |
| ICCV | 2023 | `oversized` | Panoramas from Photons | `d2b961433b1c` |
| ICCV | 2025 | `oversized` | FakeRadar: Probing Forgery Outliers to Detect Unknown Deep | `c9577476d9d5` |
| ICCV | 2025 | `oversized` | FRET: Feature Redundancy Elimination for Test Time Adaptat | `b6a035ec674a` |
| WACV | 2024 | `oversized` | Real-Time Polyp Detection in Colonoscopy Using Lightweight | `dc0ab42db3ce` |
| WACV | 2026 | `oversized` | Similarity-aware Probabilistic Embeddings Modeling for Vid | `7a3bedff56e7` |
| CVPR | 2024 | `pending` | DePT: Decoupled Prompt Tuning | `e1e26bb4d13c` |
| CVPR | 2024 | `pending` | SportsSloMo: A New Benchmark and Baselines for Human-centr | `413448b3c42a` |
| CVPR | 2024 | `pending` | GenHowTo: Learning to Generate Actions and State Transform | `3975214ed00b` |
| Interspeech | 2025 | `pending` | From Talking and Listening Devices to Intelligent Communic | `7e44034bb6d9` |
| Interspeech | 2025 | `pending` | From Speech Science to Language Transparence | `45a11a065a39` |
| Interspeech | 2025 | `pending` | Using and comprehending language in face-to-face conversat | `f86ac0551773` |
| Interspeech | 2025 | `pending` | Speech Kinematic Analysis from Acoustics: Scientific, Clin | `3234d2590c58` |

## What each status needs

- **`oversized`** — above `parser.max_pdf_bytes` (50 MB). Raising the cap and
  re-parsing is enough; the files are still on disk.
- **`empty_text`** — the PDF parsed but yielded no extractable text, which usually
  means a scanned or image-only document. A different adapter, or OCR, is the only
  route to full text; the abstract still works.
- **`corrupt`** — pypdf could not read the structure. Worth inspecting by hand.
- **`not_pdf`** — the mirrored file is not a PDF. SQLFlow's is 0 bytes, so the shard
  itself lacks that paper.
- **`pending`** — the parse never ran because the download failed. These were
  requeued on 2026-09-17 after the download allowlist was widened.
