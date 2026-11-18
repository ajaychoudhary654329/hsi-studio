# Software Requirements Specification (SRS)

## AI-Based Hyperspectral Image Classification and Object Detection
### MERN Product Layer + Python FastAPI ML Microservice

| Field | Value |
|---|---|
| Document type | Software Requirements Specification (IEEE 29148 / 830 style) |
| Source document | *AI-Based Hyperspectral Image Classification and Object Detection – System Design Document (MERN + Python ML Service)* |
| Version | 1.0 |
| Status | Baseline derived from the System Design Document (SDD) |
| Prototype codename | Prism / "HSI Studio" (frontend prototype) |

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Overall Description](#2-overall-description)
3. [System Architecture Overview](#3-system-architecture-overview)
4. [Specific Requirements – Functional](#4-functional-requirements)
5. [User Interface Requirements](#5-user-interface-requirements)
6. [External Interface Requirements](#6-external-interface-requirements)
7. [Data Requirements](#7-data-requirements)
8. [ML Service Requirements](#8-ml-service-requirements)
9. [Object Detection and Verification Requirements](#9-object-detection-and-verification-requirements)
10. [Non-Functional Requirements](#10-non-functional-requirements)
11. [Security Requirements](#11-security-requirements)
12. [Hardware and Software Requirements](#12-hardware-and-software-requirements)
13. [End-to-End Workflow](#13-end-to-end-workflow)
14. [DevOps, CI/CD and Deployment Requirements](#14-devops-cicd-and-deployment-requirements)
15. [Feature List](#15-feature-list)
16. [Project Schedule and Build Order](#16-project-schedule-and-build-order)
17. [Risks, Limitations and Mitigations](#17-risks-limitations-and-mitigations)
18. [Implementation Status and Remaining Work](#18-implementation-status-and-remaining-work)
19. [Assumptions, Dependencies and Constraints](#19-assumptions-dependencies-and-constraints)
20. [Requirements Traceability Matrix](#20-requirements-traceability-matrix)
21. [References](#21-references)
22. [Appendices](#22-appendices)

---

## 1. Introduction

### 1.1 Purpose
This SRS defines the functional, non-functional, interface, data and operational requirements of a web platform that **prepares hyperspectral imaging (HSI) data, learns spectral-spatial features, classifies pixels or regions, and detects objects where object-level annotations exist**. It is the contract between stakeholders (students, supervisor/faculty, evaluators) and the development team, and is derived entirely from the project's System Design Document.

### 1.2 Background
Hyperspectral imaging captures many narrow, contiguous spectral bands per pixel, giving spectral signatures that separate materials that look alike in RGB. HSI data is high-dimensional (hundreds of bands), redundant, noisy and spatially variable, and labelled samples are limited, which raises overfitting risk.

### 1.3 Scope
The system is a **MERN product layer plus a Python FastAPI ML microservice**:

- **MongoDB, Express, React, Node.js** – UI, authentication, job orchestration, results, experiment history.
- **Python (FastAPI)** – hyperspectral I/O, preprocessing, model training, inference, evaluation.

**Design decision (D-01):** The project synopsis proposes Python for modelling (scikit-learn, PyTorch/TensorFlow). Node.js cannot train these models efficiently, hence the Python ML microservice.

**Scope rules carried over from the synopsis:**

| ID | Rule |
|---|---|
| SC-1 | Dataset, classes, architecture and compute are finalised after data availability and supervisor guidance. |
| SC-2 | Object detection is **feature-flagged**: enabled only when a dataset has object-level annotations; otherwise it is reported as future work, **never as unsupported results**. |
| SC-3 | Results must be **reproducible, comparable and measured on held-out data**. |
| SC-4 | Results on one dataset must **not** be assumed to generalise across sensors, locations or conditions. |

### 1.4 Objectives
| ID | Objective |
|---|---|
| OBJ-1 | Study HSI and select a suitable public or approved dataset. |
| OBJ-2 | Inspect dimensions, bands, labels, class distribution and quality. |
| OBJ-3 | Apply preprocessing: normalization, noise handling, dimensionality reduction, band selection. |
| OBJ-4 | Build a baseline (SVM) and compare it with a deep model (2D/3D-CNN). |
| OBJ-5 | Explore spectral-spatial learning. |
| OBJ-6 | Implement object detection only if annotations exist. |
| OBJ-7 | Evaluate with suitable metrics and analyse errors. |
| OBJ-8 | Document workflow, settings, limitations and future work. |

### 1.5 Definitions, Acronyms and Abbreviations
| Term | Meaning |
|---|---|
| HSI | Hyperspectral Imaging |
| Cube | 3-D hyperspectral data array (height × width × bands) |
| OA / AA / Kappa | Overall Accuracy / Average (per-class) Accuracy / Cohen's Kappa |
| P / R / F1 | Precision / Recall / F1-score |
| PCA | Principal Component Analysis |
| SVM | Support Vector Machine |
| CNN (2D/3D) | Convolutional Neural Network |
| IoU | Intersection over Union |
| AP / mAP | Average Precision / mean Average Precision |
| SAM | Spectral Angle Mapper |
| JWT | JSON Web Token |
| BullMQ | Redis-based Node.js job queue |
| MinIO / S3 | S3-compatible object storage |
| CI / CD | Continuous Integration / Continuous Delivery |
| GHCR | GitHub Container Registry |
| SRS / SDD | Software Requirements Specification / System Design Document |
| RBAC | Role-Based Access Control |
| Prism | Name of the UI theme / frontend prototype |

### 1.6 Intended Audience
Students and researchers, remote-sensing practitioners, agriculture/environment/material-analysis teams, and faculty evaluators; plus developers, testers and the supervisor reviewing this document.

### 1.7 References
See [Section 21](#21-references).

---

## 2. Overall Description

### 2.1 Product Perspective
A new, self-contained, containerised web application composed of a React client, a Node/Express API gateway, MongoDB, Redis/BullMQ, MinIO/S3 and a FastAPI ML service (optionally GPU-backed). It is not an extension of an existing system.

### 2.2 Need and Significance
HSI reveals class differences invisible in RGB; an AI workflow automates analysis and enables comparison of baseline vs advanced methods.

### 2.3 User Classes and Characteristics
| User class | Description | Key needs |
|---|---|---|
| Student / Researcher | Primary user running experiments | Upload/select data, preprocess, train, compare, export report |
| Remote-sensing practitioner | Applied analysis | Classification maps, error maps, quality inspection |
| Agriculture / environment / material-analysis team | Domain end-users | Class maps, spectral signatures |
| Faculty evaluator / reviewer | Reviews results | Reviewable staged workflow, read-only result links, reports |
| Administrator | Operates the platform | Users, storage, GPU worker status |

### 2.4 Operating Environment
- Browser-based client (responsive layout, light and dark mode).
- Docker-hosted services: client (Nginx), server, ml, mongo, redis, minio.
- Linux VM (single-VM Compose) or managed hosting (Vercel/Netlify, Render/Railway, MongoDB Atlas free tier).
- Optional GPU worker or cloud notebook for deep learning.

### 2.5 Feasibility Summary
| Aspect | Summary |
|---|---|
| Technical | Python ML stack, open HSI tools; depends on dataset size, labels, compute, annotations. |
| Operational | Staged workflow: inspect → preprocess → model → evaluate → report; reviewable with guide. |
| Economic | Public datasets, open-source tools, free-tier hosting; GPU optional (cloud notebook). |
| Schedule | 8-week plan, updated once scope and data access are confirmed. |

### 2.6 Design and Implementation Constraints
- C-1: Modelling in Python; Node.js is not used for training.
- C-2: Datasets must be public or approved; licence and download terms must be confirmed before use.
- C-3: Detection is only available for datasets with object-level bounding boxes.
- C-4: Spatially disjoint or patch-based splits are mandatory for evaluation.
- C-5: Node 20, MongoDB 7, Redis 7, Python 3.11, Docker.
- C-6: Compute (RAM, GPU) is finalised after data availability.

### 2.7 Assumptions and Dependencies
See [Section 19](#19-assumptions-dependencies-and-constraints).

---

## 3. System Architecture Overview

```
React (Vite + TS)  <-- REST + JWT / Socket.IO -->  Node/Express API Gateway
                                                     |-- MongoDB (metadata, metrics)
                                                     |-- Redis + BullMQ (job queue)
                                                     |-- MinIO / S3 (cubes, models, maps, plots)
                                                     '-- FastAPI ML Service (Python, optional GPU worker)
```

### 3.1 Component Responsibilities
| Component | Responsibility |
|---|---|
| React client (Vite + TypeScript) | Guided UI, charts, map overlays, live job progress |
| Node/Express API gateway | Auth, validation, ownership checks, dataset/pipeline/experiment CRUD, job enqueue, Socket.IO relay, report generation trigger |
| MongoDB | Metadata and metrics (datasets, pipelines, experiments, users) |
| Redis + BullMQ | Job queue for run/cancel of training and related jobs |
| MinIO / S3 | Storage of cubes, prepared cubes, models, prediction maps, plots |
| FastAPI ML service | HSI I/O, inspection, preprocessing, training, inference, detection, evaluation |

### 3.2 Module Mapping (Synopsis → Implementation)
| Synopsis module | Implementation |
|---|---|
| Dataset Acquisition and Inspection | `datasets` API + ML `/inspect` |
| Preprocessing and Feature Preparation | ML `/preprocess` (normalize, PCA, band selection, denoise) |
| Model Development | ML `/train` (SVM, 2D/3D-CNN) via job queue |
| Object Detection | ML `/detect`, gated by `hasObjectAnnotations` |
| Evaluation and Visualisation | ML `/evaluate` returns metrics JSON and PNG; React charts |
| Documentation | Report generator (PDF/Markdown of config and results) |

### 3.3 Job Flow
1. `POST /experiments/:id/run` validates and enqueues a BullMQ job.
2. The worker calls FastAPI `/train`, which streams progress.
3. The worker relays progress via Socket.IO (`progress`, `log`, `done` events).
4. On completion, metrics are written to MongoDB and artifacts to S3.
5. Seed, split and git SHA are stored to satisfy reproducibility.

---

## 4. Functional Requirements

> Priority: **M** = Mandatory (core), **O** = Optional (advanced), **C** = Conditional (depends on annotations).

### 4.1 Authentication and Account Management (AUTH)
| ID | Requirement | Pri |
|---|---|---|
| FR-AUTH-01 | The system shall allow users to register with name, email and password (email format validated, password ≥ 8 characters, matching confirmation). | M |
| FR-AUTH-02 | The system shall allow users to log in with email and password via `POST /auth/login`. | M |
| FR-AUTH-03 | The system shall issue short-lived JWT access tokens and refresh tokens, and support `POST /auth/refresh`. | M |
| FR-AUTH-04 | The system shall hash passwords with bcrypt. | M |
| FR-AUTH-05 | The UI shall provide inline validation on the sign-in / create-account form and a demo shortcut login. | M |
| FR-AUTH-06 | The UI shall keep the session in the browser and provide sign-out in the header. | M |
| FR-AUTH-07 | The system shall support role-based access (student, faculty reviewer). | O |

### 4.2 Dataset Acquisition and Library (DS)
| ID | Requirement | Pri |
|---|---|---|
| FR-DS-01 | The system shall allow listing (`GET /datasets`) and creating (`POST /datasets`) datasets. | M |
| FR-DS-02 | The system shall accept uploads in **.mat**, **ENVI (.hdr + .img)** and **GeoTIFF** formats. | M |
| FR-DS-03 | For large cubes, the system shall provide a presigned URL (`POST /datasets/upload-url`) for direct-to-S3 upload; Multer shall be used only for small files. | M |
| FR-DS-04 | Presigned URLs shall expire. | M |
| FR-DS-05 | The system shall offer public datasets: **Indian Pines, Pavia University, Salinas** (classification), and **HOD3K, HOD-1, M2SODAI** (detection verification). | M |
| FR-DS-06 | Each dataset shall carry a **licence field**, and licence/access conditions shall be tracked and enforced. | M |
| FR-DS-07 | Each dataset shall show an annotation status (object-annotation badge) derived from `hasObjectAnnotations`. | M |
| FR-DS-08 | The system shall support the HOD3K and M2SODAI data formats in the ML service. | M |
| FR-DS-09 | Upon dataset creation the API shall create the Dataset record and queue `/inspect`. | M |
| FR-DS-10 | The system shall support a catalogue of public datasets with licence tracking. | M |

### 4.3 Dataset Inspection (INS)
| ID | Requirement | Pri |
|---|---|---|
| FR-INS-01 | `POST /datasets/:id/inspect` shall trigger inspection, backed by ML `/inspect`. | M |
| FR-INS-02 | Inspection shall report shape (height, width, bands), band statistics, class histogram/distribution and NaN/invalid-value report. | M |
| FR-INS-03 | Inspection shall identify noisy bands and store them in `qualityReport.noisyBands`; NaN count in `qualityReport.nanCount`. | M |
| FR-INS-04 | The Inspector shall display dimensions, band count, class distribution and invalid values. | M |
| FR-INS-05 | The Inspector shall provide a **false-colour viewer with band sliders** (band selection for R/G/B channels). | M |
| FR-INS-06 | The Inspector shall provide **click-a-pixel spectral signature** display (animated "self-drawing" spectral curve in the prototype). | M |

### 4.4 Preprocessing (PRE)
| ID | Requirement | Pri |
|---|---|---|
| FR-PRE-01 | The system shall support preprocessing steps: **normalize, PCA, band selection, denoise** (noisy-band removal). | M |
| FR-PRE-02 | Pipelines shall be stored as ordered steps with parameters and a version (`GET/POST /pipelines`). | M |
| FR-PRE-03 | ML `/preprocess` shall output a **cached prepared cube (.npz)** and a preview. | M |
| FR-PRE-04 | The Preprocessing Studio shall provide a pipeline builder, **explained-variance chart**, and **before/after preview**. | M |
| FR-PRE-05 | The Prepare screen shall provide toggles for normalize, noisy-band removal and PCA, plus a **components slider with live variance readout**. | M |
| FR-PRE-06 | Preprocessing shall be documented and cached to avoid recomputation. | M |

### 4.5 Experiment Configuration (EXP)
| ID | Requirement | Pri |
|---|---|---|
| FR-EXP-01 | The system shall allow listing/creating experiments (`GET/POST /experiments`). | M |
| FR-EXP-02 | An experiment shall specify dataset, pipeline, task (`classification` or `detection`), model (`svm`, `cnn2d`, `cnn3d`, `yolo`) with hyperparameters, and split (strategy, train/val/test ratios, seed). | M |
| FR-EXP-03 | The Experiment Builder shall provide model selection, hyperparameters, **spatially stratified split** and seed. | M |
| FR-EXP-04 | Seed, split and **git SHA** shall be stored per experiment. | M |
| FR-EXP-05 | The workflow shall support SVM baseline first, then CNN. | M |
| FR-EXP-06 | Experiment status shall be one of `queued`, `running`, `done`, `failed`, `cancelled`. | M |
| FR-EXP-07 | The system shall support hyperparameter sweeps (Optuna) with leaderboard. | O |
| FR-EXP-08 | The system shall support MLflow experiment tracking. | O |

### 4.6 Training and Job Management (TRN)
| ID | Requirement | Pri |
|---|---|---|
| FR-TRN-01 | `POST /experiments/:id/run` shall validate and enqueue a BullMQ job. | M |
| FR-TRN-02 | `POST /experiments/:id/cancel` shall cancel a queued or running job. | M |
| FR-TRN-03 | `GET /jobs/:id` shall return job status. | M |
| FR-TRN-04 | The worker shall call ML `/train`, stream progress, and relay via Socket.IO (`progress`, `log`, `done`). | M |
| FR-TRN-05 | The Training Monitor shall show **live curves, logs and a cancel control**. | M |
| FR-TRN-06 | The Train screen shall provide an SVM / 2D-CNN / 3D-CNN selector and a live progress bar. | M |
| FR-TRN-07 | Train and inference times shall be recorded (`timings.train`, `timings.infer`). | M |
| FR-TRN-08 | On completion, metrics shall be saved to MongoDB and artifacts (model, prediction map, plots) to S3. | M |
| FR-TRN-09 | The client shall use a `useJobSocket` hook to receive job events. | M |

### 4.7 Prediction (PRD)
| ID | Requirement | Pri |
|---|---|---|
| FR-PRD-01 | ML `/predict` shall output a classification map (.png / .npy). | M |
| FR-PRD-02 | The Prediction Viewer shall display ground truth vs prediction, an **error map**, an **opacity slider** and a **legend toggle**. | M |
| FR-PRD-03 | Konva/Canvas shall be used for map overlays. | M |

### 4.8 Evaluation and Results (EVL)
| ID | Requirement | Pri |
|---|---|---|
| FR-EVL-01 | ML `/evaluate` shall compute OA, AA, Kappa, precision, recall, F1 (per class), confusion matrix, and IoU/mAP where supported. | M |
| FR-EVL-02 | Metrics shall be computed **only on held-out data** with documented definitions. | M |
| FR-EVL-03 | `GET /experiments/:id/metrics` shall return results; `GET /experiments/:id/artifacts` shall return maps, plots and model file. | M |
| FR-EVL-04 | The Results page shall show OA/AA/Kappa/P/R/F1, confusion matrix and per-class bars. | M |
| FR-EVL-05 | The Results screen (guided flow) shall show accuracy (count-up animation), per-class scores, optional confusion-matrix toggle and baseline-comparison note. | M |
| FR-EVL-06 | `GET /compare?ids=a,b,c` shall provide side-by-side comparison of experiments. | M |
| FR-EVL-07 | The system shall chart train/inference time vs accuracy. | M |
| FR-EVL-08 | The system shall generate limitation notes alongside results. | M |
| FR-EVL-09 | Experiment history shall be retained and viewable on the Dashboard. | M |

### 4.9 Reporting (RPT)
| ID | Requirement | Pri |
|---|---|---|
| FR-RPT-01 | `POST /experiments/:id/report` shall generate a report (PDF/Markdown) containing configuration, results and limitations. | M |
| FR-RPT-02 | The Reports page shall provide PDF/Markdown export. | M |
| FR-RPT-03 | The report shall include config (preprocessing, model, split, seed, git SHA), results and limitations. | M |
| FR-RPT-04 | The system shall provide shareable read-only result links for evaluators. | O |

### 4.10 Administration and Settings (ADM)
| ID | Requirement | Pri |
|---|---|---|
| FR-ADM-01 | The Settings/Admin page shall manage users, storage and show GPU worker status. | M |
| FR-ADM-02 | `GET /health` shall return service health. | M |
| FR-ADM-03 | The system shall provide an audit log. | O |
| FR-ADM-04 | The system shall apply S3 lifecycle rules. | O |

### 4.11 Dashboard (DSH)
| ID | Requirement | Pri |
|---|---|---|
| FR-DSH-01 | The Dashboard shall show projects, recent experiments and job status. | M |

### 4.12 Advanced / Optional Analysis (ADV)
| ID | Requirement | Pri |
|---|---|---|
| FR-ADV-01 | Spectral Angle Mapper (SAM) and band-importance visualisation. | O |
| FR-ADV-02 | Explainability (Grad-CAM or saliency over bands). | O |
| FR-ADV-03 | Hyperparameter sweeps (Optuna) with leaderboard. | O |
| FR-ADV-04 | Training/inference time vs accuracy trade-off chart. | O |
| FR-ADV-05 | Role-based access (student, faculty reviewer). | O |
| FR-ADV-06 | Shareable read-only result links. | O |
| FR-ADV-07 | MLflow tracking. | O |
| FR-ADV-08 | Audit log, rate limiting, S3 lifecycle rules. | O |

### 4.13 Object Detection (DET) – Conditional
See also [Section 9](#9-object-detection-and-verification-requirements).

| ID | Requirement | Pri |
|---|---|---|
| FR-DET-01 | ML `/detect` shall output bounding boxes and **reject** requests for datasets lacking object annotations. | C |
| FR-DET-02 | The backend shall reject detection for any dataset whose `hasObjectAnnotations` is `false`. | C |
| FR-DET-03 | For such datasets the UI shall show: *"No objects detected: dataset has no object-level annotations"*. | C |
| FR-DET-04 | The Data step shall display an availability badge; the Results step shall display an **Objects detected / No objects detected** badge. | C |
| FR-DET-05 | The Detection Viewer shall show boxes, IoU/mAP and a confidence slider. | C |
| FR-DET-06 | The Detection panel shall show predicted vs ground-truth boxes and an IoU table with **Verified** or **Missed** status. | C |
| FR-DET-07 | Detection results shall never be reported for unsupported datasets; otherwise reported as future work. | C |

---

## 5. User Interface Requirements

### 5.1 Technology Stack (Frontend)
React + Vite + TypeScript; React Router; TanStack Query; Zustand; Tailwind CSS with shadcn/ui and Lucide icons; Recharts/Plotly.js; Konva/Canvas; Socket.IO client; React Hook Form + Zod; Vitest + Playwright.

### 5.2 Prism UI Theme Requirements
| ID | Requirement |
|---|---|
| UI-01 | A sleek, guided interface built around a **four-step flow (Data → Prepare → Train → Results)** in one centred column. |
| UI-02 | Warm, hand-picked colour grading (not neon): paper-cream surfaces in light mode, warm charcoal in dark mode; terracotta accent; sage secondary; all flat, **no rainbow gradients**. |
| UI-03 | Cards rounded, controls pill-shaped, generous spacing. |
| UI-04 | Motion (page fade-ins, animated progress bars, count-up accuracy, self-drawing spectral curve) shall be **disabled for users who prefer reduced motion**. |
| UI-05 | Light and dark modes, responsive layout, smooth transitions, toasts. |
| UI-06 | Step bar with checkmarks; finished steps can be revisited. |
| UI-07 | Font: **Inter** in all text. |

### 5.3 Colour Tokens
| Token | Dark | Light | Use |
|---|---|---|---|
| Background | `#1B1917` | `#F4EFE6` | Page |
| Surface / raised | `#242120` / `#2D2A27` | `#FBF8F2` / `#ECE5D8` | Cards, inputs, step bar |
| Border | `#3A3632` | `#DDD4C4` | Dividers |
| Text / muted | `#F1EBE2` / `#A39A8D` | `#2A2622` / `#7A7164` | Body, labels |
| Accent (clay / terracotta) | `#E0905F` | `#B5522B` | Active step, buttons, progress, selection |
| Secondary (sage / moss) | `#8FA584` | `#5F7A55` | Completed steps, Objects detected, Verified |
| Spectrum curve | `#C96A3B`, `#D9A441`, `#7D9A6A` | same | Spectral signature line only |
| Font | Inter | Inter | All text |

### 5.4 Guided-Flow Screens
| Screen | Requirements |
|---|---|
| Sign in / Create account | Email + password form, inline validation, demo shortcut. |
| Data | Dataset chips with object-annotation badge, false-colour image, spectral signature for the tapped pixel. |
| Prepare | Toggles for normalize, noisy-band removal, PCA; components slider. |
| Train | SVM / 2D-CNN / 3D-CNN selector, live progress bar. |
| Results | Accuracy, per-class scores, optional confusion matrix. |
| Detection panel | Objects detected / No objects detected badge, predicted vs ground-truth boxes, IoU table with Verified/Missed. |

### 5.5 Application Pages
| # | Page | Content |
|---|---|---|
| 1 | Login/Register | Auth forms |
| 2 | Dashboard | Projects, recent experiments, job status |
| 3 | Dataset Library | Upload (.mat, ENVI .hdr+.img, GeoTIFF), public sets (Indian Pines, Pavia, Salinas), licence field |
| 4 | Dataset Inspector | Dimensions, band count, class distribution, invalid values, false-colour viewer with band sliders, click-a-pixel spectral signature |
| 5 | Preprocessing Studio | Pipeline builder, explained-variance chart, before/after preview |
| 6 | Experiment Builder | Model, hyperparameters, spatially stratified split, seed |
| 7 | Training Monitor | Live curves, logs, cancel |
| 8 | Results and Comparison | OA/AA/Kappa/P/R/F1, confusion matrix, per-class bars, side-by-side comparison |
| 9 | Prediction Viewer | Ground truth vs prediction, error map, opacity slider, legend toggle |
| 10 | Detection Viewer (conditional) | Boxes, IoU/mAP, confidence slider |
| 11 | Reports | Export PDF/Markdown |
| 12 | Settings/Admin | Users, storage, GPU worker status |

### 5.6 Frontend Code Organisation
```
client/src/
  api/  components/
  features/{datasets, preprocess, experiments, results, detection}/
  hooks/ (useJobSocket)  store/  routes/  types/
```

### 5.7 Usability and Accessibility
- Consistent labelled plots and a guided pipeline builder.
- Reduced-motion support.
- Inline form validation; toast feedback.
- Responsive layout.

---

## 6. External Interface Requirements

### 6.1 User Interface
See Section 5.

### 6.2 REST API – Node/Express Gateway
| Method | Route | Purpose |
|---|---|---|
| POST | `/auth/register`, `/auth/login`, `/auth/refresh` | Authentication (JWT access + refresh) |
| GET/POST | `/datasets` | List/create datasets |
| POST | `/datasets/upload-url` | Presigned URL for large cubes |
| POST | `/datasets/:id/inspect` | Trigger inspection |
| GET/POST | `/pipelines` | Preprocessing configs |
| GET/POST | `/experiments` | List/create experiments |
| POST | `/experiments/:id/run`, `/cancel` | Enqueue or cancel job |
| GET | `/experiments/:id/metrics`, `/artifacts` | Results, maps, plots, model file |
| GET | `/compare?ids=a,b,c` | Compare experiments |
| POST | `/experiments/:id/report` | Generate report |
| GET | `/jobs/:id` | Job status (plus Socket.IO `progress`, `log`, `done` events) |
| GET | `/health` | Health check |

API shall be documented via Swagger/OpenAPI.

### 6.3 ML Service API (FastAPI)
| Endpoint | Output |
|---|---|
| `POST /inspect` | Shape, band stats, class histogram, NaN report |
| `POST /preprocess` | Cached prepared cube (.npz) and preview |
| `POST /train` | Model and metrics (streamed progress) |
| `POST /predict` | Classification map (.png/.npy) |
| `POST /detect` | Boxes (rejects if dataset lacks annotations) |
| `POST /evaluate` | OA/AA/Kappa/F1, confusion matrix, IoU/mAP |

### 6.4 Real-Time Interface (Socket.IO)
Events: `progress`, `log`, `done`. Relayed by the Node worker from the ML service stream to the client.

### 6.5 Storage Interfaces
- **MongoDB** (Mongoose) for metadata/metrics.
- **Redis** (ioredis) for BullMQ.
- **MinIO / S3** for cubes, models, maps, plots (presigned URLs).

### 6.6 Prototype-to-Backend Mapping
| Prototype action | Backend call | ML service |
|---|---|---|
| Sign in or register | `POST /auth/login`, `/auth/register` | none |
| Choose a dataset | `GET /datasets` | none |
| Tap a pixel, view spectrum | `POST /datasets/:id/inspect` | `/inspect` |
| Set preprocessing | `POST /pipelines` | `/preprocess` |
| Start training | `POST /experiments/:id/run` (job + Socket.IO progress) | `/train` |
| View results | `GET /experiments/:id/metrics` | `/evaluate` |
| Detection panel | `GET /experiments/:id/artifacts` | `/detect`, `/evaluate` |
| Export report | `POST /experiments/:id/report` | none |

---

## 7. Data Requirements

### 7.1 Backend Libraries
Express, Mongoose, Zod, jsonwebtoken + bcrypt, presigned S3 upload (Multer for small files), BullMQ + ioredis, Socket.IO, Pino, Helmet, express-rate-limit, Swagger/OpenAPI, Jest + Supertest.

```
server/src/
  config/ routes/ controllers/ services/ models/ middleware/
  queue/ (producers, workers) sockets/ utils/ tests/
```

### 7.2 Data Models

**Dataset**
```js
{ owner, name, source: 'public'|'upload', licence, format,
  shape: {h, w, bands}, wavelengthRange, classNames: [String],
  classDistribution: Map, hasObjectAnnotations: Boolean,
  storageKey, qualityReport: {nanCount, noisyBands: [Number]}, createdAt }
```

**Pipeline**
```js
{ owner, steps: [{ type: 'normalize'|'pca'|'bandSelect'|'denoise', params }], version }
```

**Experiment**
```js
{ owner, dataset, pipeline, task: 'classification'|'detection',
  model: { type: 'svm'|'cnn2d'|'cnn3d'|'yolo', hyperparams },
  split: { strategy, trainRatio, valRatio, testRatio, seed },
  status: 'queued'|'running'|'done'|'failed'|'cancelled',
  metrics: { oa, aa, kappa, f1PerClass, confusionMatrix, mAP, iou },
  artifacts: { modelKey, predMapKey, plots: [String] },
  timings: { train, infer }, gitSha, createdAt }
```

### 7.3 Data Requirements Statements
| ID | Requirement |
|---|---|
| DR-01 | Every dataset shall have `owner`, `name`, `source`, `licence`, `format`, `shape`, `wavelengthRange`, `classNames`, `classDistribution`, `hasObjectAnnotations`, `storageKey`, `qualityReport`, `createdAt`. |
| DR-02 | Every pipeline shall record `owner`, ordered `steps` (type + params) and `version`. |
| DR-03 | Every experiment shall record owner, dataset, pipeline, task, model, split, status, metrics, artifacts, timings, git SHA, createdAt. |
| DR-04 | Supported formats: .mat (scipy.io), ENVI .hdr+.img and GeoTIFF (spectral/rasterio), HOD3K and M2SODAI formats. |
| DR-05 | Prepared cubes are stored as cached `.npz`; prediction maps as `.png`/`.npy`; plots as PNG. |
| DR-06 | Per-user ownership shall be recorded on all entities. |

### 7.4 Supported Datasets
| Dataset | Annotation | Task use |
|---|---|---|
| Indian Pines | Pixel-level labels | Classification; detection disabled |
| Pavia University | Pixel-level labels | Classification; detection disabled |
| Salinas | Pixel-level labels | Classification; detection disabled |
| HOD3K | Bounding boxes | Detection verification |
| HOD-1 | Bounding boxes | Detection verification |
| M2SODAI | Bounding boxes | Detection verification |
| San Diego (airport) | Pixel-level only | Not used for box metrics |

---

## 8. ML Service Requirements

| ID | Requirement |
|---|---|
| ML-01 | Implemented in Python 3.11 with FastAPI + Uvicorn and Pydantic validation. |
| ML-02 | Libraries: NumPy, scipy.io, spectral/rasterio, scikit-learn (SVM, PCA), PyTorch (3D-CNN), Matplotlib. |
| ML-03 | Models: SVM baseline, 2D-CNN, 3D-CNN, and a detector (YOLO-type one-stage; e.g., spectral-spatial two-stream backbone). |
| ML-04 | **Critical evaluation rule:** use **spatially disjoint or patch-based splits**. Random pixel splits leak neighbouring pixels and inflate accuracy and are prohibited. |
| ML-05 | Training shall stream progress to the caller. |
| ML-06 | Preprocessing shall be cached and reusable. |
| ML-07 | Seeds shall be controlled for reproducibility. |
| ML-08 | Model size shall be matched to available hardware; support PCA-reduced inputs and smaller 2D-CNN when GPU unavailable. |
| ML-09 | Optional GPU worker; Compose supports optional GPU reservation on `ml`. |
| ML-10 | ML test suite (pytest) runs on a tiny synthetic cube fixture; lint with ruff. |
| ML-11 | Overfitting mitigation: augmentation, regularisation, SVM baseline. |

### 8.1 Metrics Definitions (to be documented in the system)
- **Classification:** OA, AA, Kappa, per-class precision/recall/F1, confusion matrix.
- **Detection:** per-class AP, mAP@0.5, mAP@0.5:0.95, precision, recall, mean IoU.

---

## 9. Object Detection and Verification Requirements

### 9.1 Feature Gating
| ID | Requirement |
|---|---|
| OD-01 | Detection shall be enabled only if `hasObjectAnnotations` is true. |
| OD-02 | Standard classification scenes (Indian Pines, Pavia University, Salinas) have pixel-level labels but no bounding boxes → detection disabled. |
| OD-03 | The backend and ML `/detect` shall reject unsupported datasets. |
| OD-04 | UI message for unsupported datasets: *"No objects detected: dataset has no object-level annotations"*. |

### 9.2 Detection Datasets
| Dataset | Annotation | Size and content | Source |
|---|---|---|---|
| HOD3K | Bounding boxes | 3,242 hyperspectral images of campuses, stadiums and roads; 3 classes (people, cars, bikes); 512×256 px, 16 bands, 470–620 nm | github.com/hexiao-cs/S2ADet |
| HOD-1 | Bounding boxes | 454 hyperspectral images; first hyperspectral object detection dataset | Introduced in same paper as HOD3K |
| M2SODAI | Bounding boxes | 1,257 synchronized RGB + hyperspectral (224-band) aerial images; 2 classes (ships, floating matter); 5,764 instances per class | Project page |
| San Diego (airport) | Pixel-level only | Airplane targets; no boxes → not used for box metrics | Noted in HOD3K paper |

Licence and download terms shall be confirmed before use.

### 9.3 Verification Protocol
| Step | Requirement |
|---|---|
| V-1 | Use the official train/val/test split where provided; otherwise split **by image, never by tile**, to avoid leakage. |
| V-2 | Train a detector (e.g., one-stage detector with spectral-spatial two-stream backbone, with PCA or band selection to cut redundancy) and run inference on the held-out set. |
| V-3 | Match each predicted box to a ground-truth box of the same class; **IoU ≥ 0.5 = true positive**. |
| V-4 | Report per-class AP, mAP@0.5, mAP@0.5:0.95, precision, recall, mean IoU. |
| V-5 | Detection Viewer overlays predictions (**solid**) on ground truth (**dashed**) and lists IoU per object with **Verified/Missed** status. |
| V-6 | Backend rejects `/detect` for datasets where `hasObjectAnnotations` is false; UI shows the "No objects detected" message. |

### 9.4 Prototype Note
The frontend prototype lists HOD3K, M2SODAI and HOD-1 with this verification view. **Its boxes are illustrative until the ML service is run on real data.**

---

## 10. Non-Functional Requirements

| ID | Category | Requirement | How met |
|---|---|---|---|
| NFR-01 | Usability | Consistent labelled plots; guided pipeline builder | Prism UI, four-step flow |
| NFR-02 | Availability | Services shall be containerised with health checks | Docker, `/health` |
| NFR-03 | Efficiency | Preprocessing cached; model size matched to hardware | `.npz` cache, PCA, small CNN |
| NFR-04 | Accuracy | Metrics only on held-out data with documented definitions | Spatial splits |
| NFR-05 | Performance | Train/inference time recorded and charted vs accuracy | `timings`, trade-off chart |
| NFR-06 | Reliability | Seeds, splits and git SHA stored per experiment | Experiment model |
| NFR-07 | Maintainability | Modular code, comments, setup README | Monorepo structure |
| NFR-08 | Data handling | Licence field; access conditions enforced | Dataset model |
| NFR-09 | Reproducibility | Results reproducible and comparable | Seed, split, SHA, config report |
| NFR-10 | Scalability | Large cubes via presigned direct-to-S3 upload | Presigned URLs |
| NFR-11 | Portability | Docker Compose on single VM or managed hosts | Compose |
| NFR-12 | Testability | Unit/integration/e2e tests | Vitest, Playwright, Jest, Supertest, pytest |
| NFR-13 | Observability | Structured logging | Pino |
| NFR-14 | Accessibility | Reduced-motion respected | CSS/JS media query |
| NFR-15 | Documentation | API documented via OpenAPI; workflow, settings, limitations documented | Swagger, reports |

---

## 11. Security Requirements

| ID | Requirement |
|---|---|
| SEC-01 | Passwords hashed with bcrypt. |
| SEC-02 | Short-lived JWT access tokens with refresh tokens. |
| SEC-03 | Helmet HTTP security headers. |
| SEC-04 | Rate limiting (express-rate-limit). |
| SEC-05 | Zod input validation on all endpoints. |
| SEC-06 | Per-user ownership checks on all resources. |
| SEC-07 | Presigned URLs with expiry. |
| SEC-08 | Dataset licence tracking. |
| SEC-09 | Trivy filesystem scan in CI failing on HIGH/CRITICAL. |
| SEC-10 | RBAC (student, faculty reviewer) – optional. |
| SEC-11 | Audit log – optional. |
| SEC-12 | The ML service shall not be directly exposed to clients (accessed through the API gateway). |

---

## 12. Hardware and Software Requirements

### 12.1 Hardware
- Laptop/desktop with sufficient RAM and storage for the dataset.
- GPU system or cloud notebook if deep learning needs it (ML service only).
- Internet access for public datasets and packages.

### 12.2 Software
Node 20, MongoDB 7, Redis 7, MinIO/S3, Python 3.11, Docker, Git, Jupyter (for experiments).

---

## 13. End-to-End Workflow

| Step | Action |
|---|---|
| 1 | User uploads a cube (presigned S3) or picks a public dataset. |
| 2 | API creates the Dataset record and queues `/inspect`. |
| 3 | Inspector shows bands, classes, quality and spectral signatures. |
| 4 | User builds and previews a preprocessing pipeline. |
| 5 | User configures an experiment (SVM baseline first, then CNN). |
| 6 | Job is queued, trained and streamed live to the UI. |
| 7 | Metrics, confusion matrix and prediction maps are stored. |
| 8 | User compares experiments and reviews errors on the map. |
| 9 | If annotations exist, detection runs and IoU/mAP are shown. |
| 10 | User exports the report (config, results, limitations). |

---

## 14. DevOps, CI/CD and Deployment Requirements

### 14.1 Repository
Monorepo: `client/`, `server/`, `ml-service/`, `docker-compose.yml`.

### 14.2 Continuous Integration (`.github/workflows/ci.yml`)
Runs on every push and pull request with four parallel jobs:

| Job | Steps |
|---|---|
| client | `npm ci`, lint, typecheck, Vitest, build |
| server | `npm ci`, lint, Jest with Mongo 7 and Redis 7 service containers |
| ml | `pip install`, ruff, pytest on a tiny synthetic cube fixture |
| security | Trivy filesystem scan failing on HIGH/CRITICAL |

### 14.3 Continuous Delivery (`.github/workflows/cd.yml`)
Runs on tags `v*`:
1. Build and push Docker images for client, server and ml-service to GHCR (matrix build).
2. SSH to the host and run `docker compose pull && docker compose up -d --remove-orphans`.

### 14.4 Branch Flow
Feature branch → pull request (CI required, one review) → `develop` (auto-deploy to staging) → tag `vX.Y.Z` (production).

### 14.5 Compose Services
`client` (Nginx), `server`, `ml`, `mongo`, `redis`, `minio`; named volumes for mongo and minio; optional GPU reservation on `ml`.

### 14.6 Hosting Options
- Single VM with Compose (simplest); GPU worker only for `ml`.
- Vercel/Netlify (frontend), Render/Railway (API), MongoDB Atlas free tier.

---

## 15. Feature List

### 15.1 Core (maps to synopsis)
- [ ] Multi-format upload and public dataset catalogue with licence tracking
- [ ] Data inspection: dimensions, bands, class distribution, invalid values
- [ ] False-colour viewer with band selection and pixel spectral signatures
- [ ] Preprocessing pipeline: normalization, denoising, PCA, band selection
- [ ] SVM baseline vs CNN comparison with recorded splits and seeds
- [ ] Metrics: accuracy, precision, recall, F1, Kappa, confusion matrix
- [ ] Classification and error maps
- [ ] Conditional object detection with IoU/mAP
- [ ] Experiment history, comparison and exportable report

### 15.2 Advanced (optional)
- [ ] Spectral Angle Mapper and band-importance visualisation
- [ ] Explainability (Grad-CAM or saliency over bands)
- [ ] Hyperparameter sweeps (Optuna) with leaderboard
- [ ] Training/inference time vs accuracy trade-off chart
- [ ] Role-based access (student, faculty reviewer)
- [ ] Shareable read-only result links for evaluators
- [ ] MLflow experiment tracking
- [ ] Audit log, rate limiting, S3 lifecycle rules

---

## 16. Project Schedule and Build Order

| Week | Phase | Deliverable | MERN deliverable |
|---|---|---|---|
| 1 | Scope and literature review | Requirements, references | Repo, Docker Compose, auth, CI skeleton |
| 2 | Dataset selection and inspection | Dataset profile | Upload, `/inspect`, Inspector page |
| 3–4 | Preprocessing and baseline | Prepared data, baseline results | Preprocessing studio, SVM job |
| 5–6 | Candidate models | Trained models, logs | CNN training, Socket.IO monitor |
| 7 | Evaluation and error analysis | Metric comparison, plots | Results, comparison, maps, detection gate |
| 8 | Documentation and presentation | Final report, presentation | Reports, CD deployment, demo |

The 8-week plan is updated once scope and data access are confirmed.

---

## 17. Risks, Limitations and Mitigations

### 17.1 Expected Outcomes
Documented pipeline for loading and preparing HSI; baseline and comparative models; quantitative results and visualisations; detection results only if annotations exist; discussion of limitations, compute needs and future work.

### 17.2 Risk Register
| Risk | Mitigation |
|---|---|
| Limited labelled samples, overfitting | Spatial splits, augmentation, regularisation, SVM baseline |
| No object annotations | Detection stays disabled; documented as future work |
| Large cubes slow uploads | Presigned direct-to-S3 upload, cached preprocessing |
| No GPU | Smaller 2D-CNN, PCA-reduced input, cloud notebook |
| Poor generalisation | State dataset-specific limits; test on a second sensor if possible |

### 17.3 Known Limitations
- Results on one dataset are not assumed to generalise across sensors, locations or conditions.
- Dataset, classes, architecture and compute are not final until data availability and supervisor guidance are confirmed.
- Dataset licences and download terms are still to be confirmed.

---

## 18. Implementation Status and Remaining Work

> The clickable frontend prototype (Prism / HSI Studio) uses **sample data only**. Backend, ML service and CI/CD are **specified but not yet built or run**.

### 18.1 Frontend – Achieved in Prototype
| Task | Status | Description |
|---|---|---|
| Guided four-step flow | Done | Data, Prepare, Train, Results; step bar with checkmarks; finished steps revisitable |
| Sign in / Create account | Done (client-side demo) | Validates email, password ≥ 8, name, confirmation; demo login; session in browser; sign-out |
| Dataset library | Done | Indian Pines, Pavia University, HOD3K, M2SODAI, HOD-1 with annotation status; Upload tile is a placeholder |
| Dataset inspection | Done (sample) | Generated false-colour image; tap pixel for animated spectral signature |
| Preprocessing controls | Done (sample) | Normalize, noisy-band removal, PCA toggles; components slider with live variance readout |
| Model selection and training | Done (simulated) | SVM/2D-CNN/3D-CNN selector; Start training with live progress bar |
| Results | Done (sample) | Animated OA, per-class scores, confusion matrix toggle, baseline comparison note |
| Object detection status | Done | Availability badge on Data step; Objects detected / No objects detected badge on Results |
| Object detection verification | Done (illustrative) | Predicted over ground-truth boxes, per-object IoU, Verified/Missed, mean IoU |
| Report export | Placeholder | Confirmation toast only; no file produced |
| Look and feel | Done | Prism theme, light/dark, responsive, transitions, toasts, reduced-motion |
| Design iterations | Done | Material-style theme → dark console theme → current Prism design |

### 18.2 Backend and ML Service – Designed, Not Yet Built
| Component | Specified in | Status |
|---|---|---|
| Express API, routes and middleware | SDD §6 | Specified |
| Auth: register, login, refresh; bcrypt; JWT access/refresh | SDD §6 | Specified (prototype only imitates in browser) |
| Mongoose models (Dataset, Pipeline, Experiment) | SDD §6 | Specified |
| BullMQ job queue and Socket.IO progress | SDD §6 | Specified |
| FastAPI endpoints: inspect, preprocess, train, predict, detect, evaluate | SDD §7 | Specified |
| Spatially disjoint splits and detection IoU/mAP evaluation | SDD §7, 7.1 | Specified |
| Object storage for cubes, models, maps | SDD §4 | Specified |

### 18.3 Architecture and Delivery – Designed, Not Yet Run
| Item | Status |
|---|---|
| System architecture (React, Express, MongoDB, Redis, MinIO, FastAPI) | Designed |
| Module mapping from synopsis to system | Done |
| CI workflow: client, server, ML tests, security scan | Drafted |
| CD workflow: image build, push, SSH deploy | Drafted |
| Docker Compose service layout | Drafted |
| Dataset research for detection (HOD3K, HOD-1, M2SODAI) and verification protocol | Done; licences still to be confirmed |

### 18.4 Remaining Work
1. Build the Express API, Mongoose models and real JWT authentication.
2. Build the FastAPI service and load real datasets (.mat, ENVI, GeoTIFF; HOD3K and M2SODAI formats).
3. Train the SVM baseline, the CNN and a detector; record seeds, splits and git SHA.
4. Connect the frontend to the API, replacing sample data.
5. Run the CI/CD workflows and deploy.
6. Generate real reports and confirm dataset licences.

---

## 19. Assumptions, Dependencies and Constraints

### 19.1 Assumptions
- A suitable public or approved dataset will be available.
- Supervisor guidance will finalise dataset, classes, architecture and compute.
- Free-tier hosting and optional cloud-notebook GPU are sufficient.

### 19.2 Dependencies
- Public datasets (Indian Pines, Pavia, Salinas, HOD3K, HOD-1, M2SODAI) and their licence terms.
- Open-source libraries listed in Sections 5, 7 and 8.
- GHCR, GitHub Actions, and SSH access to the deployment host.
- Internet access for datasets and packages.

### 19.3 Constraints
See Section 2.6.

---

## 20. Requirements Traceability Matrix

| Objective | Functional requirements | Related components |
|---|---|---|
| OBJ-1 Select dataset | FR-DS-01…10 | Dataset Library, `/datasets` |
| OBJ-2 Inspect data | FR-INS-01…06 | Inspector, `/inspect` |
| OBJ-3 Preprocess | FR-PRE-01…06 | Preprocessing Studio, `/preprocess`, `/pipelines` |
| OBJ-4 Baseline vs CNN | FR-EXP-02…05, FR-TRN-01…09, FR-EVL-06 | Experiment Builder, Training Monitor, `/train` |
| OBJ-5 Spectral-spatial learning | ML-03, ML-04, FR-ADV-01…02 | ML service, 3D-CNN |
| OBJ-6 Conditional detection | FR-DET-01…07, OD-01…04, V-1…6 | Detection Viewer, `/detect` |
| OBJ-7 Evaluate and error analysis | FR-EVL-01…09, FR-PRD-01…03 | `/evaluate`, Prediction Viewer, Results |
| OBJ-8 Document | FR-RPT-01…04, NFR-15 | Reports, report generator |

| Scope rule | Satisfied by |
|---|---|
| SC-2 Feature-flagged detection | FR-DET-01…07, OD-01…04 |
| SC-3 Reproducible, held-out results | FR-EXP-04, FR-EVL-02, NFR-04, NFR-06, ML-04 |
| SC-4 No assumed generalisation | Section 17.3, FR-EVL-08 |

---

## 21. References

1. A. F. H. Goetz, G. Vane, J. E. Solomon, B. N. Rock, "Imaging Spectrometry for Earth Remote Sensing," *Science*, vol. 228, no. 4704, pp. 1147–1153, 1985.
2. G. Camps-Valls and L. Bruzzone, eds., *Kernel Methods for Remote Sensing Data Analysis*. Wiley, 2009.
3. F. A. Kruse et al., "The Spectral Image Processing System (SIPS)," *Remote Sensing of Environment*, vol. 44, no. 2–3, pp. 145–163, 1993.
4. G. Cheng and J. Han, "A Survey on Object Detection in Optical Remote Sensing Images," *ISPRS J. Photogrammetry and Remote Sensing*, vol. 117, pp. 11–28, 2016.
5. System Design Document: *AI-Based Hyperspectral Image Classification and Object Detection (MERN + Python ML Service)*.
6. HOD3K / HOD-1: github.com/hexiao-cs/S2ADet; M2SODAI project page (as cited in the SDD).

---

## 22. Appendices

### Appendix A – Open Items Requiring Confirmation
| # | Item |
|---|---|
| A-1 | Final dataset(s), classes, model architecture and compute (pending data availability and supervisor guidance). |
| A-2 | Licence and download terms for all datasets. |
| A-3 | Final 8-week schedule once data access is confirmed. |
| A-4 | Report format specifics (PDF vs Markdown generator implementation). |
| A-5 | Hosting choice (single VM Compose vs split managed hosting). |

### Appendix B – Requirement Priority Summary
| Priority | Meaning | Examples |
|---|---|---|
| M | Mandatory core | Auth, datasets, inspection, preprocessing, SVM/CNN, metrics, reports |
| C | Conditional | Object detection (requires object annotations) |
| O | Optional advanced | SAM, Grad-CAM, Optuna, MLflow, RBAC, share links, audit log |

*End of Document*
