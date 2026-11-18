# Architecture Overview

## Monorepo layout

```text
project-root/
├── client/               # React + TypeScript frontend
├── server/               # Express + TypeScript API gateway
├── ml-service/           # Python FastAPI model service
├── docs/                 # project documentation and implementation status
├── infra/                # local configuration notes
├── .env.example          # environment template
├── README.md
└── .gitignore
```

## Frontend responsibilities

The React client provides the user-facing workflow for the HSI system.

- Dashboard summarises datasets, experiments, and model readiness.
- Dataset catalogue supports creation and metadata display.
- Preprocessing studio contains pipeline definitions and parameter toggles.
- Experiment builder allows dataset selection, model choice, split strategy, and queue status.

## Backend responsibilities

The Express layer acts as the API gateway and orchestration point.

- Auth and JWT issuance.
- Dataset CRUD and upload URL creation.
- Pipeline configuration and job orchestration.
- Experiment creation and run trigger flow.
- Socket-based progress event handling for future real-time job updates.

## ML service responsibilities

The Python FastAPI layer covers ML system tasks.

- Dataset inspection and HSI metadata extraction.
- Preprocessing and PCA preparation.
- Model training for classification and detection tasks.
- Evaluation and artifact generation.

## Future dependency replacements

This local repo intentionally uses in-memory and generated demo data. In production, the following services should be connected:

- MongoDB for persistent metadata.
- Redis + BullMQ for queue state and scheduling.
- MinIO/S3 for uploaded cubes and generated artifacts.
- GPU cluster or cloud notebook for deep models such as CNN/YOLO.
