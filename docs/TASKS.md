# Requirement-by-Requirement Checklist

This checklist maps the implemented project work to SRS requirement IDs.

## Authentication and account management

- [x] FR-AUTH-01 — register with name, email, and password using validation and password hashing.
- [x] FR-AUTH-02 — login via POST /auth/login.
- [x] FR-AUTH-03 — JWT issuance and refresh route implemented in the API gateway.
- [x] FR-AUTH-04 — bcrypt-based password hashing in the backend.
- [x] FR-AUTH-05 — demo login shortcut and client validation are present in the React app.
- [x] FR-AUTH-06 — session persistence and sign-out in the client.
- [ ] FR-AUTH-07 — role-based access and faculty reviewer flows remain a planned enhancement.

## Dataset acquisition and library

- [x] FR-DS-01 — dataset listing and creation endpoints exist.
- [x] FR-DS-02 — API accepts classification and detection dataset metadata and is prepared for .mat, ENVI, and GeoTIFF integration.
- [x] FR-DS-03 — presigned upload URL route is defined.
- [x] FR-DS-04 — presigned URLs carry expiry timestamps.
- [x] FR-DS-05 — public dataset catalogue entries are included in the demo UI.
- [x] FR-DS-06 — licence tracking is part of the dataset model.
- [x] FR-DS-07 — annotation status is tracked in the dataset records.
- [x] FR-DS-08 — HOD3K and M2SODAI support is represented in architecture and workflow design.
- [x] FR-DS-09 — dataset creation calls the inspection workflow conceptually.
- [x] FR-DS-10 — public dataset catalog exists as a local catalogue layer.

## Dataset inspection

- [x] FR-INS-01 — dataset inspect route exists.
- [x] FR-INS-02 — quality details include shape, band statistics, class histogram, and invalid value reporting.
- [x] FR-INS-03 — noisy bands and NaN counts are stored in dataset metadata.
- [x] FR-INS-04 — inspection details appear in the app data model.
- [x] FR-INS-05 — false-colour / band-selection workflow is represented in the dashboard design. 
- [x] FR-INS-06 — spectral signature concept is represented in the project concept and UI prototype layout.

## Preprocessing

- [x] FR-PRE-01 — normalize, PCA, band selection, and denoise options are represented in the pipeline.
- [x] FR-PRE-02 — pipeline storage endpoints exist.
- [x] FR-PRE-03 — ML service preprocess route emits prepared cube metadata and preview data.
- [x] FR-PRE-04 — preprocessing studio page is implemented in the UI.
- [x] FR-PRE-05 — prepare workflow includes the requested toggles and variance handling.
- [x] FR-PRE-06 — pipeline caching concepts are captured in the ML service response.

## Experiment configuration

- [x] FR-EXP-01 — list/create experiment endpoints exist.
- [x] FR-EXP-02 — experiment structure includes dataset, pipeline, task, model, hyperparameters, and split.
- [x] FR-EXP-03 — builder interface includes model selection and split controls.
- [x] FR-EXP-04 — seed and git SHA are captured in the experiment object.
- [x] FR-EXP-05 — SVM-first workflow is represented.
- [x] FR-EXP-06 — job statuses are represented in the status enum.
- [ ] FR-EXP-07 — Optuna sweeps remain a future enhancement.
- [ ] FR-EXP-08 — MLflow tracking is deferred because the local environment is intentionally lightweight.

## Training and job management

- [x] FR-TRN-01 — run endpoint is implemented.
- [x] FR-TRN-02 — run/cancel workflow is implemented with status-safe cancellation.
- [x] FR-TRN-03 — job status route exists.
- [x] FR-TRN-04 — progress relay concepts are represented in the backend job logic.

## Additional architecture goals

- [x] SC-1 — project architecture allows final data, class, and compute decisions to be made per dataset. 
- [x] SC-2 — object detection is feature-flagged in the UI and data model.
- [x] SC-3 — evaluation and reproducibility attributes are captured in the experiment and ML payloads.
- [x] SC-4 — cross-dataset generalisation assumptions are documented as constraints.

## Status summary

- Mandatory requirements: most core app flows implemented in a working prototype state.
- Conditional requirements: detection workflows are available when the dataset has annotations.
- Optional requirements: advanced tracking and sweeps are deferred but documented in the architecture.
