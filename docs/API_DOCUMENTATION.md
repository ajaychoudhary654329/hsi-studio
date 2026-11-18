# API Documentation

## Base URL

- Local API gateway: `http://localhost:4000`
- ML service: `http://localhost:8000`

## Authentication

### POST /auth/login

Request body:

```json
{
  "email": "demo@hsistudio.dev",
  "password": "password123"
}
```

Response:

```json
{
  "token": "jwt-token",
  "user": {
    "id": "u-demo",
    "name": "Demo Researcher",
    "email": "demo@hsistudio.dev",
    "role": "student"
  }
}
```

### POST /auth/register

Request body:

```json
{
  "name": "New User",
  "email": "new@demo.com",
  "password": "password123"
}
```

## Dataset endpoints

### GET /datasets

Returns the available dataset catalogue.

### POST /datasets

Creates a dataset entry.

### POST /datasets/upload-url

Returns a presigned/example URL for direct-to-storage uploads.

### POST /datasets/:id/inspect

Triggers a dataset inspection job and returns metadata summary.

## Pipeline endpoints

### GET /pipelines

Returns available preprocessing pipelines.

### POST /pipelines

Creates a new pipeline entry.

## Experiment endpoints

### GET /experiments

Returns experiment entries.

### POST /experiments

Creates a new experiment.

### POST /experiments/:id/run

Starts an execution and returns job status.

### POST /experiments/:id/cancel

Cancels a queued or running experiment and emits a cancellation progress event.

## Results and reporting

- `GET /experiments/:id/metrics`
- `GET /experiments/:id/artifacts`
- `GET /compare?ids=a,b,c`
- `POST /experiments/:id/report`
- `POST /experiments/:id/detect`
- `GET /openapi.json`

### GET /jobs/:id

Returns the queued or completed run status.

## ML service endpoints

### GET /health

Returns service health metadata.

### POST /inspect

Returns basic cube metadata including shape, noisy bands, and class histogram.

### POST /preprocess

Runs a demo PCA + normalization workflow and returns prepared data metadata.

### POST /train

Runs a demo training response for classification or detection tasks.

### POST /detect

Returns a synthetic detection summary and map metadata.

### POST /evaluate

Returns evaluation metrics and a confusion matrix.
