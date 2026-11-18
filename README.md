# HSI Studio

A working prototype of the AI-Based Hyperspectral Image Classification and Object Detection platform.

## Architecture

- `client/` – React + TypeScript dashboard for dataset inspection, preprocessing, and experiment management.
- `server/` – Express + TypeScript API gateway with JWT auth, dataset CRUD, pipeline management, and job orchestration.
- `ml-service/` – FastAPI Python service for inspection, preprocessing, training, and evaluation.
- `docs/` – project documentation and implementation status.

## Quick start

1. Install frontend dependencies:
   - `cd client && npm install`
2. Install backend dependencies:
   - `cd server && npm install`
3. Create virtual environment for Python ML service:
   - `python -m venv .venv`
   - `source .venv/bin/activate` (or `.venv\Scripts\activate` on Windows)
   - `pip install -r ml-service/requirements.txt`
4. Start services:
   - `cd server && npm run dev`
   - `cd ml-service && uvicorn app.main:app --reload --port 8000`
   - `cd client && npm run dev -- --host 0.0.0.0`

## Docker stack

The complete local service topology can be started with:

- `docker compose up --build`

This starts the React client, Express gateway, FastAPI service, MongoDB, Redis and MinIO.
The ML service is reachable only inside the Compose network. Cube uploads of 20 MiB and above use a presigned MinIO upload; if S3 is unconfigured, files up to the API's 100 MiB limit fall back to multipart upload. The browser-facing MinIO endpoint defaults to `http://localhost:9000`. Set `MINIO_PUBLIC_ENDPOINT` if the browser reaches the stack at another host, and set `MINIO_CORS_ORIGIN` to the client origin.

## Notes

The local Compose credentials and fallback JWT secret are for development only. Production deployments must provide a private `JWT_SECRET` of at least 32 characters and private MinIO/MongoDB/Redis credentials. The server persists refresh-token hashes and rotates tokens; the browser refreshes expired access tokens.

Uploads accept `.mat`, GeoTIFF, `.npy`/`.npz`, or a matching ENVI `.hdr`/`.img` pair. Classification training requires a real uploaded cube and matching pixel-label map. The Inspector visualizes real cube bands and spectra. Detection is gated on validated COCO annotations; detector training and IoU/mAP verification still require a compatible trained detector and licensed image data. Public-catalogue entries are metadata only and are not shipped with data or assumed licences.

Verification commands:

- `cd client && npm run build && npm run lint`
- `cd server && npm run build && npm test -- --runInBand`
- `cd ml-service && python -m pytest -q`
- `docker compose config --quiet`
