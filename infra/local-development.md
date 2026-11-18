# Local development guide

1. Install dependencies for the client and server.
2. Create a Python venv for the ML service.
3. Start the API gateway on port 4000.
4. Start the FastAPI service on port 8000.
5. Start the Vite client on its default port.

```bash
cd client && npm install && npm run dev -- --host 0.0.0.0
cd server && npm install && npm run dev
python -m venv .venv
source .venv/bin/activate
pip install -r ml-service/requirements.txt
cd ml-service && uvicorn app.main:app --reload --port 8000
```

This local setup intentionally mirrors the SRS architecture without requiring external cloud or GPU permissions.
