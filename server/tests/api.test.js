const { spawn } = require('child_process');
const fs = require('fs');
const os = require('os');
const path = require('path');

describe('HSI API gateway', () => {
  let server;
  let dataDirectory;
  const baseUrl = 'http://127.0.0.1:43123';

  beforeAll(async () => {
    dataDirectory = fs.mkdtempSync(path.join(os.tmpdir(), 'hsi-api-test-'));
    server = spawn(process.execPath, ['node_modules/tsx/dist/cli.mjs', 'src/index.ts'], {
      cwd: process.cwd(),
      env: {
        ...process.env,
        NODE_ENV: 'test',
        PORT: '43123',
        HSI_DATA_DIR: dataDirectory,
        MONGO_URI: '',
        REDIS_URL: '',
        JWT_SECRET: 'test-secret',
        MINIO_ENDPOINT: '',
        MINIO_PUBLIC_ENDPOINT: '',
        MINIO_ACCESS_KEY: '',
        MINIO_SECRET_KEY: '',
      },
      stdio: 'ignore',
    });
    for (let attempt = 0; attempt < 30; attempt += 1) {
      try {
        const response = await fetch(`${baseUrl}/health`);
        if (response.ok) return;
      } catch {}
      await new Promise((resolve) => setTimeout(resolve, 100));
    }
    throw new Error('API server did not become healthy');
  });

  afterAll(async () => {
    if (server) {
      server.kill();
      await new Promise((resolve) => server.once('exit', resolve));
    }
    if (dataDirectory) fs.rmSync(dataDirectory, { recursive: true, force: true });
  });

  test('health endpoint is available', async () => {
    const response = await fetch(`${baseUrl}/health`);
    expect(response.ok).toBe(true);
    expect(await response.json()).toEqual({ status: 'ok', service: 'hsi-studio-server' });
  });

  test('login and protected datasets work', async () => {
    const login = await fetch(`${baseUrl}/auth/login`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ email: 'demo@hsistudio.dev', password: 'password123' }),
    });
    expect(login.status).toBe(200);
    const { token } = await login.json();

    const datasets = await fetch(`${baseUrl}/datasets`, {
      headers: { authorization: `Bearer ${token}` },
    });
    expect(datasets.status).toBe(200);
    expect((await datasets.json()).datasets.length).toBeGreaterThan(0);
  });

  test('refresh tokens rotate and cannot be reused', async () => {
    const login = await fetch(`${baseUrl}/auth/login`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ email: 'demo@hsistudio.dev', password: 'password123' }),
    });
    const { refreshToken } = await login.json();
    const persistedState = fs.readFileSync(path.join(dataDirectory, 'hsi-state.json'), 'utf-8');
    expect(persistedState).not.toContain(refreshToken);
    const refresh = () => fetch(`${baseUrl}/auth/refresh`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ refreshToken }),
    });
    expect((await refresh()).status).toBe(200);
    expect((await refresh()).status).toBe(401);
  });

  test('large uploads fail explicitly when S3 is not configured', async () => {
    const login = await fetch(`${baseUrl}/auth/login`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ email: 'demo@hsistudio.dev', password: 'password123' }),
    });
    const { token } = await login.json();
    const datasetResponse = await fetch(`${baseUrl}/datasets`, {
      method: 'POST',
      headers: { authorization: `Bearer ${token}`, 'content-type': 'application/json' },
      body: JSON.stringify({ name: 'S3 test', type: 'classification' }),
    });
    const dataset = await datasetResponse.json();
    const response = await fetch(`${baseUrl}/datasets/upload-url`, {
      method: 'POST',
      headers: { authorization: `Bearer ${token}`, 'content-type': 'application/json' },
      body: JSON.stringify({ datasetId: dataset.id, fileName: 'cube.mat', fileType: 'application/octet-stream' }),
    });
    expect(response.status).toBe(503);
  });

  test('authenticated system status reports service and storage configuration', async () => {
    const login = await fetch(`${baseUrl}/auth/login`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ email: 'demo@hsistudio.dev', password: 'password123' }),
    });
    const { token } = await login.json();
    const response = await fetch(`${baseUrl}/system/status`, {
      headers: { authorization: `Bearer ${token}` },
    });
    expect(response.status).toBe(200);
    const status = await response.json();
    expect(status.api).toBe('healthy');
    expect(status.storage.s3).toBe('not configured');
  });

  test('detection experiments do not start unsupported training', async () => {
    const login = await fetch(`${baseUrl}/auth/login`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ email: 'demo@hsistudio.dev', password: 'password123' }),
    });
    const { token } = await login.json();
    const datasetResponse = await fetch(`${baseUrl}/datasets`, {
      method: 'POST',
      headers: { authorization: `Bearer ${token}`, 'content-type': 'application/json' },
      body: JSON.stringify({ name: 'Detection run test', type: 'detection', licence: 'Test-only' }),
    });
    const dataset = await datasetResponse.json();
    const annotations = {
      images: [{ id: 1, file_name: 'scene.tif', width: 10, height: 10 }],
      categories: [{ id: 1, name: 'target' }],
      annotations: [{ id: 1, image_id: 1, category_id: 1, bbox: [1, 1, 3, 4] }],
    };
    const annotationForm = new FormData();
    annotationForm.append('annotations', new Blob([JSON.stringify(annotations)], { type: 'application/json' }), 'boxes.json');
    const uploaded = await fetch(`${baseUrl}/datasets/${dataset.id}/annotations`, {
      method: 'POST',
      headers: { authorization: `Bearer ${token}` },
      body: annotationForm,
    });
    expect(uploaded.status).toBe(201);
    const created = await fetch(`${baseUrl}/experiments`, {
      method: 'POST',
      headers: { authorization: `Bearer ${token}`, 'content-type': 'application/json' },
      body: JSON.stringify({
        datasetId: dataset.id,
        pipelineId: 'pipe-default',
        task: 'detection',
        model: 'yolo',
        hyperparameters: {},
        split: { strategy: 'spatial', train: 0.6, val: 0.2, test: 0.2, seed: 42 },
      }),
    });
    expect(created.status).toBe(201);
    const { id } = await created.json();

    const run = await fetch(`${baseUrl}/experiments/${id}/run`, {
      method: 'POST',
      headers: { authorization: `Bearer ${token}` },
    });
    expect(run.status).toBe(501);

    const metrics = await fetch(`${baseUrl}/experiments/${id}/metrics`, {
      headers: { authorization: `Bearer ${token}` },
    });
    expect((await metrics.json()).status).toBe('queued');
  });

  test('classification run requires an uploaded cube and labels', async () => {
    const login = await fetch(`${baseUrl}/auth/login`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ email: 'demo@hsistudio.dev', password: 'password123' }),
    });
    const { token } = await login.json();
    const created = await fetch(`${baseUrl}/experiments`, {
      method: 'POST',
      headers: { authorization: `Bearer ${token}`, 'content-type': 'application/json' },
      body: JSON.stringify({
        datasetId: 'ds-indian-pines',
        pipelineId: 'pipe-default',
        task: 'classification',
        model: 'svm',
        hyperparameters: {},
        split: { strategy: 'spatial', train: 0.6, val: 0.2, test: 0.2, seed: 42 },
      }),
    });
    expect(created.status).toBe(201);
    const { id } = await created.json();

    const run = await fetch(`${baseUrl}/experiments/${id}/run`, {
      method: 'POST',
      headers: { authorization: `Bearer ${token}` },
    });
    expect(run.status).toBe(422);
    expect((await run.json()).message).toContain('Upload a hyperspectral cube');
  });

  test('object annotations only enable detection after valid COCO upload', async () => {
    const login = await fetch(`${baseUrl}/auth/login`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ email: 'demo@hsistudio.dev', password: 'password123' }),
    });
    const { token } = await login.json();
    const created = await fetch(`${baseUrl}/datasets`, {
      method: 'POST',
      headers: { authorization: `Bearer ${token}`, 'content-type': 'application/json' },
      body: JSON.stringify({
        name: 'Detection test',
        type: 'detection',
        hasObjectAnnotations: true,
        licence: 'Test-only',
      }),
    });
    expect(created.status).toBe(201);
    const dataset = await created.json();
    expect(dataset.hasObjectAnnotations).toBe(false);

    const invalidForm = new FormData();
    invalidForm.append('annotations', new Blob(['{"annotations":[]}'], { type: 'application/json' }), 'empty.json');
    const invalidUpload = await fetch(`${baseUrl}/datasets/${dataset.id}/annotations`, {
      method: 'POST',
      headers: { authorization: `Bearer ${token}` },
      body: invalidForm,
    });
    expect(invalidUpload.status).toBe(422);

    const coco = {
      images: [{ id: 1, file_name: 'scene.tif', width: 10, height: 10 }],
      categories: [{ id: 1, name: 'target' }],
      annotations: [{ id: 1, image_id: 1, category_id: 1, bbox: [1, 1, 3, 4], area: 12, iscrowd: 0 }],
    };
    const invalidReferenceForm = new FormData();
    invalidReferenceForm.append('annotations', new Blob([JSON.stringify({
      ...coco,
      annotations: [{ ...coco.annotations[0], category_id: 999 }],
    })], { type: 'application/json' }), 'bad-reference.json');
    const invalidReferenceUpload = await fetch(`${baseUrl}/datasets/${dataset.id}/annotations`, {
      method: 'POST',
      headers: { authorization: `Bearer ${token}` },
      body: invalidReferenceForm,
    });
    expect(invalidReferenceUpload.status).toBe(422);

    const validForm = new FormData();
    validForm.append('annotations', new Blob([JSON.stringify(coco)], { type: 'application/json' }), 'boxes.json');
    const annotationUpload = await fetch(`${baseUrl}/datasets/${dataset.id}/annotations`, {
      method: 'POST',
      headers: { authorization: `Bearer ${token}` },
      body: validForm,
    });
    expect(annotationUpload.status).toBe(201);
    expect((await annotationUpload.json()).hasObjectAnnotations).toBe(true);
  });
});
