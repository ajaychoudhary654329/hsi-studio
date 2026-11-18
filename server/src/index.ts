import express, { Request, Response } from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import helmet from 'helmet';
import rateLimit from 'express-rate-limit';
import jwt from 'jsonwebtoken';
import bcrypt from 'bcryptjs';
import { Server } from 'socket.io';
import { createServer } from 'http';
import { z } from 'zod';
import multer from 'multer';
import {
  S3Client,
  CreateBucketCommand,
  GetObjectCommand,
  HeadObjectCommand,
  PutObjectCommand,
} from '@aws-sdk/client-s3';
import { getSignedUrl } from '@aws-sdk/s3-request-presigner';
import fs from 'node:fs';
import path from 'node:path';
import { MongoClient } from 'mongodb';
import { Queue, Worker } from 'bullmq';
import IORedis from 'ioredis';
import crypto from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { Readable } from 'node:stream';
import { pipeline } from 'node:stream/promises';

dotenv.config();

const app = express();
const server = createServer(app);
const io = new Server(server, {
  cors: { origin: '*' }
});

const port = Number(process.env.PORT ?? 4000);
const jwtSecret = process.env.JWT_SECRET ?? 'demo-secret';
if (process.env.NODE_ENV === 'production' && (!process.env.JWT_SECRET || jwtSecret.length < 32)) {
  throw new Error('Production requires a JWT_SECRET of at least 32 characters');
}
const refreshTokens = new Map<string, { userId: string; expiresAt: number }>();
const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: 100 * 1024 * 1024 } });

io.use((socket, next) => {
  const token: unknown = socket.handshake.auth?.token;
  if (typeof token !== 'string') {
    next(new Error('Unauthorized'));
    return;
  }
  try {
    const payload = jwt.verify(token, jwtSecret) as { userId: string };
    socket.data.userId = payload.userId;
    next();
  } catch {
    next(new Error('Unauthorized'));
  }
});
io.on('connection', (socket) => {
  socket.join(`user:${socket.data.userId}`);
});

app.use(helmet());
app.use(cors());
app.use(express.json({ limit: '5mb' }));
app.use(rateLimit({ windowMs: 15 * 60 * 1000, max: 300 }));

const registerSchema = z.object({
  name: z.string().min(2),
  email: z.string().email(),
  password: z.string().min(8)
});

const loginSchema = z.object({
  email: z.string().email(),
  password: z.string().min(8)
});

const datasetSchema = z.object({
  name: z.string().trim().min(1),
  sensor: z.string().default('AVIRIS'),
  type: z.enum(['classification', 'detection']),
  licence: z.string().default('Licence confirmation required before use')
});

const experimentSchema = z.object({
  datasetId: z.string().min(1),
  pipelineId: z.string().min(1),
  task: z.enum(['classification', 'detection']),
  model: z.enum(['svm', 'cnn2d', 'cnn3d', 'yolo']),
  hyperparameters: z.record(z.string(), z.any()).default({}),
  split: z.object({
    strategy: z.enum(['random', 'spatial']),
    train: z.number().min(0.1).max(0.9),
    val: z.number().min(0.05).max(0.4),
    test: z.number().min(0.1).max(0.5),
    seed: z.number().int().default(42)
  })
}).refine((experiment) => {
  const total = experiment.split.train + experiment.split.val + experiment.split.test;
  return Math.abs(total - 1) < 0.001;
}, {
  message: 'Train, validation, and test splits must sum to 1',
  path: ['split']
}).refine((experiment) => experiment.split.strategy === 'spatial', {
  message: 'Spatially disjoint or patch-based splitting is required',
  path: ['split', 'strategy'],
});

type User = {
  id: string;
  name: string;
  email: string;
  passwordHash: string;
  role: 'student' | 'faculty';
};

type Dataset = {
  ownerId?: string;
  id: string;
  name: string;
  sensor: string;
  source?: 'public' | 'upload';
  format?: string;
  storageKey?: string;
  labelsKey?: string;
  annotationsKey?: string;
  wavelengthRange?: [number, number];
  classNames?: string[];
  type: 'classification' | 'detection';
  hasObjectAnnotations: boolean;
  licence: string;
  status: 'draft' | 'inspected' | 'ready';
  qualityReport?: {
    height: number;
    width: number;
    bands: number;
    nanCount: number;
    noisyBands: number[];
    classHistogram: Record<string, number>;
  };
};

type Pipeline = {
  ownerId?: string;
  id: string;
  name: string;
  version: string;
  steps: Array<{ name: string; params: Record<string, number | boolean | string> }>;
};

type Experiment = {
  ownerId?: string;
  id: string;
  datasetId: string;
  pipelineId: string;
  task: 'classification' | 'detection';
  model: 'svm' | 'cnn2d' | 'cnn3d' | 'yolo';
  hyperparameters: Record<string, unknown>;
  status: 'queued' | 'running' | 'done' | 'failed' | 'cancelled';
  split: {
    strategy: 'random' | 'spatial';
    train: number;
    val: number;
    test: number;
    seed: number;
  };
  gitSha: string;
  metrics?: Record<string, unknown>;
  artifacts?: Record<string, unknown>;
  timings?: { train: number; infer: number };
  confusionMatrix?: number[][];
  perClass?: Record<string, number[]>;
};

const users: User[] = [
  {
    id: 'u-demo',
    name: 'Demo Researcher',
    email: 'demo@hsistudio.dev',
    passwordHash: bcrypt.hashSync('password123', 10),
    role: 'student'
  }
];

const datasets: Dataset[] = [
  {
    id: 'ds-indian-pines',
    name: 'Indian Pines',
    sensor: 'AVIRIS',
    source: 'public',
    type: 'classification',
    hasObjectAnnotations: false,
    licence: 'Licence confirmation required before use',
    status: 'draft'
  },
  {
    id: 'ds-hod3k',
    name: 'HOD3K',
    sensor: 'AISA',
    source: 'public',
    type: 'detection',
    hasObjectAnnotations: false,
    licence: 'Licence confirmation required before use',
    status: 'draft'
  },
  {
    id: 'ds-pavia-university',
    name: 'Pavia University',
    sensor: 'ROSIS',
    source: 'public',
    type: 'classification',
    hasObjectAnnotations: false,
    licence: 'Licence confirmation required before use',
    status: 'draft'
  },
  {
    id: 'ds-salinas',
    name: 'Salinas',
    sensor: 'AVIRIS',
    source: 'public',
    type: 'classification',
    hasObjectAnnotations: false,
    licence: 'Licence confirmation required before use',
    status: 'draft'
  },
  {
    id: 'ds-hod1',
    name: 'HOD-1',
    sensor: 'Hyperspectral',
    source: 'public',
    type: 'detection',
    hasObjectAnnotations: false,
    licence: 'Licence confirmation required before use',
    status: 'draft'
  },
  {
    id: 'ds-m2sodai',
    name: 'M2SODAI',
    sensor: 'RGB + 224-band hyperspectral',
    source: 'public',
    type: 'detection',
    hasObjectAnnotations: false,
    licence: 'Licence confirmation required before use',
    status: 'draft'
  }
];

const pipelines: Pipeline[] = [
  {
    id: 'pipe-default',
    name: 'Baseline Spectral Pipeline',
    version: '1.0.0',
    steps: [
      { name: 'normalize', params: { method: 'minmax' } },
      { name: 'band-selection', params: { keep: 30 } },
      { name: 'pca', params: { components: 25, explainedVariance: 0.94 } }
    ]
  }
];

const experiments: Experiment[] = [];

const stateDirectory = process.env.HSI_DATA_DIR
  ? path.resolve(process.env.HSI_DATA_DIR)
  : path.resolve(process.cwd(), 'data');
const stateFile = path.join(stateDirectory, 'hsi-state.json');
let mongoClient: MongoClient | null = null;

if (fs.existsSync(stateFile)) {
  try {
    const restored = JSON.parse(fs.readFileSync(stateFile, 'utf-8')) as {
      users?: User[];
      datasets?: Dataset[];
      pipelines?: Pipeline[];
      experiments?: Experiment[];
      refreshTokens?: Array<[string, { userId: string; expiresAt: number }]>;
    };
    if (restored.users?.length) users.splice(0, users.length, ...restored.users);
    if (restored.datasets?.length) datasets.splice(0, datasets.length, ...restored.datasets);
    if (restored.pipelines?.length) pipelines.splice(0, pipelines.length, ...restored.pipelines);
    if (restored.experiments?.length) experiments.splice(0, experiments.length, ...restored.experiments);
    for (const [hash, record] of restored.refreshTokens ?? []) {
      if (record.expiresAt > Date.now()) refreshTokens.set(hash, record);
    }
  } catch (error) {
    console.error('Could not restore JSON state', error);
  }
}

const persistState = () => {
  fs.mkdirSync(stateDirectory, { recursive: true });
  fs.writeFileSync(stateFile, JSON.stringify({
    users,
    datasets,
    pipelines,
    experiments,
    refreshTokens: Array.from(refreshTokens.entries()),
  }, null, 2));
  if (mongoClient) {
    const db = mongoClient.db();
    void Promise.all([
      db.collection('datasets').bulkWrite(datasets.map((dataset) => ({
        updateOne: { filter: { id: dataset.id }, update: { $set: dataset }, upsert: true },
      }))),
      db.collection('pipelines').bulkWrite(pipelines.map((pipeline) => ({
        updateOne: { filter: { id: pipeline.id }, update: { $set: pipeline }, upsert: true },
      }))),
      db.collection('experiments').bulkWrite(experiments.map((experiment) => ({
        updateOne: { filter: { id: experiment.id }, update: { $set: experiment }, upsert: true },
      }))),
    ]).catch((error) => console.error('MongoDB persistence failed', error));
  }
};

if (process.env.MONGO_URI) {
  mongoClient = new MongoClient(process.env.MONGO_URI);
  mongoClient.connect()
    .then(async () => {
      const db = mongoClient!.db();
      const [storedDatasets, storedPipelines, storedExperiments] = await Promise.all([
        db.collection<Dataset>('datasets').find({}).toArray(),
        db.collection<Pipeline>('pipelines').find({}).toArray(),
        db.collection<Experiment>('experiments').find({}).toArray(),
      ]);
      if (storedDatasets.length) datasets.splice(0, datasets.length, ...storedDatasets);
      if (storedPipelines.length) pipelines.splice(0, pipelines.length, ...storedPipelines);
      if (storedExperiments.length) experiments.splice(0, experiments.length, ...storedExperiments);
      console.log('MongoDB connected and state restored');
    })
    .catch((error) => {
      console.error('MongoDB unavailable; using JSON state fallback', error);
      mongoClient = null;
    });
}

type MlTrainingResult = {
  metrics: Record<string, unknown>;
  artifacts: Record<string, unknown>;
  confusion_matrix?: number[][];
  per_class?: Record<string, number[]>;
  timings?: { train: number; infer: number };
};

const requiredIntegerParameter = (step: Pipeline['steps'][number], name: string, fallback: number) => {
  const value = step.params[name] ?? fallback;
  if (typeof value !== 'number' || !Number.isInteger(value) || value <= 0) {
    throw new Error(`Pipeline parameter "${name}" must be a positive integer`);
  }
  return value;
};

const trainExperiment = async (experiment: Experiment): Promise<MlTrainingResult> => {
  const dataset = datasets.find((entry) => entry.id === experiment.datasetId);
  const pipeline = pipelines.find((entry) => entry.id === experiment.pipelineId);
  if (!dataset?.storageKey) throw new Error('Dataset has no uploaded cube');
  if (!dataset.labelsKey) throw new Error('Dataset has no uploaded label map');
  if (!pipeline) throw new Error('Experiment pipeline not found');

  const normalizeStep = pipeline.steps.find((step) => step.name.toLowerCase() === 'normalize');
  const denoiseStep = pipeline.steps.find((step) => ['denoise', 'remove-noisy-bands'].includes(step.name.toLowerCase()));
  const bandSelectionStep = pipeline.steps.find((step) => ['band-selection', 'band_selection'].includes(step.name.toLowerCase()));
  const pcaStep = pipeline.steps.find((step) => step.name.toLowerCase() === 'pca');
  const method = normalizeStep?.params.method;
  const preprocessingPayload = {
    cubePath: dataset.storageKey,
    normalize: Boolean(normalizeStep),
    method: typeof method === 'string' ? method : 'minmax',
    remove_noisy_bands: Boolean(denoiseStep),
    keep_bands: bandSelectionStep
      ? requiredIntegerParameter(bandSelectionStep, 'keep', dataset.qualityReport?.bands ?? 25)
      : null,
    pca_components: pcaStep
      ? requiredIntegerParameter(pcaStep, 'components', 25)
      : null,
  };
  const mlUrl = process.env.ML_SERVICE_URL ?? 'http://localhost:8000';
  const preprocessingResponse = await fetch(`${mlUrl}/preprocess`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(preprocessingPayload),
  });
  if (!preprocessingResponse.ok) {
    const failure = await preprocessingResponse.json() as { detail?: unknown };
    const detail = typeof failure.detail === 'string' ? failure.detail : JSON.stringify(failure.detail);
    throw new Error(detail || `ML preprocessing returned ${preprocessingResponse.status}`);
  }
  const preprocessingResult = await preprocessingResponse.json() as { cache?: { path?: string } };
  if (!preprocessingResult.cache?.path) throw new Error('ML preprocessing returned no prepared cube path');

  const trainingResponse = await fetch(`${mlUrl}/train`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({
      model: experiment.model,
      task: experiment.task,
      dataset: dataset.name,
      seed: experiment.split.seed,
      cubePath: preprocessingResult.cache.path,
      labelsPath: dataset.labelsKey,
      splitStrategy: experiment.split.strategy,
      trainRatio: experiment.split.train,
      valRatio: experiment.split.val,
      testRatio: experiment.split.test,
      hyperparameters: experiment.hyperparameters,
    }),
  });
  if (!trainingResponse.ok) {
    const failure = await trainingResponse.json() as { detail?: unknown };
    const detail = typeof failure.detail === 'string' ? failure.detail : JSON.stringify(failure.detail);
    throw new Error(detail || `ML training returned ${trainingResponse.status}`);
  }
  return await trainingResponse.json() as MlTrainingResult;
};

const persistArtifactCopies = async (experiment: Experiment, result: MlTrainingResult) => {
  const endpoint = process.env.MINIO_ENDPOINT;
  const accessKeyId = process.env.MINIO_ACCESS_KEY;
  const secretAccessKey = process.env.MINIO_SECRET_KEY;
  if (!endpoint || !accessKeyId || !secretAccessKey) {
    return { ...result.artifacts, storage: 'local-only; S3 is not configured' };
  }
  const client = createS3Client(endpoint);
  const bucket = process.env.MINIO_BUCKET ?? 'hsi-cubes';
  await ensureS3Bucket(client, bucket);
  const s3Artifacts: Record<string, string> = {};
  for (const [name, artifactPath] of Object.entries(result.artifacts)) {
    if (typeof artifactPath !== 'string' || !fs.existsSync(artifactPath)) continue;
    const key = `artifacts/${experiment.ownerId ?? 'shared'}/${experiment.id}/${path.basename(artifactPath)}`;
    await client.send(new PutObjectCommand({
      Bucket: bucket,
      Key: key,
      Body: fs.createReadStream(artifactPath),
    }));
    s3Artifacts[name] = key;
  }
  return { ...result.artifacts, s3Artifacts, storage: 'local-and-s3' };
};

let experimentQueue: Queue | null = null;
const redisConnection = process.env.REDIS_URL ? new IORedis(process.env.REDIS_URL, { maxRetriesPerRequest: null }) : null;
if (redisConnection) {
  experimentQueue = new Queue('hsi-experiments', { connection: redisConnection });
  const isCancelled = (id: string) => experiments.find((entry) => entry.id === id)?.status === 'cancelled';
  void new Worker('hsi-experiments', async (job) => {
    const experiment = experiments.find((entry) => entry.id === job.data.experimentId);
    if (!experiment) throw new Error('Experiment not found');
    if (experiment.status === 'cancelled') return;
    experiment.status = 'running';
    emitToUser(experiment.ownerId, 'progress', { id: experiment.id, progress: 15, logs: ['BullMQ worker started'] });
    emitToUser(experiment.ownerId, 'log', { id: experiment.id, message: 'Calling FastAPI ML service' });
    emitToUser(experiment.ownerId, 'progress', { id: experiment.id, progress: 45, logs: ['Loading cube and split metadata'] });
    const mlResult = await trainExperiment(experiment);
    if (isCancelled(experiment.id)) return;
    mlResult.artifacts = await persistArtifactCopies(experiment, mlResult);
    experiment.status = 'done';
    experiment.metrics = mlResult.metrics;
    experiment.artifacts = mlResult.artifacts;
    experiment.confusionMatrix = mlResult.confusion_matrix;
    experiment.perClass = mlResult.per_class;
    experiment.timings = mlResult.timings;
    persistState();
    emitToUser(experiment.ownerId, 'progress', { id: experiment.id, progress: 100, logs: ['Training completed'] });
    emitToUser(experiment.ownerId, 'done', { id: experiment.id, status: 'done', metrics: experiment.metrics });
  }, { connection: redisConnection }).on('failed', (job, error) => {
    if (!job) return;
    const experiment = experiments.find((entry) => entry.id === job.data.experimentId);
    if (experiment) {
      if (experiment.status !== 'running') return;
      experiment.status = 'failed';
      persistState();
      emitToUser(experiment.ownerId, 'done', { id: experiment.id, status: 'failed', error: error.message });
    }
  });
}

const authMiddleware = (req: Request, res: Response, next: () => void) => {
  const header = req.headers.authorization;

  if (!header?.startsWith('Bearer ')) {
    res.status(401).json({ message: 'Unauthorized' });
    return;
  }

  try {
    const token = header.slice('Bearer '.length);
    const payload = jwt.verify(token, jwtSecret) as { userId: string };
    (req as Request & { user?: { userId: string } }).user = { userId: payload.userId };
    next();
  } catch (error) {
    res.status(401).json({ message: 'Invalid or expired token' });
  }
};

const createJwt = (userId: string) => jwt.sign({ userId }, jwtSecret, { expiresIn: '1h' });
const hashRefreshToken = (token: string) => crypto.createHash('sha256').update(token).digest('hex');

const createRefreshToken = (userId: string) => {
  const token = crypto.randomBytes(32).toString('hex');
  refreshTokens.set(hashRefreshToken(token), { userId, expiresAt: Date.now() + 7 * 24 * 60 * 60 * 1000 });
  persistState();
  return token;
};

const authenticatedUserId = (req: Request) =>
  (req as Request & { user?: { userId: string } }).user?.userId;
const emitToUser = (userId: string | undefined, event: string, payload: unknown) => {
  if (userId) io.to(`user:${userId}`).emit(event, payload);
};
const createS3Client = (endpoint: string) => new S3Client({
  endpoint,
  region: process.env.MINIO_REGION ?? 'us-east-1',
  forcePathStyle: true,
  credentials: {
    accessKeyId: process.env.MINIO_ACCESS_KEY ?? '',
    secretAccessKey: process.env.MINIO_SECRET_KEY ?? '',
  },
});
const ensureS3Bucket = async (client: S3Client, bucket: string) => {
  try {
    await client.send(new CreateBucketCommand({ Bucket: bucket }));
  } catch (error) {
    const status = typeof error === 'object' && error !== null && '$metadata' in error
      ? (error as { $metadata?: { httpStatusCode?: number } }).$metadata?.httpStatusCode
      : undefined;
    if (status !== 409) throw error;
  }
};

const canAccessResource = (req: Request, resource: { ownerId?: string }) =>
  !resource.ownerId || resource.ownerId === authenticatedUserId(req);
const canModifyResource = (req: Request, resource: { ownerId?: string }) =>
  Boolean(resource.ownerId && resource.ownerId === authenticatedUserId(req));

app.get('/health', (_req, res) => {
  res.json({ status: 'ok', service: 'hsi-studio-server' });
});

app.get('/system/status', authMiddleware, async (_req, res) => {
  const mlUrl = process.env.ML_SERVICE_URL ?? 'http://localhost:8000';
  let mlStatus = 'unavailable';
  try {
    const response = await fetch(`${mlUrl}/health`, { signal: AbortSignal.timeout(2000) });
    if (response.ok) mlStatus = 'healthy';
  } catch {
    mlStatus = 'unavailable';
  }
  res.json({
    api: 'healthy',
    ml: mlStatus,
    storage: {
      mongo: mongoClient ? 'configured' : 'not configured',
      redis: redisConnection?.status === 'ready' ? 'connected' : redisConnection ? 'unavailable' : 'not configured',
      s3: process.env.MINIO_ENDPOINT && process.env.MINIO_ACCESS_KEY && process.env.MINIO_SECRET_KEY
        ? 'configured'
        : 'not configured',
    },
    worker: {
      queue: experimentQueue ? 'BullMQ' : 'in-process fallback',
      gpu: 'not detected by this deployment',
    },
  });
});

app.get('/openapi.json', (_req, res) => {
  res.json({
    openapi: '3.0.3',
    info: { title: 'HSI Studio API', version: '1.0.0' },
    servers: [{ url: `http://localhost:${port}` }],
    paths: {
      '/auth/register': { post: { summary: 'Register a user account' } },
      '/auth/login': { post: { summary: 'Authenticate a user' } },
      '/auth/refresh': { post: { summary: 'Rotate an access and refresh token' } },
      '/datasets/upload': { post: { summary: 'Upload a small cube through the API' } },
      '/datasets': { get: { summary: 'List datasets' }, post: { summary: 'Create a dataset' } },
      '/datasets/upload-url': { post: { summary: 'Create a presigned S3 cube upload URL' } },
      '/datasets/upload-complete': { post: { summary: 'Validate and attach presigned S3 cube uploads' } },
      '/datasets/{id}/labels': { post: { summary: 'Upload a pixel label map' } },
      '/datasets/{id}/annotations': { post: { summary: 'Upload validated COCO object annotations' } },
      '/datasets/{id}/inspect': { post: { summary: 'Inspect an uploaded cube' } },
      '/datasets/{id}/visualize': { post: { summary: 'Render false color and pixel spectrum' } },
      '/pipelines': { get: { summary: 'List pipelines' }, post: { summary: 'Create a pipeline' } },
      '/preprocess': { post: { summary: 'Preprocess and cache a cube' } },
      '/experiments': { get: { summary: 'List experiments' }, post: { summary: 'Create an experiment' } },
      '/experiments/{id}/run': { post: { summary: 'Run an experiment' } },
      '/experiments/{id}/cancel': { post: { summary: 'Cancel an experiment' } },
      '/experiments/{id}/metrics': { get: { summary: 'Get experiment metrics' } },
      '/experiments/{id}/artifacts': { get: { summary: 'Get experiment artifacts' } },
      '/experiments/{id}/maps': { get: { summary: 'Render held-out truth, prediction, and error maps' } },
      '/experiments/{id}/report': { post: { summary: 'Generate a Markdown report' } },
      '/experiments/{id}/report/pdf': { post: { summary: 'Generate a PDF report' } },
      '/experiments/{id}/detect': { post: { summary: 'Run supported object detection' } },
      '/compare': { get: { summary: 'Compare experiments' } },
      '/system/status': { get: { summary: 'Authenticated storage and worker status' } },
      '/health': { get: { summary: 'Service health' } },
    },
  });
});

app.post('/auth/register', (req, res) => {
  const parsed = registerSchema.safeParse(req.body);
  if (!parsed.success) {
    res.status(400).json({ message: parsed.error.issues[0]?.message ?? 'Invalid payload' });
    return;
  }

  const { name, email, password } = parsed.data;
  if (users.some((u) => u.email.toLowerCase() === email.toLowerCase())) {
    res.status(409).json({ message: 'Email already exists' });
    return;
  }

  const user: User = {
    id: `u-${crypto.randomUUID()}`,
    name,
    email,
    passwordHash: bcrypt.hashSync(password, 10),
    role: 'student'
  };

  users.push(user);
  persistState();
  const token = createJwt(user.id);
  const refreshToken = createRefreshToken(user.id);
  res.status(201).json({ token, refreshToken, user: { id: user.id, name, email, role: user.role } });
});

app.post('/auth/login', (req, res) => {
  const parsed = loginSchema.safeParse(req.body);
  if (!parsed.success) {
    res.status(400).json({ message: parsed.error.issues[0]?.message ?? 'Invalid payload' });
    return;
  }

  const user = users.find((entry) => entry.email.toLowerCase() === parsed.data.email.toLowerCase());
  if (!user || !bcrypt.compareSync(parsed.data.password, user.passwordHash)) {
    res.status(401).json({ message: 'Invalid email or password' });
    return;
  }

  res.json({
    token: createJwt(user.id),
    refreshToken: createRefreshToken(user.id),
    user: { id: user.id, name: user.name, email: user.email, role: user.role }
  });
});

app.post('/auth/refresh', (req, res) => {
  const { refreshToken } = req.body as { refreshToken?: string };
  if (!refreshToken) {
    res.status(400).json({ message: 'Refresh token required' });
    return;
  }
  const tokenHash = hashRefreshToken(refreshToken);
  const record = refreshTokens.get(tokenHash);
  if (!record || record.expiresAt < Date.now()) {
    if (record) refreshTokens.delete(tokenHash);
    res.status(401).json({ message: 'Refresh token expired' });
    return;
  }
  refreshTokens.delete(tokenHash);
  const nextRefreshToken = createRefreshToken(record.userId);
  res.json({ token: createJwt(record.userId), refreshToken: nextRefreshToken });
});

app.get('/datasets', authMiddleware, (req, res) => {
  const userId = (req as Request & { user?: { userId: string } }).user?.userId;
  res.json({ datasets: datasets.filter((dataset) => !dataset.ownerId || dataset.ownerId === userId) });
});

app.post('/datasets', authMiddleware, (req, res) => {
  const parsed = datasetSchema.safeParse(req.body);
  if (!parsed.success) {
    res.status(400).json({ message: parsed.error.issues[0]?.message ?? 'Invalid dataset payload' });
    return;
  }

  const dataset: Dataset = {
    id: `ds-${crypto.randomUUID()}`,
    ...parsed.data,
    ownerId: authenticatedUserId(req),
    source: 'upload',
    hasObjectAnnotations: false,
    status: 'draft'
  };

  datasets.push(dataset);
  persistState();
  emitToUser(dataset.ownerId, 'dataset-created', dataset);
  res.status(201).json(dataset);
});

app.post('/datasets/upload-url', authMiddleware, async (req, res) => {
  const parsed = z.object({
    fileName: z.string().min(1),
    fileType: z.string().min(1),
    datasetId: z.string().min(1),
  }).safeParse(req.body);
  if (!parsed.success) {
    res.status(400).json({ message: parsed.error.issues[0]?.message ?? 'Invalid upload request' });
    return;
  }
  const dataset = datasets.find((entry) => entry.id === parsed.data.datasetId);
  if (!dataset || !canModifyResource(req, dataset)) {
    res.status(404).json({ message: 'Dataset not found' });
    return;
  }
  const safeName = path.basename(parsed.data.fileName).replace(/[^a-zA-Z0-9._-]/g, '_');
  const extension = path.extname(safeName).toLowerCase();
  if (!['.mat', '.hdr', '.img', '.tif', '.tiff', '.npy', '.npz'].includes(extension)) {
    res.status(415).json({ message: 'Unsupported cube format' });
    return;
  }
  const endpoint = process.env.MINIO_ENDPOINT;
  const publicEndpoint = process.env.MINIO_PUBLIC_ENDPOINT ?? endpoint;
  const accessKeyId = process.env.MINIO_ACCESS_KEY;
  const secretAccessKey = process.env.MINIO_SECRET_KEY;
  if (!endpoint || !publicEndpoint || !accessKeyId || !secretAccessKey) {
    res.status(503).json({ message: 'S3/MinIO storage is not configured' });
    return;
  }
  const bucket = process.env.MINIO_BUCKET ?? 'hsi-cubes';
  const key = `uploads/${authenticatedUserId(req)}/${dataset.id}/${crypto.randomUUID()}-${safeName}`;
  try {
    const internalClient = createS3Client(endpoint);
    await ensureS3Bucket(internalClient, bucket);
    const signingClient = createS3Client(publicEndpoint);
    const url = await getSignedUrl(signingClient, new PutObjectCommand({
      Bucket: bucket,
      Key: key,
      ContentType: parsed.data.fileType,
    }), { expiresIn: 3600 });
    res.json({
      url,
      key,
      fileName: safeName,
      expiresAt: new Date(Date.now() + 3600 * 1000).toISOString(),
      fileType: parsed.data.fileType,
    });
  } catch (error) {
    res.status(500).json({
      message: 'Could not create presigned upload',
      detail: error instanceof Error ? error.message : 'Unknown error',
    });
  }
});

app.post('/datasets/upload-complete', authMiddleware, async (req, res) => {
  const parsed = z.object({
    datasetId: z.string().min(1),
    files: z.array(z.object({ key: z.string().min(1), fileName: z.string().min(1) })).min(1).max(2),
  }).safeParse(req.body);
  if (!parsed.success) {
    res.status(400).json({ message: parsed.error.issues[0]?.message ?? 'Invalid upload completion request' });
    return;
  }
  const dataset = datasets.find((entry) => entry.id === parsed.data.datasetId);
  if (!dataset || !canModifyResource(req, dataset)) {
    res.status(404).json({ message: 'Dataset not found' });
    return;
  }
  const prefix = `uploads/${authenticatedUserId(req)}/${dataset.id}/`;
  const files = parsed.data.files.map((file) => ({
    ...file,
    fileName: path.basename(file.fileName).replace(/[^a-zA-Z0-9._-]/g, '_'),
    extension: path.extname(file.fileName).toLowerCase(),
  }));
  const extensions = files.map((file) => file.extension);
  const enviPair = files.length === 2 && extensions.includes('.hdr') && extensions.includes('.img');
  if (files.some((file) => !file.key.startsWith(prefix) || path.extname(file.key).toLowerCase() !== file.extension)) {
    res.status(403).json({ message: 'Upload object does not belong to this dataset or filename' });
    return;
  }
  if (files.length === 1 && ['.hdr', '.img'].includes(extensions[0])) {
    res.status(422).json({ message: 'ENVI cubes require both the matching .hdr and .img files' });
    return;
  }
  if ((files.length > 1 && !enviPair)
    || (enviPair && path.parse(files[0].fileName).name.toLowerCase() !== path.parse(files[1].fileName).name.toLowerCase())
    || (!enviPair && !['.mat', '.tif', '.tiff', '.npy', '.npz'].includes(extensions[0]))) {
    res.status(422).json({ message: 'Upload one supported cube file or a matching ENVI .hdr/.img pair' });
    return;
  }
  const endpoint = process.env.MINIO_ENDPOINT;
  const accessKeyId = process.env.MINIO_ACCESS_KEY;
  const secretAccessKey = process.env.MINIO_SECRET_KEY;
  if (!endpoint || !accessKeyId || !secretAccessKey) {
    res.status(503).json({ message: 'S3/MinIO storage is not configured' });
    return;
  }
  const client = createS3Client(endpoint);
  const bucket = process.env.MINIO_BUCKET ?? 'hsi-cubes';
  const datasetDirectory = path.resolve(process.cwd(), 'uploads', crypto.randomUUID());
  try {
    fs.mkdirSync(datasetDirectory, { recursive: true });
    let totalBytes = 0;
    let storageKey = '';
    for (const file of files) {
      const metadata = await client.send(new HeadObjectCommand({ Bucket: bucket, Key: file.key }));
      if (!metadata.ContentLength || metadata.ContentLength > 2 * 1024 * 1024 * 1024) {
        res.status(413).json({ message: 'Uploaded cube must be between 1 byte and 2 GB' });
        return;
      }
      totalBytes += metadata.ContentLength;
      const object = await client.send(new GetObjectCommand({ Bucket: bucket, Key: file.key }));
      if (!(object.Body instanceof Readable)) {
        throw new Error('S3 object response did not contain a readable file stream');
      }
      const destination = path.join(datasetDirectory, file.fileName);
      await pipeline(object.Body, fs.createWriteStream(destination, { flags: 'wx' }));
      if (file.extension === '.hdr' || files.length === 1) storageKey = destination;
    }
    dataset.storageKey = storageKey;
    dataset.format = path.extname(storageKey).slice(1).toLowerCase();
    dataset.status = 'draft';
    persistState();
    res.status(201).json({
      storageKey,
      fileName: path.basename(storageKey),
      bytes: totalBytes,
      format: dataset.format,
      files: files.map((file) => file.fileName),
    });
  } catch (error) {
    res.status(502).json({
      message: 'Could not finalize the S3 upload',
      detail: error instanceof Error ? error.message : 'Unknown error',
    });
  }
});

app.post('/datasets/upload', authMiddleware, upload.array('cube', 2), (req, res) => {
  const files = Array.isArray(req.files) ? req.files : [];
  if (!files.length) {
    res.status(400).json({ message: 'cube file is required' });
    return;
  }
  const datasetId = typeof req.body.datasetId === 'string' ? req.body.datasetId : undefined;
  if (!datasetId) {
    res.status(400).json({ message: 'datasetId is required to associate the uploaded cube' });
    return;
  }
  const linkedDataset = datasets.find((entry) => entry.id === datasetId);
  if (!linkedDataset || !canModifyResource(req, linkedDataset)) {
    res.status(404).json({ message: 'Dataset not found' });
    return;
  }
  const extensions = files.map((file) => path.extname(file.originalname).toLowerCase());
  const enviPair = extensions.length === 2 && extensions.includes('.hdr') && extensions.includes('.img');
  if (extensions.length === 1 && ['.hdr', '.img'].includes(extensions[0])) {
    res.status(422).json({ message: 'ENVI cubes require both the matching .hdr and .img files' });
    return;
  }
  if (files.length > 1 && !enviPair) {
    res.status(415).json({ message: 'Upload either one supported cube file or one matching ENVI .hdr/.img pair' });
    return;
  }
  if (enviPair) {
    const header = files[extensions.indexOf('.hdr')];
    const image = files[extensions.indexOf('.img')];
    if (path.parse(header.originalname).name.toLowerCase() !== path.parse(image.originalname).name.toLowerCase()) {
      res.status(422).json({ message: 'The ENVI .hdr and .img filenames must have the same basename' });
      return;
    }
  } else if (!['.mat', '.tif', '.tiff', '.npy', '.npz'].includes(extensions[0])) {
    res.status(415).json({ message: 'Unsupported cube format' });
    return;
  }
  const uploadsDirectory = path.resolve(process.cwd(), 'uploads');
  const datasetDirectory = path.join(uploadsDirectory, crypto.randomUUID());
  fs.mkdirSync(datasetDirectory, { recursive: true });
  let storageKey = '';
  for (const file of files) {
    const safeName = enviPair
      ? path.basename(file.originalname)
      : path.basename(file.originalname).replace(/[^a-zA-Z0-9._-]/g, '_');
    const destination = path.join(datasetDirectory, safeName);
    fs.writeFileSync(destination, file.buffer);
    if (path.extname(file.originalname).toLowerCase() === '.hdr' || files.length === 1) {
      storageKey = destination;
    }
  }
  linkedDataset.storageKey = storageKey;
  linkedDataset.format = path.extname(storageKey).slice(1).toLowerCase();
  linkedDataset.status = 'draft';
  persistState();
  res.status(201).json({
    storageKey,
    fileName: path.basename(storageKey),
    bytes: files.reduce((total, file) => total + file.size, 0),
    format: linkedDataset.format,
    files: files.map((file) => path.basename(file.originalname)),
  });
});

app.post('/datasets/:id/labels', authMiddleware, upload.single('labels'), (req: Request<{ id: string }>, res) => {
  const dataset = datasets.find((entry) => entry.id === req.params.id);
  if (!dataset || !canModifyResource(req, dataset)) {
    res.status(404).json({ message: 'Dataset not found' });
    return;
  }
  if (!req.file) {
    res.status(400).json({ message: 'labels file is required' });
    return;
  }
  if (!['.mat', '.npy', '.npz', '.tif', '.tiff', '.json'].includes(path.extname(req.file.originalname).toLowerCase())) {
    res.status(415).json({ message: 'Unsupported labels format' });
    return;
  }
  const uploadsDirectory = path.resolve(process.cwd(), 'uploads');
  fs.mkdirSync(uploadsDirectory, { recursive: true });
  const safeName = path.basename(req.file.originalname).replace(/[^a-zA-Z0-9._-]/g, '_');
  const storageKey = path.join(uploadsDirectory, `${crypto.randomUUID()}-${safeName}`);
  fs.writeFileSync(storageKey, req.file.buffer);
  dataset.labelsKey = storageKey;
  persistState();
  res.status(201).json({ labelsKey: storageKey, fileName: safeName, bytes: req.file.size });
});

app.post('/datasets/:id/annotations', authMiddleware, upload.single('annotations'), (req: Request<{ id: string }>, res) => {
  const dataset = datasets.find((entry) => entry.id === req.params.id);
  if (!dataset || !canModifyResource(req, dataset)) {
    res.status(404).json({ message: 'Dataset not found' });
    return;
  }
  if (dataset.type !== 'detection') {
    res.status(422).json({ message: 'Object annotations can only be attached to detection datasets' });
    return;
  }
  if (!req.file || path.extname(req.file.originalname).toLowerCase() !== '.json') {
    res.status(415).json({ message: 'A COCO-format JSON annotation file is required' });
    return;
  }
  let document: unknown;
  try {
    document = JSON.parse(req.file.buffer.toString('utf-8'));
  } catch {
    res.status(422).json({ message: 'Annotation file is not valid JSON' });
    return;
  }
  if (typeof document !== 'object' || document === null
    || !Array.isArray((document as { images?: unknown }).images)
    || !Array.isArray((document as { categories?: unknown }).categories)
    || !Array.isArray((document as { annotations?: unknown }).annotations)) {
    res.status(422).json({ message: 'Annotations must contain COCO images, categories, and annotations arrays' });
    return;
  }
  const images = (document as { images: unknown[] }).images;
  const categories = (document as { categories: unknown[] }).categories;
  const annotations = (document as { annotations: unknown[] }).annotations;
  const imageDimensions = new Map<number, { width: number; height: number }>();
  const validImages = images.every((entry) => {
    if (typeof entry !== 'object' || entry === null) return false;
    const image = entry as { id?: unknown; width?: unknown; height?: unknown };
    if (typeof image.id !== 'number' || typeof image.width !== 'number' || typeof image.height !== 'number'
      || !Number.isInteger(image.id) || !Number.isFinite(image.width) || !Number.isFinite(image.height)
      || image.width <= 0 || image.height <= 0 || imageDimensions.has(image.id)) return false;
    imageDimensions.set(image.id, { width: image.width, height: image.height });
    return true;
  });
  const categoryIds = new Set<number>();
  const validCategories = categories.every((entry) => {
    if (typeof entry !== 'object' || entry === null) return false;
    const category = entry as { id?: unknown; name?: unknown };
    if (typeof category.id !== 'number' || !Number.isInteger(category.id)
      || typeof category.name !== 'string' || !category.name.trim() || categoryIds.has(category.id)) return false;
    categoryIds.add(category.id);
    return true;
  });
  const validBoxes = annotations.length > 0 && annotations.every((entry) => {
    if (typeof entry !== 'object' || entry === null || !('bbox' in entry)) return false;
    const annotation = entry as { image_id?: unknown; category_id?: unknown; bbox?: unknown };
    const bbox = annotation.bbox;
    if (!Array.isArray(bbox) || bbox.length !== 4
      || !bbox.every((coordinate) => typeof coordinate === 'number' && Number.isFinite(coordinate))) return false;
    const [x, y, boxWidth, boxHeight] = bbox as number[];
    const dimensions = typeof annotation.image_id === 'number' ? imageDimensions.get(annotation.image_id) : undefined;
    return typeof annotation.image_id === 'number'
      && typeof annotation.category_id === 'number'
      && categoryIds.has(annotation.category_id)
      && dimensions !== undefined
      && x >= 0
      && y >= 0
      && boxWidth > 0
      && boxHeight > 0
      && x + boxWidth <= dimensions.width
      && y + boxHeight <= dimensions.height;
  });
  if (!validImages || !validCategories || !images.length || !categories.length || !validBoxes) {
    res.status(422).json({ message: 'COCO annotations must include images, categories, and at least one valid bounding box' });
    return;
  }
  const uploadsDirectory = path.resolve(process.cwd(), 'uploads');
  fs.mkdirSync(uploadsDirectory, { recursive: true });
  const safeName = path.basename(req.file.originalname).replace(/[^a-zA-Z0-9._-]/g, '_');
  const storageKey = path.join(uploadsDirectory, `${crypto.randomUUID()}-${safeName}`);
  fs.writeFileSync(storageKey, req.file.buffer);
  dataset.annotationsKey = storageKey;
  dataset.hasObjectAnnotations = true;
  persistState();
  res.status(201).json({ annotationsKey: storageKey, hasObjectAnnotations: true, fileName: safeName, bytes: req.file.size });
});

app.post('/datasets/:id/inspect', authMiddleware, (req: Request<{ id: string }>, res) => {
  const dataset = datasets.find((entry) => entry.id === req.params.id);
  if (!dataset || !canAccessResource(req, dataset)) {
    res.status(404).json({ message: 'Dataset not found' });
    return;
  }
  const mlUrl = process.env.ML_SERVICE_URL ?? 'http://localhost:8000';
  void fetch(`${mlUrl}/inspect`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({
      name: dataset.name,
      bands: dataset.qualityReport?.bands ?? 220,
      height: dataset.qualityReport?.height ?? 128,
      width: dataset.qualityReport?.width ?? 128,
      cubePath: dataset.storageKey,
      labelsPath: dataset.labelsKey,
    }),
  }).then(async (response) => {
    if (!response.ok) throw new Error(`ML inspect returned ${response.status}`);
    const payload = await response.json() as {
      shape: { height: number; width: number; bands: number };
      band_statistics: { mean: number; std: number };
      quality_report: { nanCount: number; noisyBands: number[]; classHistogram: Record<string, number>; invalidPixels: number };
      status: string;
    };
    dataset.status = 'inspected';
    dataset.qualityReport = {
      height: payload.shape.height,
      width: payload.shape.width,
      bands: payload.shape.bands,
      nanCount: payload.quality_report.nanCount,
      noisyBands: payload.quality_report.noisyBands,
      classHistogram: payload.quality_report.classHistogram,
    };
    persistState();
    res.json({
      datasetId: dataset.id,
      shape: payload.shape,
      bandStatistics: payload.band_statistics,
      classHistogram: payload.quality_report.classHistogram,
      nanCount: payload.quality_report.nanCount,
      noisyBands: payload.quality_report.noisyBands,
      status: payload.status,
    });
  }).catch((error) => {
    res.status(502).json({ message: 'ML inspection failed', detail: error instanceof Error ? error.message : 'Unknown error' });
  });
});

app.post('/datasets/:id/visualize', authMiddleware, async (req: Request<{ id: string }>, res) => {
  const dataset = datasets.find((entry) => entry.id === req.params.id);
  if (!dataset || !canAccessResource(req, dataset)) {
    res.status(404).json({ message: 'Dataset not found' });
    return;
  }
  if (!dataset.storageKey) {
    res.status(422).json({ message: 'Upload a real cube before requesting a visualization.' });
    return;
  }
  const parsed = z.object({
    redBand: z.number().int().nonnegative(),
    greenBand: z.number().int().nonnegative(),
    blueBand: z.number().int().nonnegative(),
    x: z.number().int().nonnegative(),
    y: z.number().int().nonnegative(),
  }).safeParse(req.body);
  if (!parsed.success) {
    res.status(400).json({ message: 'Color bands and pixel coordinates must be non-negative integers.' });
    return;
  }
  try {
    const mlUrl = process.env.ML_SERVICE_URL ?? 'http://localhost:8000';
    const response = await fetch(`${mlUrl}/visualize`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ ...parsed.data, cubePath: dataset.storageKey }),
    });
    const result = await response.json() as { detail?: unknown };
    if (!response.ok) {
      res.status(response.status).json({
        message: typeof result.detail === 'string' ? result.detail : 'ML visualization failed',
      });
      return;
    }
    res.json(result);
  } catch (error) {
    res.status(502).json({ message: 'Could not reach ML visualization service', detail: error instanceof Error ? error.message : 'Unknown error' });
  }
});

app.post('/preprocess', authMiddleware, async (req, res) => {
  const parsed = z.object({
    datasetId: z.string().optional(),
  }).passthrough().safeParse(req.body);
  if (!parsed.success) {
    res.status(400).json({ message: 'Invalid preprocessing request' });
    return;
  }
  const dataset = parsed.data.datasetId
    ? datasets.find((entry) => entry.id === parsed.data.datasetId)
    : undefined;
  if (parsed.data.datasetId && (!dataset || !canAccessResource(req, dataset))) {
    res.status(404).json({ message: 'Dataset not found' });
    return;
  }
  const mlUrl = process.env.ML_SERVICE_URL ?? 'http://localhost:8000';
  try {
    const response = await fetch(`${mlUrl}/preprocess`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ ...parsed.data, cubePath: dataset?.storageKey }),
    });
    if (!response.ok) {
      res.status(response.status).json({ message: 'ML preprocessing failed' });
      return;
    }
    res.json(await response.json());
  } catch (error) {
    res.status(502).json({ message: 'Could not reach ML preprocessing service', detail: error instanceof Error ? error.message : 'Unknown error' });
  }
});

app.get('/pipelines', authMiddleware, (req, res) => {
  const userId = (req as Request & { user?: { userId: string } }).user?.userId;
  res.json({ pipelines: pipelines.filter((pipeline) => !pipeline.ownerId || pipeline.ownerId === userId) });
});

app.post('/pipelines', authMiddleware, (req, res) => {
  const payload = z.object({
    name: z.string(),
    version: z.string(),
    steps: z.array(z.object({ name: z.string(), params: z.record(z.string(), z.union([z.string(), z.number(), z.boolean()])) }))
  }).safeParse(req.body);

  if (!payload.success) {
    res.status(400).json({ message: payload.error.issues[0]?.message ?? 'Invalid pipeline payload' });
    return;
  }

  const pipeline: Pipeline = { id: `pipe-${crypto.randomUUID()}`, ...payload.data };
  pipeline.ownerId = authenticatedUserId(req);
  pipelines.push(pipeline);
  persistState();
  res.status(201).json(pipeline);
});

app.get('/experiments', authMiddleware, (req, res) => {
  const userId = (req as Request & { user?: { userId: string } }).user?.userId;
  res.json({ experiments: experiments.filter((experiment) => !experiment.ownerId || experiment.ownerId === userId) });
});

app.post('/experiments', authMiddleware, (req, res) => {
  const parsed = experimentSchema.safeParse(req.body);
  if (!parsed.success) {
    res.status(400).json({ message: parsed.error.issues[0]?.message ?? 'Invalid experiment payload' });
    return;
  }

  const userId = authenticatedUserId(req);
  const dataset = datasets.find((entry) => entry.id === parsed.data.datasetId);
  const pipeline = pipelines.find((entry) => entry.id === parsed.data.pipelineId);
  if (!dataset || !canAccessResource(req, dataset) || !pipeline || !canAccessResource(req, pipeline)) {
    res.status(404).json({ message: 'Dataset or pipeline not found' });
    return;
  }
  if (parsed.data.task === 'detection' && !dataset.hasObjectAnnotations) {
    res.status(422).json({ message: 'Detection requires object-level annotations' });
    return;
  }
  if ((parsed.data.task === 'detection') !== (parsed.data.model === 'yolo')) {
    res.status(422).json({ message: 'YOLO is only valid for detection; classification requires SVM or CNN' });
    return;
  }
  let gitSha = 'git-sha-unavailable';
  try {
    gitSha = execFileSync('git', ['rev-parse', 'HEAD'], { cwd: process.cwd(), encoding: 'utf-8' }).trim();
  } catch {
    gitSha = 'git-sha-unavailable';
  }
  const experiment: Experiment = {
    id: `exp-${crypto.randomUUID()}`,
    ...parsed.data,
    ownerId: userId,
    status: 'queued',
    gitSha
  };

  experiments.push(experiment);
  persistState();
  emitToUser(experiment.ownerId, 'experiment-created', experiment);
  res.status(201).json(experiment);
});

app.post('/experiments/:id/run', authMiddleware, (req: Request<{ id: string }>, res) => {
  const experiment = experiments.find((entry) => entry.id === req.params.id);
  if (!experiment || !canAccessResource(req, experiment)) {
    res.status(404).json({ message: 'Experiment not found' });
    return;
  }
  if (experiment.status !== 'queued') {
    res.status(409).json({ message: `Experiment cannot run from ${experiment.status} status` });
    return;
  }
  if (experiment.task === 'detection') {
    res.status(501).json({ message: 'Detection training is not implemented; classification training only is available.' });
    return;
  }
  const dataset = datasets.find((entry) => entry.id === experiment.datasetId);
  if (!dataset?.storageKey) {
    res.status(422).json({ message: 'Upload a hyperspectral cube before running this experiment.' });
    return;
  }
  if (!dataset.labelsKey) {
    res.status(422).json({ message: 'Upload a matching label map before running classification training.' });
    return;
  }
  if (!pipelines.some((entry) => entry.id === experiment.pipelineId)) {
    res.status(404).json({ message: 'Experiment pipeline not found' });
    return;
  }

  experiment.status = 'running';
  emitToUser(experiment.ownerId, 'progress', { id: experiment.id, status: 'running', progress: 0, logs: ['Queued for worker'] });
  persistState();

  if (experimentQueue) {
    void experimentQueue.add('train', { experimentId: experiment.id }, {
      removeOnComplete: true,
      removeOnFail: false,
      jobId: experiment.id,
    }).catch((error) => {
      if (experiment.status !== 'running') return;
      experiment.status = 'failed';
      persistState();
      emitToUser(experiment.ownerId, 'done', { id: experiment.id, status: 'failed', error: error instanceof Error ? error.message : 'Queue failed' });
    });
    res.json({ ok: true, experimentId: experiment.id, status: 'running', queued: true });
    return;
  }

  void trainExperiment(experiment).then(async (mlResult) => {
    if (experiment.status !== 'running') return;
    mlResult.artifacts = await persistArtifactCopies(experiment, mlResult);
    experiment.status = 'done';
    experiment.metrics = mlResult.metrics;
    experiment.artifacts = mlResult.artifacts;
    experiment.confusionMatrix = mlResult.confusion_matrix;
    experiment.perClass = mlResult.per_class;
    experiment.timings = mlResult.timings;
    persistState();
    emitToUser(experiment.ownerId, 'progress', { id: experiment.id, status: 'done', progress: 100, logs: ['Training completed successfully'] });
    emitToUser(experiment.ownerId, 'done', { id: experiment.id, status: 'done', metrics: experiment.metrics });
  }).catch((error) => {
    if (experiment.status !== 'running') return;
    experiment.status = 'failed';
    persistState();
    emitToUser(experiment.ownerId, 'done', { id: experiment.id, status: 'failed', error: error instanceof Error ? error.message : 'Training failed' });
  });

  res.json({ ok: true, experimentId: experiment.id, status: 'running' });
});

app.post('/experiments/:id/cancel', authMiddleware, (req: Request<{ id: string }>, res) => {
  const experiment = experiments.find((entry) => entry.id === req.params.id);
  if (!experiment || !canAccessResource(req, experiment)) {
    res.status(404).json({ message: 'Experiment not found' });
    return;
  }
  if (experiment.status !== 'running' && experiment.status !== 'queued') {
    res.status(409).json({ message: `Experiment cannot be cancelled from ${experiment.status} status` });
    return;
  }

  experiment.status = 'cancelled';
  persistState();
  if (experimentQueue) {
    void experimentQueue.getJob(experiment.id).then((job) => job?.remove()).catch((error) => {
      console.error('Could not remove BullMQ job', error);
    });
  }
  emitToUser(experiment.ownerId, 'done', {
    id: experiment.id,
    status: 'cancelled',
    message: 'Execution cancelled by the user'
  });
  res.json({ ok: true, experimentId: experiment.id, status: 'cancelled' });
});

app.get('/experiments/:id/metrics', authMiddleware, (req: Request<{ id: string }>, res) => {
  const experiment = experiments.find((entry) => entry.id === req.params.id);
  if (!experiment || !canAccessResource(req, experiment)) {
    res.status(404).json({ message: 'Experiment not found' });
    return;
  }
  res.json({
    experimentId: experiment.id,
    status: experiment.status,
    confusionMatrix: experiment.confusionMatrix,
    perClass: experiment.perClass,
    metrics: experiment.metrics ?? null,
  });
});

app.get('/experiments/:id/artifacts', authMiddleware, (req: Request<{ id: string }>, res) => {
  const experiment = experiments.find((entry) => entry.id === req.params.id);
  if (!experiment || !canAccessResource(req, experiment)) {
    res.status(404).json({ message: 'Experiment not found' });
    return;
  }
  res.json({
    experimentId: experiment.id,
    artifacts: experiment.artifacts ?? {},
  });
});

app.get('/experiments/:id/maps', authMiddleware, async (req: Request<{ id: string }>, res) => {
  const experiment = experiments.find((entry) => entry.id === req.params.id);
  if (!experiment || !canAccessResource(req, experiment)) {
    res.status(404).json({ message: 'Experiment not found' });
    return;
  }
  const truthPath = experiment.artifacts?.truth_map_path;
  const predictionPath = experiment.artifacts?.map_path;
  if (experiment.status !== 'done' || typeof truthPath !== 'string' || typeof predictionPath !== 'string') {
    res.status(409).json({ message: 'Held-out prediction maps are not available for this experiment' });
    return;
  }
  try {
    const mlUrl = process.env.ML_SERVICE_URL ?? 'http://localhost:8000';
    const response = await fetch(`${mlUrl}/render-maps`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ truthPath, predictionPath }),
    });
    const payload = await response.json() as Record<string, unknown>;
    if (!response.ok) {
      res.status(response.status).json({ message: payload.detail ?? 'Could not render experiment maps' });
      return;
    }
    res.json(payload);
  } catch (error) {
    res.status(502).json({
      message: 'Could not render experiment maps',
      detail: error instanceof Error ? error.message : 'Unknown error',
    });
  }
});

app.get('/compare', authMiddleware, (req, res) => {
  const ids = String(req.query.ids ?? '').split(',').map((id) => id.trim()).filter(Boolean);
  const selected = experiments.filter((experiment) => ids.includes(experiment.id) && canAccessResource(req, experiment));
  res.json({
    experiments: selected.map((experiment) => ({
      id: experiment.id,
      model: experiment.model,
      task: experiment.task,
      status: experiment.status,
      timings: experiment.timings ?? null,
      metrics: experiment.metrics ?? null,
    })),
  });
});

app.post('/experiments/:id/report', authMiddleware, (req: Request<{ id: string }>, res) => {
  const experiment = experiments.find((entry) => entry.id === req.params.id);
  if (!experiment || !canAccessResource(req, experiment)) {
    res.status(404).json({ message: 'Experiment not found' });
    return;
  }
  const dataset = datasets.find((entry) => entry.id === experiment.datasetId);
  const pipeline = pipelines.find((entry) => entry.id === experiment.pipelineId);
  const report = [
    '# HSI Studio Experiment Report',
    '',
    `## Experiment ${experiment.id}`,
    '',
    `- Dataset: ${dataset?.name ?? experiment.datasetId}`,
    `- Pipeline: ${pipeline?.name ?? experiment.pipelineId}`,
    `- Task: ${experiment.task}`,
    `- Model: ${experiment.model}`,
    `- Split: ${experiment.split.strategy} (${experiment.split.train}/${experiment.split.val}/${experiment.split.test})`,
    `- Seed: ${experiment.split.seed}`,
    `- Git SHA: ${experiment.gitSha}`,
    '',
    '## Results',
    '',
    '```json',
    JSON.stringify(experiment.metrics ?? {}, null, 2),
    '```',
    '',
    '## Limitations',
    '',
    '- Results are dataset and sensor specific.',
    '- Random or spatial split details must be verified before external interpretation.',
    '- Detection results are reported only for datasets with object annotations.',
  ].join('\n');

  res.json({
    format: 'markdown',
    filename: `${experiment.id}-report.md`,
    content: report,
  });
});

app.post('/experiments/:id/report/pdf', authMiddleware, async (req: Request<{ id: string }>, res) => {
  const experiment = experiments.find((entry) => entry.id === req.params.id);
  if (!experiment || !canAccessResource(req, experiment)) {
    res.status(404).json({ message: 'Experiment not found' });
    return;
  }
  const mlUrl = process.env.ML_SERVICE_URL ?? 'http://localhost:8000';
  try {
    const response = await fetch(`${mlUrl}/report/pdf`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({
        experimentId: experiment.id,
        dataset: experiment.datasetId,
        model: experiment.model,
        split: experiment.split,
        metrics: experiment.metrics ?? {},
      }),
    });
    if (!response.ok) {
      res.status(response.status).json({ message: 'PDF report service unavailable' });
      return;
    }
    res.setHeader('content-type', 'application/pdf');
    res.setHeader('content-disposition', `attachment; filename="${experiment.id}-report.pdf"`);
    const buffer = Buffer.from(await response.arrayBuffer());
    res.send(buffer);
  } catch (error) {
    res.status(502).json({ message: 'Could not reach ML report service', detail: error instanceof Error ? error.message : 'Unknown error' });
  }
});

app.get('/jobs/:id', authMiddleware, (req, res) => {
  const experiment = experiments.find((entry) => entry.id === req.params.id);
  if (!experiment || !canAccessResource(req, experiment)) {
    res.status(404).json({ message: 'Job not found' });
    return;
  }

  res.json({ id: experiment.id, status: experiment.status });
});

app.post('/experiments/:id/detect', authMiddleware, (req: Request<{ id: string }>, res) => {
  const experiment = experiments.find((entry) => entry.id === req.params.id);
  if (!experiment || !canAccessResource(req, experiment)) {
    res.status(404).json({ message: 'Experiment not found' });
    return;
  }
  const dataset = datasets.find((entry) => entry.id === experiment.datasetId);
  if (!dataset?.hasObjectAnnotations || !dataset.annotationsKey) {
    res.status(422).json({
      message: 'No objects detected: dataset has no object-level annotations',
      detectionSupported: false,
    });
    return;
  }
  const mlUrl = process.env.ML_SERVICE_URL ?? 'http://localhost:8000';
  void fetch(`${mlUrl}/detect`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({
      dataset: dataset.name,
      hasObjectAnnotations: true,
      confidenceThreshold: typeof req.body.confidenceThreshold === 'number' ? req.body.confidenceThreshold : 0.5,
      annotationsPath: dataset.annotationsKey,
      imagePath: typeof req.body.imagePath === 'string' ? req.body.imagePath : undefined,
      modelPath: typeof req.body.modelPath === 'string' ? req.body.modelPath : undefined,
    }),
  }).then(async (response) => {
    const result = await response.json() as { detail?: string };
    if (!response.ok) {
      res.status(response.status).json({ message: result.detail ?? `ML detect returned ${response.status}` });
      return;
    }
    res.json(result);
  }).catch((error) => {
    res.status(502).json({ message: 'ML detection failed', detail: error instanceof Error ? error.message : 'Unknown error' });
  });
});

io.on('connection', (socket) => {
  socket.emit('welcome', { service: 'hsi-studio-server' });
});

server.listen(port, () => {
  console.log(`HSI Studio server listening on http://localhost:${port}`);
});

export default app;
