import { BrowserRouter, NavLink, Route, Routes } from 'react-router-dom';
import { useEffect, useMemo, useRef, useState } from 'react';
import type { Dispatch, SetStateAction } from 'react';
import './App.css';
import { authApi, datasetApi, experimentApi, pipelineApi, systemApi } from './services/api';
import type { Dataset, Experiment, Pipeline } from './types';
import { io } from 'socket.io-client';
import jsPDF from 'jspdf';
import {
  Bar, BarChart, CartesianGrid, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis,
} from 'recharts';

const defaultDatasets: Dataset[] = [
  {
    id: 'ds-indian-pines',
    name: 'Indian Pines',
    sensor: 'AVIRIS',
    type: 'classification',
    hasObjectAnnotations: false,
    licence: 'Licence confirmation required before use',
    status: 'draft',
  },
  {
    id: 'ds-hod3k',
    name: 'HOD3K',
    sensor: 'AISA',
    type: 'detection',
    hasObjectAnnotations: false,
    licence: 'Licence confirmation required before use',
    status: 'draft',
  },
  {
    id: 'ds-pavia-university',
    name: 'Pavia University',
    sensor: 'ROSIS',
    type: 'classification',
    hasObjectAnnotations: false,
    licence: 'Licence confirmation required before use',
    status: 'draft',
  },
  {
    id: 'ds-salinas',
    name: 'Salinas',
    sensor: 'AVIRIS',
    type: 'classification',
    hasObjectAnnotations: false,
    licence: 'Licence confirmation required before use',
    status: 'draft',
  },
  {
    id: 'ds-hod1',
    name: 'HOD-1',
    sensor: 'Hyperspectral',
    type: 'detection',
    hasObjectAnnotations: false,
    licence: 'Licence confirmation required before use',
    status: 'draft',
  },
  {
    id: 'ds-m2sodai',
    name: 'M2SODAI',
    sensor: 'RGB + 224-band hyperspectral',
    type: 'detection',
    hasObjectAnnotations: false,
    licence: 'Dataset licence confirmation required',
    status: 'draft',
  },
];

const defaultPipelines: Pipeline[] = [
  {
    id: 'pipe-default',
    name: 'Baseline Spectral Pipeline',
    version: '1.0.0',
    steps: [
      { name: 'normalize', params: { method: 'minmax' } },
      { name: 'band-selection', params: { keep: 30 } },
      { name: 'pca', params: { components: 25, explainedVariance: 0.94 } },
    ],
  },
];

const defaultExperiments: Experiment[] = [
];

function App() {
  const [token, setToken] = useState<string | null>(() => localStorage.getItem('hsi-token'));
  const [datasets, setDatasets] = useState<Dataset[]>(defaultDatasets);
  const [pipelines, setPipelines] = useState<Pipeline[]>(defaultPipelines);
  const [experiments, setExperiments] = useState<Experiment[]>(defaultExperiments);
  const [appMessage, setAppMessage] = useState('');

  useEffect(() => {
    const expireSession = () => {
      setToken(null);
      setAppMessage('Your session expired. Sign in again to continue.');
    };
    window.addEventListener('hsi-auth-expired', expireSession);
    return () => window.removeEventListener('hsi-auth-expired', expireSession);
  }, []);

  useEffect(() => {
    if (token) {
      let active = true;
      Promise.all([
        datasetApi.list(),
        pipelineApi.list(),
        experimentApi.list(),
      ]).then(([datasetsResp, pipelinesResp, experimentsResp]) => {
        if (!active) return;
        setDatasets(datasetsResp.data.datasets ?? defaultDatasets);
        setPipelines(pipelinesResp.data.pipelines ?? defaultPipelines);
        setExperiments(experimentsResp.data.experiments ?? defaultExperiments);
        setAppMessage('');
      }).catch(() => {
        if (active) setAppMessage('Could not load saved project data. The visible catalogue may be sample data.');
      });
      return () => {
        active = false;
      };
    }
  }, [token]);

  return (
    <BrowserRouter>
      <div className="app-shell">
        <header className="topbar">
          <div className="topbar-brand">
            <span className="brand-mark" aria-hidden="true">H</span>
            <div className="brand-copy">
            <div className="eyebrow">PRISM / HSI Studio</div>
              <h1>Hyperspectral Studio</h1>
            </div>
          </div>
          <nav aria-label="Main navigation">
            <NavLink to="/">Dashboard</NavLink>
            <NavLink to="/datasets">Datasets</NavLink>
            <NavLink to="/inspector">Inspector</NavLink>
            <NavLink to="/preprocess">Prepare</NavLink>
            <NavLink to="/experiments">Experiments</NavLink>
            <NavLink to="/results">Results</NavLink>
            <NavLink to="/detection">Detection</NavLink>
            <NavLink to="/reports">Reports</NavLink>
            <NavLink to="/settings">Settings</NavLink>
          </nav>
          <div className="topbar-actions">
            {token ? (
              <button type="button" className="ghost-button signout-button" onClick={() => {
                localStorage.removeItem('hsi-token');
                localStorage.removeItem('hsi-refresh-token');
                setToken(null);
                setDatasets(defaultDatasets);
                setPipelines(defaultPipelines);
                setExperiments(defaultExperiments);
              }}>
                Sign out
              </button>
            ) : (
              <>
                <NavLink className="account-link" to="/register">Create account</NavLink>
                <button type="button" className="primary-button demo-button" onClick={() => handleDemoLogin(setToken, setAppMessage)}>
                  Demo login
                </button>
              </>
            )}
          </div>
        </header>

        {appMessage ? <p className="app-message" role="status">{appMessage}</p> : null}

        <main className="content-wrap">
          <Routes>
            <Route path="/" element={<DashboardPage datasets={datasets} experiments={experiments} />} />
            <Route path="/datasets" element={<DatasetPage datasets={datasets} setDatasets={setDatasets} />} />
            <Route path="/inspector" element={<InspectorPage datasets={datasets} setDatasets={setDatasets} />} />
            <Route path="/preprocess" element={<PreprocessPage datasets={datasets} pipelines={pipelines} setPipelines={setPipelines} />} />
            <Route path="/experiments" element={<ExperimentPage token={token} datasets={datasets} pipelines={pipelines} experiments={experiments} setExperiments={setExperiments} />} />
            <Route path="/results" element={<ResultsPage experiments={experiments} />} />
            <Route path="/detection" element={<DetectionPage experiments={experiments} />} />
            <Route path="/reports" element={<ReportsPage experiments={experiments} />} />
            <Route path="/settings" element={<SettingsPage token={token} />} />
            <Route path="/register" element={<RegisterPage setToken={setToken} />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}

function RegisterPage({ setToken }: { setToken: (token: string) => void }) {
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmation, setConfirmation] = useState('');
  const [error, setError] = useState('');

  const submit = () => {
    if (name.trim().length < 2) {
      setError('Name must contain at least 2 characters.');
      return;
    }
    if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
      setError('Enter a valid email address.');
      return;
    }
    if (password.length < 8) {
      setError('Password must contain at least 8 characters.');
      return;
    }
    if (password !== confirmation) {
      setError('Password confirmation does not match.');
      return;
    }
    authApi.register(name, email, password).then((response) => {
      localStorage.setItem('hsi-token', response.data.token);
      localStorage.setItem('hsi-refresh-token', response.data.refreshToken);
      setToken(response.data.token);
    }).catch((caught) => setError(caught.response?.data?.message ?? 'Registration failed.'));
  };

  return (
    <section className="panel form-panel" style={{ maxWidth: '560px', margin: '2rem auto' }}>
      <h2>Create account</h2>
      <div className="form-grid">
        <label>Name<input value={name} onChange={(event) => setName(event.target.value)} /></label>
        <label>Email<input type="email" value={email} onChange={(event) => setEmail(event.target.value)} /></label>
        <label>Password<input type="password" value={password} onChange={(event) => setPassword(event.target.value)} /></label>
        <label>Confirm password<input type="password" value={confirmation} onChange={(event) => setConfirmation(event.target.value)} /></label>
      </div>
      {error ? <p style={{ color: '#f0a080' }}>{error}</p> : null}
      <button type="button" className="primary-button" onClick={submit}>Create account</button>
    </section>
  );
}

function handleDemoLogin(setToken: (token: string) => void, setError: (message: string) => void) {
  authApi.login('demo@hsistudio.dev', 'password123')
    .then((response) => {
      const nextToken = response.data.token;
      localStorage.setItem('hsi-token', nextToken);
      localStorage.setItem('hsi-refresh-token', response.data.refreshToken);
      setToken(nextToken);
      setError('');
    })
    .catch(() => setError('Demo login failed. Confirm the API is running, then retry.'));
}

function DashboardPage({ datasets, experiments }: { datasets: Dataset[]; experiments: Experiment[] }) {
  const metrics = useMemo(() => {
    const ready = datasets.filter((dataset) => dataset.status === 'ready').length;
    const detection = datasets.filter((dataset) => dataset.type === 'detection').length;
    const classes = datasets.reduce((sum, dataset) => sum + Object.keys(dataset.qualityReport?.classHistogram ?? {}).length, 0);
    return { ready, detection, classes };
  }, [datasets]);

  return (
    <div className="stack">
      <section className="panel highlights">
        <div className="metric-card">
          <span>Total datasets</span>
          <strong>{datasets.length}</strong>
        </div>
        <div className="metric-card">
          <span>Ready for run</span>
          <strong>{metrics.ready}</strong>
        </div>
        <div className="metric-card">
          <span>Detection sets</span>
          <strong>{metrics.detection}</strong>
        </div>
        <div className="metric-card">
          <span>Tracked classes</span>
          <strong>{metrics.classes}</strong>
        </div>
      </section>

      <section className="grid-two">
        <article className="panel">
          <h2>Dataset library</h2>
          <ul className="list">
            {datasets.map((dataset) => (
              <li key={dataset.id}>
                <div>
                  <strong>{dataset.name}</strong>
                  <small>{dataset.sensor} • {dataset.type}</small>
                </div>
                <span className="badge">{dataset.status}</span>
              </li>
            ))}
          </ul>
        </article>

        <article className="panel">
          <h2>Recent experiments</h2>
          <ul className="list">
            {experiments.map((experiment) => (
              <li key={experiment.id}>
                <div>
                  <strong>{experiment.model.toUpperCase()}</strong>
                  <small>{experiment.task}</small>
                </div>
                <span className="badge status-badge">{experiment.status}</span>
              </li>
            ))}
          </ul>
        </article>
      </section>
    </div>
  );
}

function DatasetPage({ datasets, setDatasets }: { datasets: Dataset[]; setDatasets: Dispatch<SetStateAction<Dataset[]>>; }) {
  const [name, setName] = useState('');
  const [sensor, setSensor] = useState('AVIRIS');
  const [type, setType] = useState<'classification' | 'detection'>('classification');
  const [licence, setLicence] = useState('Licence confirmation required before use');
  const [uploadFiles, setUploadFiles] = useState<File[]>([]);
  const [message, setMessage] = useState('');
  const [busy, setBusy] = useState(false);

  const handleCreate = async () => {
    setBusy(true);
    setMessage('');
    try {
      const response = await datasetApi.create({
        name: name.trim() || uploadFiles[0]?.name || 'Custom cube',
        sensor,
        type,
        licence,
      });
      let created = response.data as Dataset;
      setDatasets((current) => [created, ...current]);
      if (uploadFiles.length) {
        const uploaded = await datasetApi.upload(uploadFiles, created.id);
        created = { ...created, storageKey: uploaded.data.storageKey, format: uploaded.data.format };
        setDatasets((current) => current.map((dataset) => dataset.id === created.id ? created : dataset));
      }
      setName('');
      setUploadFiles([]);
      setMessage(uploadFiles.length ? 'Dataset and cube uploaded.' : 'Dataset created. Upload its cube and labels before training.');
    } catch (caught) {
      setMessage(caught instanceof Error ? caught.message : 'Could not create or upload the dataset.');
    } finally {
      setBusy(false);
    }
  };

  const handleInspect = async (id: string) => {
    setBusy(true);
    setMessage('');
    try {
      const response = await datasetApi.inspect(id);
      setDatasets((current) => current.map((dataset) => dataset.id === id
        ? { ...dataset, status: 'inspected', qualityReport: response.data.shape ? {
            height: response.data.shape.height,
            width: response.data.shape.width,
            bands: response.data.shape.bands,
            nanCount: response.data.nanCount,
            noisyBands: response.data.noisyBands,
            classHistogram: response.data.classHistogram,
          } : dataset.qualityReport }
        : dataset));
      setMessage('Dataset inspection completed.');
    } catch (caught) {
      setMessage(caught instanceof Error ? caught.message : 'Dataset inspection failed.');
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="stack">
      <section className="panel form-panel">
        <h2>Create dataset</h2>
        <div className="form-grid">
          <label>
            Name
            <input value={name} onChange={(event) => setName(event.target.value)} placeholder="e.g. WHU-Longkou" />
          </label>
          <label>
            Sensor
            <input value={sensor} onChange={(event) => setSensor(event.target.value)} />
          </label>
          <label>
            Type
            <select value={type} onChange={(event) => setType(event.target.value as 'classification' | 'detection')}>
              <option value="classification">Classification</option>
              <option value="detection">Detection</option>
            </select>
          </label>
          <label>
            Licence
            <input value={licence} onChange={(event) => setLicence(event.target.value)} />
          </label>
        </div>
        <button type="button" className="primary-button" onClick={handleCreate} disabled={busy}>Add dataset</button>
        <label>
          Upload cube (.mat / ENVI .hdr+.img / GeoTIFF)
          <input type="file" multiple accept=".mat,.hdr,.img,.tif,.tiff,.npy,.npz" onChange={(event) => setUploadFiles(Array.from(event.target.files ?? []))} />
        </label>
        <button
          type="button"
          className="ghost-button"
          disabled={!uploadFiles.length}
          onClick={() => {
            if (!uploadFiles.length) return;
            void handleCreate();
          }}
        >
          Upload cube
        </button>
        {message ? <p role="status">{message}</p> : null}
      </section>

      <section className="panel">
        <h2>Catalog</h2>
        <ul className="dataset-list">
          {datasets.map((dataset) => (
            <li key={dataset.id} className="dataset-item">
              <div>
                <h3>{dataset.name}</h3>
                <p>{dataset.sensor} • {dataset.type}</p>
                <p>{dataset.licence}</p>
              </div>
              <div className="meta-col">
                <span className="badge">{dataset.status}</span>
                {dataset.hasObjectAnnotations ? <span className="badge detection">object annotations</span> : null}
                {dataset.status === 'draft' ? (
                  <button type="button" className="ghost-button" onClick={() => void handleInspect(dataset.id)} disabled={busy}>Inspect</button>
                ) : null}
                {dataset.source === 'upload' && dataset.type === 'detection' ? (
                  <label>
                    COCO annotations (.json)
                    <input type="file" accept=".json" onChange={(event) => {
                      const file = event.target.files?.[0];
                      if (file) datasetApi.uploadAnnotations(dataset.id, file).then((response) => {
                        setDatasets((current) => current.map((entry) => entry.id === dataset.id
                          ? { ...entry, hasObjectAnnotations: response.data.hasObjectAnnotations, annotationsKey: response.data.annotationsKey }
                          : entry));
                        setMessage('Object annotations uploaded and validated.');
                      }).catch((caught) => setMessage(caught instanceof Error ? caught.message : 'Annotation upload failed.'));
                    }} />
                  </label>
                ) : dataset.source === 'upload' ? (
                  <label>
                    Pixel label map
                    <input type="file" accept=".mat,.npy,.npz,.tif,.tiff,.json" onChange={(event) => {
                      const file = event.target.files?.[0];
                      if (file) datasetApi.uploadLabels(dataset.id, file).then((response) => {
                        setDatasets((current) => current.map((entry) => entry.id === dataset.id
                          ? { ...entry, labelsKey: response.data.labelsKey }
                          : entry));
                        setMessage('Label map uploaded.');
                      }).catch((caught) => setMessage(caught instanceof Error ? caught.message : 'Label upload failed.'));
                    }} />
                  </label>
                ) : null}
              </div>
            </li>
          ))}
        </ul>
      </section>
    </div>
  );
}

function PreprocessPage({ datasets, pipelines, setPipelines }: { datasets: Dataset[]; pipelines: Pipeline[]; setPipelines: Dispatch<SetStateAction<Pipeline[]>>; }) {
  const [name, setName] = useState('Spectral denoise + PCA');
  const [version, setVersion] = useState('1.0.1');
  const [datasetId, setDatasetId] = useState(datasets[0]?.id ?? '');
  const [normalize, setNormalize] = useState(true);
  const [removeNoisyBands, setRemoveNoisyBands] = useState(true);
  const [components, setComponents] = useState(25);
  const [preprocessPreview, setPreprocessPreview] = useState<Record<string, unknown> | null>(null);
  const [message, setMessage] = useState('');
  const activeDatasetId = datasets.some((dataset) => dataset.id === datasetId) ? datasetId : datasets[0]?.id ?? '';

  const handleCreate = async () => {
    const next: Pipeline = {
      id: `pipe-${Date.now()}`,
      name,
      version,
      steps: [
        ...(normalize ? [{ name: 'normalize', params: { method: 'minmax' } }] : []),
        ...(removeNoisyBands ? [{ name: 'denoise', params: { threshold: 0.03 } }] : []),
        { name: 'pca', params: { components, explainedVariance: 0.96 } },
      ],
    };

    setMessage('');
    try {
      const response = await pipelineApi.create({
        name: next.name,
        version: next.version,
        steps: next.steps,
      });
      setPipelines((current) => [response.data, ...current]);
    } catch {
      setMessage('Could not save this pipeline. Check the API connection and retry.');
    }
  };

  return (
    <div className="stack">
      <section className="panel form-panel">
        <h2>Preprocessing studio</h2>
        <div className="form-grid">
          <label>
            Dataset
            <select value={activeDatasetId} onChange={(event) => setDatasetId(event.target.value)}>
              {datasets.map((dataset) => <option key={dataset.id} value={dataset.id}>{dataset.name}</option>)}
            </select>
          </label>
          <label>
            Pipeline name
            <input value={name} onChange={(event) => setName(event.target.value)} />
          </label>
          <label>
            Version
            <input value={version} onChange={(event) => setVersion(event.target.value)} />
          </label>
          <label>
            <span><input type="checkbox" checked={normalize} onChange={(event) => setNormalize(event.target.checked)} /> Normalize</span>
          </label>
          <label>
            <span><input type="checkbox" checked={removeNoisyBands} onChange={(event) => setRemoveNoisyBands(event.target.checked)} /> Remove noisy bands</span>
          </label>
          <label>
            PCA components: {components}
            <input type="range" min="2" max="100" value={components} onChange={(event) => setComponents(Number(event.target.value))} />
          </label>
        </div>
        <button
          type="button"
          className="ghost-button"
          onClick={() => {
            pipelineApi.preprocess({
              datasetId: activeDatasetId,
              normalize,
              remove_noisy_bands: removeNoisyBands,
              pca_components: components,
            }).then((response) => {
              setPreprocessPreview(response.data);
              setMessage('');
            }).catch(() => {
              setPreprocessPreview(null);
              setMessage('Preprocessing preview failed. Verify the selected dataset and ML service.');
            });
          }}
        >
          Preview preprocessing
        </button>
        {preprocessPreview ? (
          <p>
            Explained variance: {Number((preprocessPreview.pca as { explainedVariance?: number })?.explainedVariance ?? 0).toFixed(4)}
            {' · '}Cached: {String(preprocessPreview.cached)}
          </p>
        ) : null}
        <button type="button" className="primary-button" onClick={() => void handleCreate()}>Save pipeline</button>
        {message ? <p role="status">{message}</p> : null}
      </section>

      <section className="panel">
        <h2>Pipeline list</h2>
        {pipelines.map((pipeline) => (
          <div key={pipeline.id} className="pipeline-item">
            <div>
              <h3>{pipeline.name}</h3>
              <small>v{pipeline.version}</small>
            </div>
            <ul>
              {pipeline.steps.map((step) => (
                <li key={`${pipeline.id}-${step.name}`}>{step.name}</li>
              ))}
            </ul>
          </div>
        ))}
      </section>
    </div>
  );
}

function ExperimentPage({ token, datasets, pipelines, experiments, setExperiments }: { token: string | null; datasets: Dataset[]; pipelines: Pipeline[]; experiments: Experiment[]; setExperiments: Dispatch<SetStateAction<Experiment[]>>; }) {
  const [datasetId, setDatasetId] = useState(datasets[0]?.id ?? '');
  const [pipelineId, setPipelineId] = useState(pipelines[0]?.id ?? '');
  const [model, setModel] = useState<'svm' | 'cnn2d' | 'cnn3d' | 'yolo'>('svm');
  const [strategy, setStrategy] = useState<'spatial'>('spatial');
  const [trainSplit, setTrainSplit] = useState(60);
  const [valSplit, setValSplit] = useState(20);
  const [seed, setSeed] = useState(42);
  const [jobEvent, setJobEvent] = useState<{ type: string; payload: unknown } | null>(null);
  const [jobUpdates, setJobUpdates] = useState<Record<string, { progress: number; logs: string[] }>>({});
  const [message, setMessage] = useState('');
  const activeDatasetId = datasets.some((dataset) => dataset.id === datasetId) ? datasetId : datasets[0]?.id ?? '';
  const activePipelineId = pipelines.some((pipeline) => pipeline.id === pipelineId) ? pipelineId : pipelines[0]?.id ?? '';
  const selectedDataset = datasets.find((dataset) => dataset.id === activeDatasetId);
  useJobSocket(token, (event) => {
    setJobEvent(event);
    if (typeof event.payload !== 'object' || event.payload === null) return;
    const payload = event.payload as { id?: unknown; status?: unknown; metrics?: unknown; error?: unknown; message?: unknown };
    if (typeof payload.id === 'string') {
      const progress = payload as { progress?: unknown; logs?: unknown };
      setJobUpdates((current) => {
        const previous = current[payload.id as string] ?? { progress: 0, logs: [] };
        const incomingLogs = Array.isArray(progress.logs)
          ? progress.logs.filter((log): log is string => typeof log === 'string')
          : typeof payload.message === 'string' ? [payload.message] : [];
        return {
          ...current,
          [payload.id as string]: {
            progress: typeof progress.progress === 'number' ? progress.progress : previous.progress,
            logs: [...previous.logs, ...incomingLogs].slice(-8),
          },
        };
      });
    }
    if (typeof payload.id !== 'string' || typeof payload.status !== 'string'
      || !['queued', 'running', 'done', 'failed', 'cancelled'].includes(payload.status)) return;
    if (typeof payload.error === 'string') setMessage(payload.error);
    else if (typeof payload.message === 'string' && payload.status === 'cancelled') setMessage(payload.message);
    setExperiments((current) => current.map((experiment) => experiment.id === payload.id
      ? {
          ...experiment,
          status: payload.status as Experiment['status'],
          ...(typeof payload.metrics === 'object' && payload.metrics !== null
            ? { metrics: payload.metrics as Record<string, number> }
            : {}),
        }
      : experiment));
  });

  const handleCreate = async () => {
    const testSplit = 100 - trainSplit - valSplit;
    if (!activeDatasetId || !activePipelineId || testSplit < 10) {
      setMessage('Select a dataset and pipeline, and leave at least 10% for testing.');
      return;
    }
    const next: Experiment = {
      id: `exp-${Date.now()}`,
      datasetId: activeDatasetId,
      pipelineId: activePipelineId,
      task: selectedDataset?.type ?? 'classification',
      model,
      hyperparameters: { C: 10, epochs: 25 },
      status: 'queued',
      split: { strategy: 'spatial', train: 0.6, val: 0.2, test: 0.2, seed: 42 },
      gitSha: '',
    };

    next.split = {
      strategy,
      train: trainSplit / 100,
      val: valSplit / 100,
      test: testSplit / 100,
      seed,
    };

    setMessage('');
    try {
      const response = await experimentApi.create({
        datasetId: next.datasetId,
        pipelineId: next.pipelineId,
        task: next.task,
        model: next.model,
        hyperparameters: next.hyperparameters,
        split: next.split,
      });
      setExperiments((current) => [response.data, ...current]);
    } catch {
      setMessage('Could not create the experiment. Check the dataset and pipeline, then retry.');
    }
  };

  const handleRun = (id: string) => {
    setMessage('');
    experimentApi.run(id).then((response) => {
      setExperiments((current) => current.map((experiment) =>
        experiment.id === id && experiment.status === 'queued'
          ? { ...experiment, status: response.data.status }
          : experiment
      ));
    }).catch((caught) => {
      const error = caught as { response?: { data?: { message?: string } } };
      setMessage(error.response?.data?.message ?? 'Experiment could not start. Check its dataset, labels, and service status.');
    });
  };

  const handleCancel = (id: string) => {
    experimentApi.cancel(id).then((response) => {
      setExperiments((current) => current.map((experiment) =>
        experiment.id === id ? { ...experiment, status: response.data.status } : experiment
      ));
    }).catch(() => setMessage('Could not cancel this experiment.'));
  };

  return (
    <div className="stack">
      <section className="panel form-panel">
        <h2>Experiment builder</h2>
        <div className="form-grid">
          <label>
            Dataset
            <select value={activeDatasetId} onChange={(event) => {
              const nextId = event.target.value;
              const nextDataset = datasets.find((dataset) => dataset.id === nextId);
              setDatasetId(nextId);
              if (nextDataset?.type === 'detection') setModel('yolo');
              else if (model === 'yolo') setModel('svm');
            }}>
              {datasets.map((dataset) => (
                <option key={dataset.id} value={dataset.id}>{dataset.name}</option>
              ))}
            </select>
          </label>
          <label>
            Pipeline
            <select value={activePipelineId} onChange={(event) => setPipelineId(event.target.value)}>
              {pipelines.map((pipeline) => (
                <option key={pipeline.id} value={pipeline.id}>{pipeline.name}</option>
              ))}
            </select>
          </label>
          <label>
            Model
            <select value={model} onChange={(event) => setModel(event.target.value as 'svm' | 'cnn2d' | 'cnn3d' | 'yolo')}>
              <option value="svm" disabled={selectedDataset?.type === 'detection'}>SVM</option>
              <option value="cnn2d" disabled={selectedDataset?.type === 'detection'}>2D CNN</option>
              <option value="cnn3d" disabled={selectedDataset?.type === 'detection'}>3D CNN</option>
              <option value="yolo" disabled={selectedDataset?.type !== 'detection'}>YOLO</option>
            </select>
          </label>
          <label>
            Split strategy
            <select value={strategy} onChange={(event) => setStrategy(event.target.value as 'spatial')}>
              <option value="spatial">Spatial</option>
            </select>
          </label>
          <label>
            Train split (%)
            <input type="number" min="10" max="90" value={trainSplit} onChange={(event) => setTrainSplit(Number(event.target.value))} />
          </label>
          <label>
            Validation split (%)
            <input type="number" min="5" max="40" value={valSplit} onChange={(event) => setValSplit(Number(event.target.value))} />
          </label>
          <label>
            Seed
            <input type="number" value={seed} onChange={(event) => setSeed(Number(event.target.value))} />
          </label>
        </div>
        <button type="button" className="primary-button" onClick={() => void handleCreate()}>Create experiment</button>
        {message ? <p role="status">{message}</p> : null}
      </section>

      <section className="panel">
        <h2>Run queue</h2>
        {jobEvent ? <p className="badge">Live event: {jobEvent.type}</p> : null}
        {!experiments.length ? <p>No experiments are saved yet.</p> : null}
        {experiments.map((experiment) => (
          <div key={experiment.id} className="experiment-row">
            <div>
              <strong>{experiment.model.toUpperCase()}</strong>
              <small>{experiment.task}</small>
            </div>
            <span className="badge status-badge">{experiment.status}</span>
            {experiment.status === 'running' ? (
              <div className="job-monitor" aria-live="polite">
                <progress max="100" value={jobUpdates[experiment.id]?.progress ?? 0} />
                <small>{jobUpdates[experiment.id]?.progress ?? 0}%</small>
                <ul>{(jobUpdates[experiment.id]?.logs ?? []).map((log, index) => <li key={`${log}-${index}`}>{log}</li>)}</ul>
              </div>
            ) : null}
            {experiment.status === 'running' ? (
              <button type="button" className="ghost-button" onClick={() => handleCancel(experiment.id)}>Cancel</button>
            ) : experiment.status === 'queued' ? (
              <button type="button" className="ghost-button" onClick={() => handleRun(experiment.id)}>Run</button>
            ) : null}
          </div>
        ))}
      </section>
    </div>
  );
}

function useJobSocket(token: string | null, onEvent: (event: { type: string; payload: unknown }) => void) {
  const callbackRef = useRef(onEvent);
  useEffect(() => {
    callbackRef.current = onEvent;
  }, [onEvent]);
  useEffect(() => {
    if (!token) return;
    const socket = io(import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:4000', {
      auth: (callback) => callback({ token: localStorage.getItem('hsi-token') }),
    });
    socket.on('progress', (payload) => callbackRef.current({ type: 'progress', payload }));
    socket.on('log', (payload) => callbackRef.current({ type: 'log', payload }));
    socket.on('done', (payload) => callbackRef.current({ type: 'done', payload }));
    return () => {
      socket.disconnect();
    };
  }, [token]);
}

function InspectorPage({ datasets, setDatasets }: { datasets: Dataset[]; setDatasets: Dispatch<SetStateAction<Dataset[]>>; }) {
  const [selectedId, setSelectedId] = useState(datasets[0]?.id ?? '');
  const activeSelectedId = datasets.some((dataset) => dataset.id === selectedId) ? selectedId : datasets[0]?.id ?? '';
  const [redBand, setRedBand] = useState(0);
  const [greenBand, setGreenBand] = useState(1);
  const [blueBand, setBlueBand] = useState(2);
  const [pixel, setPixel] = useState({ x: 0, y: 0 });
  const [visual, setVisual] = useState<{
    datasetId: string;
    width: number;
    height: number;
    bands: number;
    rgbBase64: string;
    spectrum: Array<number | null>;
  } | null>(null);
  const [message, setMessage] = useState('');
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const selected = datasets.find((dataset) => dataset.id === activeSelectedId);
  const bandCount = selected?.qualityReport?.bands ?? 0;
  const activeVisual = selected?.storageKey && bandCount > 0 && visual?.datasetId === activeSelectedId ? visual : null;
  const channelMaximum = Math.max(0, bandCount - 1);
  const activeRed = Math.min(redBand, channelMaximum);
  const activeGreen = Math.min(greenBand, channelMaximum);
  const activeBlue = Math.min(blueBand, channelMaximum);

  const inspectSelected = async () => {
    if (!activeSelectedId) return;
    setMessage('');
    try {
      const response = await datasetApi.inspect(activeSelectedId);
      setDatasets((current) => current.map((dataset) => dataset.id === activeSelectedId
        ? {
            ...dataset,
            status: 'inspected',
            qualityReport: {
              height: response.data.shape.height,
              width: response.data.shape.width,
              bands: response.data.shape.bands,
              nanCount: response.data.nanCount,
              noisyBands: response.data.noisyBands,
              classHistogram: response.data.classHistogram,
            },
          }
        : dataset));
      setPixel({ x: 0, y: 0 });
    } catch (error) {
      setMessage(error instanceof Error ? error.message : 'Cube inspection failed.');
    }
  };

  useEffect(() => {
    if (!activeSelectedId || !selected?.storageKey || bandCount === 0) {
      return;
    }
    let active = true;
    datasetApi.visualize(activeSelectedId, {
      redBand: activeRed,
      greenBand: activeGreen,
      blueBand: activeBlue,
      x: pixel.x,
      y: pixel.y,
    }).then((response) => {
      if (!active) return;
      setVisual({ ...response.data, datasetId: activeSelectedId });
      setMessage('');
    }).catch((error: { response?: { data?: { message?: string } } }) => {
      if (active) {
        setMessage(error.response?.data?.message ?? 'Could not render this cube.');
      }
    });
    return () => {
      active = false;
    };
  }, [activeSelectedId, selected?.storageKey, bandCount, activeRed, activeGreen, activeBlue, pixel.x, pixel.y]);

  useEffect(() => {
    if (!canvasRef.current) return;
    const canvas = canvasRef.current;
    const context = canvas.getContext('2d');
    if (!context) return;
    if (!activeVisual) {
      context.clearRect(0, 0, canvas.width, canvas.height);
      return;
    }
    const bytes = Uint8Array.from(atob(activeVisual.rgbBase64), (character) => character.charCodeAt(0));
    canvas.width = activeVisual.width;
    canvas.height = activeVisual.height;
    context.putImageData(new ImageData(new Uint8ClampedArray(bytes), activeVisual.width, activeVisual.height), 0, 0);
  }, [activeVisual]);

  const finiteSpectrum = activeVisual?.spectrum.filter((value): value is number => value !== null && Number.isFinite(value)) ?? [];
  const spectrumMin = finiteSpectrum.length ? Math.min(...finiteSpectrum) : 0;
  const spectrumRange = finiteSpectrum.length ? Math.max(...finiteSpectrum) - spectrumMin : 0;

  return (
    <div className="stack">
      <section className="panel form-panel">
        <h2>Dataset inspector</h2>
        <label>
          Dataset
          <select value={activeSelectedId} onChange={(event) => setSelectedId(event.target.value)}>
            {datasets.map((dataset) => <option key={dataset.id} value={dataset.id}>{dataset.name}</option>)}
          </select>
        </label>
        <button type="button" className="ghost-button" onClick={() => void inspectSelected()} disabled={!selected?.storageKey}>
          Inspect uploaded cube
        </button>
        {message ? <p role="status">{message}</p> : null}
        {selected?.qualityReport ? (
          <div className="form-grid">
            <div className="metric-card"><span>Dimensions</span><strong>{selected.qualityReport.height} × {selected.qualityReport.width}</strong></div>
            <div className="metric-card"><span>Bands</span><strong>{selected.qualityReport.bands}</strong></div>
            <div className="metric-card"><span>NaN / invalid</span><strong>{selected.qualityReport.nanCount}</strong></div>
            <div className="metric-card"><span>Noisy bands</span><strong>{selected.qualityReport.noisyBands.join(', ') || 'None'}</strong></div>
          </div>
        ) : null}
        {selected?.qualityReport?.classHistogram ? (
          <ul className="list">
            {Object.entries(selected.qualityReport.classHistogram).map(([label, count]) => (
              <li key={label}><span>Class {label}</span><strong>{count}</strong></li>
            ))}
          </ul>
        ) : null}
      </section>
      <section className="grid-two">
        <article className="panel">
          <h2>False-colour viewer</h2>
          <canvas
            ref={canvasRef}
            className="false-colour-canvas"
            aria-label="False-colour image"
            onClick={(event) => {
              if (!activeVisual) return;
              const rect = event.currentTarget.getBoundingClientRect();
              setPixel({
                x: Math.min(activeVisual.width - 1, Math.floor((event.clientX - rect.left) / rect.width * activeVisual.width)),
                y: Math.min(activeVisual.height - 1, Math.floor((event.clientY - rect.top) / rect.height * activeVisual.height)),
              });
            }}
          />
          {!activeVisual && !message ? <p>Upload and inspect a real cube to render its selected spectral bands.</p> : null}
          <div className="form-grid">
            <label>Red band ({activeRed}) <input type="range" min="0" max={channelMaximum} value={activeRed} disabled={!bandCount} onChange={(event) => setRedBand(Number(event.target.value))} /></label>
            <label>Green band ({activeGreen}) <input type="range" min="0" max={channelMaximum} value={activeGreen} disabled={!bandCount} onChange={(event) => setGreenBand(Number(event.target.value))} /></label>
            <label>Blue band ({activeBlue}) <input type="range" min="0" max={channelMaximum} value={activeBlue} disabled={!bandCount} onChange={(event) => setBlueBand(Number(event.target.value))} /></label>
          </div>
        </article>
        <article className="panel">
          <h2>Spectral signature — pixel ({Math.round(pixel.x)}, {Math.round(pixel.y)})</h2>
          <div className="spectrum-plot">
            {activeVisual?.spectrum.map((value, index) => (
              <span
                key={index}
                title={`Band ${index}: ${value === null ? 'invalid' : value.toFixed(5)}`}
                style={{ height: `${value === null || spectrumRange === 0 ? 0 : (value - spectrumMin) / spectrumRange * 180}px` }}
              />
            ))}
          </div>
          {!activeVisual ? <p>Select an inspected dataset with an uploaded cube to sample a pixel.</p> : null}
        </article>
      </section>
    </div>
  );
}

function ResultsPage({ experiments }: { experiments: Experiment[] }) {
  const [selectedId, setSelectedId] = useState(experiments[0]?.id ?? '');
  const [metrics, setMetrics] = useState<Record<string, unknown> | null>(null);
  const [confusionMatrix, setConfusionMatrix] = useState<number[][] | null>(null);
  const [perClass, setPerClass] = useState<Record<string, number[]> | null>(null);
  const [maps, setMaps] = useState<{
    experimentId: string;
    truth: string;
    prediction: string;
    error: string;
    legend: Array<{ classId: number; color: string }>;
  } | null>(null);
  const [comparison, setComparison] = useState<Array<{ id: string; model: string; metrics: Record<string, number> | null }>>([]);
  const [message, setMessage] = useState('');
  const activeSelectedId = experiments.some((experiment) => experiment.id === selectedId) ? selectedId : experiments[0]?.id ?? '';
  const selectedExperiment = experiments.find((experiment) => experiment.id === activeSelectedId);

  useEffect(() => {
    if (!activeSelectedId) return;
    experimentApi.metrics(activeSelectedId).then((response) => {
      setMetrics(response.data.metrics);
      setConfusionMatrix(response.data.confusionMatrix ?? null);
      setPerClass(response.data.perClass ?? null);
      setMessage('');
    }).catch(() => {
      setMetrics(null);
      setConfusionMatrix(null);
      setPerClass(null);
      setMessage('Could not load experiment metrics.');
    });
  }, [activeSelectedId]);

  useEffect(() => {
    if (!activeSelectedId || selectedExperiment?.status !== 'done') return;
    let active = true;
    experimentApi.maps(activeSelectedId).then((response) => {
      if (active) setMaps({ ...response.data, experimentId: activeSelectedId });
    }).catch(() => {
      if (active) setMaps(null);
    });
    return () => {
      active = false;
    };
  }, [activeSelectedId, selectedExperiment?.status]);

  const handleCompare = () => {
    const ids = experiments.slice(0, 3).map((experiment) => experiment.id);
    experimentApi.compare(ids).then((response) => {
      setComparison(response.data.experiments);
      setMessage('');
    }).catch(() => setMessage('Could not compare experiments.'));
  };

  const values = metrics as Record<string, number> | null;
  const classScores = (perClass?.f1 ?? []).map((score, index) => ({
    className: `Class ${index + 1}`,
    f1: score,
  }));
  const timingAccuracy = experiments.flatMap((experiment) => {
    const accuracy = experiment.metrics?.OA;
    const trainingSeconds = experiment.timings?.train;
    return experiment.status === 'done' && typeof accuracy === 'number' && typeof trainingSeconds === 'number'
      ? [{ experiment: experiment.id, accuracy: accuracy * 100, trainingSeconds }]
      : [];
  });
  return (
    <div className="stack">
      <section className="panel form-panel">
        <h2>Results and comparison</h2>
        {experiments.length ? (
          <>
            <div className="form-grid">
              <label>
                Experiment
                <select value={activeSelectedId} onChange={(event) => setSelectedId(event.target.value)}>
                  {experiments.map((experiment) => (
                    <option key={experiment.id} value={experiment.id}>{experiment.id} — {experiment.model}</option>
                  ))}
                </select>
              </label>
            </div>
            <button type="button" className="primary-button" onClick={handleCompare}>Compare recent experiments</button>
          </>
        ) : <p>No experiments are saved yet. Create and run one to view real results.</p>}
        {message ? <p role="status">{message}</p> : null}
      </section>

      <section className="grid-two">
        <article className="panel">
          <h2>Held-out metrics</h2>
          {values ? (
            <div className="metric-grid">
              {Object.entries(values).map(([name, value]) => (
                <div className="metric-card" key={name}><span>{name}</span><strong>{Number(value).toFixed(3)}</strong></div>
              ))}
            </div>
          ) : <p>No metrics available.</p>}
        </article>
        <article className="panel">
          <h2>Experiment comparison</h2>
          {comparison.length ? (
            <ul className="list">
              {comparison.map((item) => (
                <li key={item.id}>
                  <div><strong>{item.model.toUpperCase()}</strong><small>{item.id}</small></div>
                  <span className="badge">
                    {item.metrics ? Number(item.metrics.OA ?? item.metrics.mAP ?? 0).toFixed(3) : 'No metrics'}
                  </span>
                </li>
              ))}
            </ul>
          ) : <p>Select comparison to inspect side-by-side results.</p>}
        </article>
      </section>

      {confusionMatrix ? (
        <section className="panel">
          <h2>Confusion matrix</h2>
          <div className="confusion-matrix" role="table" aria-label="Confusion matrix">
            {confusionMatrix.map((row, rowIndex) => (
              <div className="confusion-row" role="row" key={rowIndex}>
                {row.map((value, columnIndex) => (
                  <span role="cell" key={columnIndex} style={{ background: `rgba(224, 144, 95, ${0.15 + value / 100})` }}>{value}</span>
                ))}
              </div>
            ))}
          </div>
        </section>
      ) : null}

      {classScores.length ? (
        <section className="panel">
          <h2>Per-class F1 score</h2>
          <div style={{ width: '100%', height: 280 }}>
            <ResponsiveContainer>
              <BarChart data={classScores}>
                <CartesianGrid strokeDasharray="3 3" stroke="#283238" />
                <XAxis dataKey="className" stroke="#aab5b9" />
                <YAxis domain={[0, 1]} stroke="#aab5b9" />
                <Tooltip />
                <Bar dataKey="f1" fill="#39c5ac" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </section>
      ) : null}

      {timingAccuracy.length ? (
        <section className="panel">
          <h2>Training time vs. held-out accuracy</h2>
          <div style={{ width: '100%', height: 280 }}>
            <ResponsiveContainer>
              <ScatterChart>
                <CartesianGrid strokeDasharray="3 3" stroke="#283238" />
                <XAxis type="number" dataKey="accuracy" name="Accuracy" unit="%" stroke="#aab5b9" />
                <YAxis type="number" dataKey="trainingSeconds" name="Training time" unit="s" stroke="#aab5b9" />
                <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                <Scatter data={timingAccuracy} fill="#e0905f" />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        </section>
      ) : null}

      {selectedExperiment?.timings ? (
        <p>Selected run: training {selectedExperiment.timings.train.toFixed(2)} s; inference {selectedExperiment.timings.infer.toFixed(2)} s.</p>
      ) : null}

      {maps?.experimentId === activeSelectedId && selectedExperiment?.status === 'done'
        ? <PredictionMapViewer maps={maps} />
        : null}

      <section className="panel">
        <h2>Limitation note</h2>
        <p>These values describe held-out performance for this dataset and sensor only. They must not be assumed to generalise across locations or conditions.</p>
      </section>
    </div>
  );
}

function PredictionMapViewer({ maps }: {
  maps: {
    experimentId: string;
    truth: string;
    prediction: string;
    error: string;
    legend: Array<{ classId: number; color: string }>;
  };
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [opacity, setOpacity] = useState(55);
  const [layer, setLayer] = useState<'overlay' | 'error'>('overlay');
  const [showLegend, setShowLegend] = useState(true);

  useEffect(() => {
    const canvas = canvasRef.current;
    const context = canvas?.getContext('2d');
    if (!canvas || !context) return;
    let active = true;
    const loadImage = (source: string) => new Promise<HTMLImageElement>((resolve, reject) => {
      const image = new Image();
      image.onload = () => resolve(image);
      image.onerror = () => reject(new Error('Could not load a prediction map'));
      image.src = `data:image/png;base64,${source}`;
    });
    Promise.all([loadImage(maps.truth), loadImage(layer === 'error' ? maps.error : maps.prediction)]).then(([truth, overlay]) => {
      if (!active) return;
      canvas.width = truth.naturalWidth;
      canvas.height = truth.naturalHeight;
      context.clearRect(0, 0, canvas.width, canvas.height);
      context.globalAlpha = 1;
      context.drawImage(truth, 0, 0);
      context.globalAlpha = layer === 'error' ? 1 : opacity / 100;
      context.drawImage(overlay, 0, 0);
      context.globalAlpha = 1;
    });
    return () => {
      active = false;
    };
  }, [maps, layer, opacity]);

  return (
    <section className="panel">
      <h2>Held-out prediction viewer</h2>
      <div className="form-grid">
        <label>
          Map layer
          <select value={layer} onChange={(event) => setLayer(event.target.value as 'overlay' | 'error')}>
            <option value="overlay">Ground truth with prediction overlay</option>
            <option value="error">Correct / incorrect pixels</option>
          </select>
        </label>
        {layer === 'overlay' ? (
          <label>
            Prediction opacity ({opacity}%)
            <input type="range" min="0" max="100" value={opacity} onChange={(event) => setOpacity(Number(event.target.value))} />
          </label>
        ) : null}
        <label className="checkbox-label">
          <input type="checkbox" checked={showLegend} onChange={(event) => setShowLegend(event.target.checked)} />
          Show legend
        </label>
      </div>
      <canvas ref={canvasRef} className="prediction-map-canvas" aria-label="Held-out ground truth and prediction map" />
      {showLegend ? (
        <ul className="map-legend">
          {layer === 'error' ? (
            <>
              <li><span style={{ backgroundColor: '#5f7a55' }} />Correct</li>
              <li><span style={{ backgroundColor: '#b5522b' }} />Incorrect</li>
              <li><span style={{ backgroundColor: '#000000' }} />Not held out</li>
            </>
          ) : maps.legend.map((item) => (
            <li key={item.classId}><span style={{ backgroundColor: item.color }} />Class {item.classId + 1}</li>
          ))}
        </ul>
      ) : null}
    </section>
  );
}

function DetectionPage({ experiments }: { experiments: Experiment[] }) {
  const [selectedId, setSelectedId] = useState(experiments.find((experiment) => experiment.task === 'detection')?.id ?? experiments[0]?.id ?? '');
  const [response, setResponse] = useState<Record<string, any> | null>(null);
  const [confidence, setConfidence] = useState(50);
  const activeSelectedId = experiments.some((experiment) => experiment.id === selectedId)
    ? selectedId
    : experiments.find((experiment) => experiment.task === 'detection')?.id ?? experiments[0]?.id ?? '';

  const runDetection = () => {
    experimentApi.detect(activeSelectedId, confidence / 100).then((result) => setResponse(result.data)).catch((error) => {
      setResponse(error.response?.data ?? { message: 'Detection unavailable' });
    });
  };

  return (
    <div className="stack">
      <section className="panel form-panel">
        <h2>Object detection verification</h2>
        <label>
          Experiment
          <select value={activeSelectedId} onChange={(event) => setSelectedId(event.target.value)}>
            {experiments.map((experiment) => <option key={experiment.id} value={experiment.id}>{experiment.id} — {experiment.task}</option>)}
          </select>
        </label>
        <label>
          Confidence threshold: {confidence}%
          <input type="range" min="0" max="100" value={confidence} onChange={(event) => setConfidence(Number(event.target.value))} />
        </label>
        <button type="button" className="primary-button" onClick={runDetection}>Verify detection</button>
      </section>
      <section className="panel">
        {response ? (
          response.detectionSupported === false || response.detectionsSupported === false ? (
            <div><span className="badge detection">Detection unsupported</span><p>{response.message}</p></div>
          ) : response.message && !Array.isArray(response.detections) ? (
            <div><h2>Detection unavailable</h2><p>{response.message}</p></div>
          ) : (
            <>
              <h2>Detection predictions</h2>
              {response.message ? <p>{response.message}</p> : null}
              <ul className="list">
                {(response.detections ?? []).map((detection: any, index: number) => (
                  <li key={`${detection.className}-${index}`}>
                    <div>
                      <strong>{detection.className}</strong>
                      <small>Confidence {Number(detection.confidence).toFixed(3)}</small>
                      {Array.isArray(detection.box) ? <small>Box: {detection.box.map((value: number) => Number(value).toFixed(1)).join(', ')}</small> : null}
                    </div>
                  </li>
                ))}
              </ul>
            </>
          )
        ) : <p>Run detection verification to inspect predictions.</p>}
      </section>
    </div>
  );
}

function ReportsPage({ experiments }: { experiments: Experiment[] }) {
  const [selectedId, setSelectedId] = useState(experiments[0]?.id ?? '');
  const [report, setReport] = useState('');
  const [message, setMessage] = useState('');
  const activeSelectedId = experiments.some((experiment) => experiment.id === selectedId) ? selectedId : experiments[0]?.id ?? '';

  const generate = () => {
    experimentApi.report(activeSelectedId).then((response) => {
      setReport(response.data.content);
      setMessage('');
    }).catch(() => {
      setReport('');
      setMessage('Could not generate the report. Verify the experiment and API status.');
    });
  };

  const download = () => {
    if (!report) return;
    const url = URL.createObjectURL(new Blob([report], { type: 'text/markdown' }));
    const link = document.createElement('a');
    link.href = url;
    link.download = `${activeSelectedId}-report.md`;
    link.click();
    URL.revokeObjectURL(url);
  };

  const downloadPdf = () => {
    if (!report) return;
    const pdf = new jsPDF();
    pdf.setFontSize(16);
    pdf.text('HSI Studio Experiment Report', 10, 20);
    pdf.setFontSize(10);
    const lines = pdf.splitTextToSize(report, 180);
    pdf.text(lines, 10, 35);
    pdf.save(`${activeSelectedId}-report.pdf`);
  };

  return (
    <div className="stack">
      <section className="panel form-panel">
        <h2>Reports</h2>
        <label>
          Experiment
          <select value={activeSelectedId} onChange={(event) => setSelectedId(event.target.value)}>
            {experiments.map((experiment) => <option key={experiment.id} value={experiment.id}>{experiment.id}</option>)}
          </select>
        </label>
        <div>
          <button type="button" className="primary-button" onClick={generate}>Generate Markdown report</button>
          <button type="button" className="ghost-button" onClick={download} disabled={!report}>Download .md</button>
          <button type="button" className="ghost-button" onClick={downloadPdf} disabled={!report}>Download .pdf</button>
        </div>
        {message ? <p role="status">{message}</p> : null}
      </section>
      {report ? <section className="panel report-preview"><pre>{report}</pre></section> : null}
    </div>
  );
}

function SettingsPage({ token }: { token: string | null }) {
  const [status, setStatus] = useState<{
    api: string;
    ml: string;
    storage: Record<string, string>;
    worker: Record<string, string>;
  } | null>(null);
  const [message, setMessage] = useState('');

  useEffect(() => {
    if (!token) return;
    let active = true;
    systemApi.status().then((response) => {
      if (active) {
        setStatus(response.data);
        setMessage('');
      }
    }).catch(() => {
      if (active) setMessage('Could not load infrastructure status.');
    });
    return () => {
      active = false;
    };
  }, [token]);

  return (
    <div className="stack">
      <section className="grid-two">
        <article className="panel">
          <h2>Storage</h2>
          {status ? Object.entries(status.storage).map(([name, value]) => <p key={name}>{name}: {value}</p>) : <p>Storage health is unavailable.</p>}
        </article>
        <article className="panel">
          <h2>Service and worker status</h2>
          {status ? (
            <>
              <p>API: {status.api}</p>
              <p>ML service: {status.ml}</p>
              <p>Queue: {status.worker.queue}</p>
              <p>GPU: {status.worker.gpu}</p>
            </>
          ) : <p>Sign in and load status to inspect the deployment.</p>}
        </article>
      </section>
      <section className="panel">
        <h2>Administration</h2>
        <p>User-role management is not available in this build. Access roles remain limited to the account role returned by the API.</p>
        {message ? <p role="status">{message}</p> : null}
      </section>
    </div>
  );
}

export default App;
