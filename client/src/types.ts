export type Dataset = {
  id: string;
  name: string;
  sensor: string;
  source?: 'public' | 'upload';
  format?: string;
  storageKey?: string;
  labelsKey?: string;
  annotationsKey?: string;
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

export type Pipeline = {
  id: string;
  name: string;
  version: string;
  steps: Array<{ name: string; params: Record<string, unknown> }>;
};

export type Experiment = {
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
  metrics?: Record<string, number>;
  artifacts?: Record<string, unknown>;
  timings?: { train: number; infer: number };
  confusionMatrix?: number[][];
  perClass?: Record<string, number[]>;
};
