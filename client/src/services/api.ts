import axios, { type AxiosError, type InternalAxiosRequestConfig } from 'axios';

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:4000',
});

api.interceptors.request.use((config) => {
  const token = localStorage.getItem('hsi-token');
  if (token && config.headers) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

let refreshInFlight: Promise<string> | null = null;
api.interceptors.response.use(undefined, async (error: AxiosError) => {
  const request = error.config as (InternalAxiosRequestConfig & { _retry?: boolean }) | undefined;
  if (error.response?.status !== 401 || !request || request._retry || request.url?.startsWith('/auth/')) {
    return Promise.reject(error);
  }
  const refreshToken = localStorage.getItem('hsi-refresh-token');
  if (!refreshToken) {
    localStorage.removeItem('hsi-token');
    window.dispatchEvent(new Event('hsi-auth-expired'));
    return Promise.reject(error);
  }
  request._retry = true;
  if (!refreshInFlight) {
    refreshInFlight = axios.post<{ token: string; refreshToken: string }>(
      `${api.defaults.baseURL}/auth/refresh`,
      { refreshToken },
    ).then((response) => {
      localStorage.setItem('hsi-token', response.data.token);
      localStorage.setItem('hsi-refresh-token', response.data.refreshToken);
      return response.data.token;
    }).catch((refreshError: unknown) => {
      localStorage.removeItem('hsi-token');
      localStorage.removeItem('hsi-refresh-token');
      window.dispatchEvent(new Event('hsi-auth-expired'));
      throw refreshError;
    }).finally(() => {
      refreshInFlight = null;
    });
  }
  try {
    const token = await refreshInFlight;
    request.headers.Authorization = `Bearer ${token}`;
    return await api(request);
  } catch {
    return Promise.reject(error);
  }
});

export const authApi = {
  login: (email: string, password: string) => api.post('/auth/login', { email, password }),
  register: (name: string, email: string, password: string) => api.post('/auth/register', { name, email, password }),
};

export const systemApi = {
  status: () => api.get('/system/status'),
};

export const datasetApi = {
  list: () => api.get('/datasets'),
  create: (payload: Record<string, unknown>) => api.post('/datasets', payload),
  inspect: (id: string) => api.post(`/datasets/${id}/inspect`),
  visualize: (id: string, payload: { redBand: number; greenBand: number; blueBand: number; x: number; y: number }) =>
    api.post(`/datasets/${id}/visualize`, payload),
  uploadUrl: (fileName: string, fileType: string, datasetId: string) =>
    api.post('/datasets/upload-url', { fileName, fileType, datasetId }),
  uploadComplete: (datasetId: string, files: Array<{ key: string; fileName: string }>) =>
    api.post('/datasets/upload-complete', { datasetId, files }),
  upload: async (files: File[], datasetId?: string) => {
    const totalBytes = files.reduce((total, file) => total + file.size, 0);
    if (datasetId && totalBytes >= 20 * 1024 * 1024) {
      const uploadedFiles: Array<{ key: string; fileName: string }> = [];
      let presignedUploadUnavailable = false;
      for (const file of files) {
        const fileType = file.type || 'application/octet-stream';
        let signed;
        try {
          signed = await datasetApi.uploadUrl(file.name, fileType, datasetId);
        } catch (error) {
          if (!uploadedFiles.length && totalBytes <= 100 * 1024 * 1024
            && axios.isAxiosError(error) && error.response?.status === 503) {
            presignedUploadUnavailable = true;
            break;
          }
          throw error;
        }
        const uploadResponse = await fetch(signed.data.url, {
          method: 'PUT',
          headers: { 'Content-Type': fileType },
          body: file,
        });
        if (!uploadResponse.ok) {
          throw new Error(`Direct upload failed for ${file.name} (${uploadResponse.status})`);
        }
        uploadedFiles.push({ key: signed.data.key, fileName: signed.data.fileName });
      }
      if (!presignedUploadUnavailable) return datasetApi.uploadComplete(datasetId, uploadedFiles);
    }
    const form = new FormData();
    files.forEach((file) => form.append('cube', file));
    if (datasetId) form.append('datasetId', datasetId);
    return api.post('/datasets/upload', form, { headers: { 'Content-Type': 'multipart/form-data' } });
  },
  uploadLabels: (id: string, file: File) => {
    const form = new FormData();
    form.append('labels', file);
    return api.post(`/datasets/${id}/labels`, form, { headers: { 'Content-Type': 'multipart/form-data' } });
  },
  uploadAnnotations: (id: string, file: File) => {
    const form = new FormData();
    form.append('annotations', file);
    return api.post(`/datasets/${id}/annotations`, form, { headers: { 'Content-Type': 'multipart/form-data' } });
  },
};

export const pipelineApi = {
  list: () => api.get('/pipelines'),
  create: (payload: Record<string, unknown>) => api.post('/pipelines', payload),
  preprocess: (payload: Record<string, unknown>) => api.post('/preprocess', payload),
};

export const experimentApi = {
  list: () => api.get('/experiments'),
  create: (payload: Record<string, unknown>) => api.post('/experiments', payload),
  run: (id: string) => api.post(`/experiments/${id}/run`),
  cancel: (id: string) => api.post(`/experiments/${id}/cancel`),
  metrics: (id: string) => api.get(`/experiments/${id}/metrics`),
  artifacts: (id: string) => api.get(`/experiments/${id}/artifacts`),
  maps: (id: string) => api.get(`/experiments/${id}/maps`),
  compare: (ids: string[]) => api.get('/compare', { params: { ids: ids.join(',') } }),
  report: (id: string) => api.post(`/experiments/${id}/report`),
  reportPdf: (id: string) => api.post(`/experiments/${id}/report/pdf`, {}, { responseType: 'blob' }),
  detect: (id: string, confidenceThreshold = 0.5) => api.post(`/experiments/${id}/detect`, { confidenceThreshold }),
};

export default api;
