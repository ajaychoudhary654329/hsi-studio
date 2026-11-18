from __future__ import annotations

import base64
from typing import Any
from pathlib import Path
import time
import uuid

import numpy as np
import torch
from torch import nn
import joblib
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from io import BytesIO
from fastapi.responses import StreamingResponse
from reportlab.pdfgen import canvas as pdf_canvas

app = FastAPI(title='HSI Studio ML Service', version='1.0.0')


class InspectRequest(BaseModel):
    name: str = 'IndianPines'
    bands: int = Field(default=220, gt=0)
    height: int = Field(default=145, gt=0)
    width: int = Field(default=145, gt=0)
    cubePath: str | None = None
    labelsPath: str | None = None
    allowSynthetic: bool = False


class VisualizeRequest(BaseModel):
    cubePath: str
    redBand: int = 0
    greenBand: int = 1
    blueBand: int = 2
    x: int = 0
    y: int = 0


class PreprocessRequest(BaseModel):
    normalize: bool = True
    remove_noisy_bands: bool = True
    pca_components: int | None = Field(default=25, gt=0)
    keep_bands: int | None = Field(default=None, gt=0)
    method: str = 'minmax'
    cubePath: str | None = None
    allowSynthetic: bool = False


class TrainRequest(BaseModel):
    model: str = 'svm'
    task: str = 'classification'
    dataset: str = 'IndianPines'
    seed: int = 42
    cubePath: str | None = None
    labelsPath: str | None = None
    splitStrategy: str = 'spatial'
    trainRatio: float = 0.6
    valRatio: float = 0.2
    testRatio: float = 0.2
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    allowSynthetic: bool = False

    @model_validator(mode='after')
    def validate_split(self) -> 'TrainRequest':
        if self.splitStrategy != 'spatial':
            raise ValueError('Only spatial splits are supported')
        if min(self.trainRatio, self.valRatio, self.testRatio) <= 0 or not np.isclose(
            self.trainRatio + self.valRatio + self.testRatio, 1.0,
        ):
            raise ValueError('Train, validation, and test ratios must be positive and sum to 1')
        return self


class PredictRequest(BaseModel):
    dataset: str = 'IndianPines'
    model: str = 'svm'
    cubePath: str | None = None
    modelPath: str | None = None


class DetectRequest(BaseModel):
    dataset: str = 'HOD3K'
    hasObjectAnnotations: bool = True
    confidenceThreshold: float = 0.5
    annotationsPath: str | None = None
    imagePath: str | None = None
    modelPath: str | None = None

    @model_validator(mode='after')
    def validate_confidence(self) -> 'DetectRequest':
        if not 0 <= self.confidenceThreshold <= 1:
            raise ValueError('Confidence threshold must be between 0 and 1')
        return self


class SweepRequest(BaseModel):
    cubePath: str | None = None
    labelsPath: str | None = None
    trials: int = 5


class EvaluateRequest(BaseModel):
    truthPath: str
    predictionPath: str


class RenderMapsRequest(BaseModel):
    truthPath: str
    predictionPath: str


class ReportRequest(BaseModel):
    experimentId: str
    dataset: str = 'IndianPines'
    model: str = 'svm'
    split: dict[str, Any] = Field(default_factory=lambda: {
        'strategy': 'spatial', 'train': 0.6, 'val': 0.2, 'test': 0.2, 'seed': 42,
    })
    metrics: dict[str, Any] = Field(default_factory=dict)


def spatial_split(
    height: int,
    width: int,
    seed: int,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    block_size: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if height <= 0 or width <= 0:
        raise ValueError('Cube dimensions must be positive')
    if min(train_ratio, val_ratio, test_ratio) <= 0 or not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError('Train, validation, and test ratios must be positive and sum to 1')
    block_size = max(1, block_size)
    if (height + block_size - 1) // block_size * ((width + block_size - 1) // block_size) < 3:
        block_size = 1
    block_rows = (height + block_size - 1) // block_size
    block_cols = (width + block_size - 1) // block_size
    rng = np.random.default_rng(seed)
    block_ids = np.arange(block_rows * block_cols)
    if len(block_ids) < 3:
        raise HTTPException(status_code=422, detail='Spatial split requires at least three spatial blocks')
    rng.shuffle(block_ids)
    ratios = np.array([train_ratio, val_ratio, test_ratio], dtype=np.float64)
    remaining = len(block_ids) - 3
    quotas = remaining * ratios
    counts = np.ones(3, dtype=int) + np.floor(quotas).astype(int)
    leftover = len(block_ids) - int(np.sum(counts))
    for index in np.argsort(-(quotas - np.floor(quotas)))[:leftover]:
        counts[index] += 1
    train_count, val_count, _ = counts.tolist()
    train_blocks = set(block_ids[:train_count].tolist())
    val_blocks = set(block_ids[train_count:train_count + val_count].tolist())
    test_blocks = set(block_ids[train_count + val_count:].tolist())

    train_mask = np.zeros((height, width), dtype=bool)
    val_mask = np.zeros((height, width), dtype=bool)
    test_mask = np.zeros((height, width), dtype=bool)
    for block_y in range(block_rows):
        for block_x in range(block_cols):
            block_id = block_y * block_cols + block_x
            y0, y1 = block_y * block_size, min((block_y + 1) * block_size, height)
            x0, x1 = block_x * block_size, min((block_x + 1) * block_size, width)
            if block_id in train_blocks:
                train_mask[y0:y1, x0:x1] = True
            elif block_id in val_blocks:
                val_mask[y0:y1, x0:x1] = True
            elif block_id in test_blocks:
                test_mask[y0:y1, x0:x1] = True
    return train_mask, val_mask, test_mask


def load_real_cube(path_value: str | None) -> np.ndarray | None:
    if not path_value:
        return None
    path = Path(path_value)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f'Cube file not found: {path_value}')
    suffix = path.suffix.lower()
    if suffix == '.mat':
        from scipy.io import loadmat
        mat = loadmat(str(path))
        arrays = [
            value for key, value in mat.items()
            if not key.startswith('__') and isinstance(value, np.ndarray) and value.ndim == 3
        ]
        if not arrays:
            raise ValueError('MAT cube file contains no three-dimensional array')
        return arrays[0].astype(np.float32)
    if suffix in {'.hdr', '.img'}:
        from spectral import open_image
        return open_image(str(path)).load().astype(np.float32)
    if suffix in {'.tif', '.tiff'}:
        import rasterio
        with rasterio.open(str(path)) as source:
            return np.moveaxis(source.read().astype(np.float32), 0, -1)
    if suffix == '.npy':
        array = np.load(str(path), allow_pickle=False)
        if array.ndim == 3:
            return array.astype(np.float32)
        raise ValueError('NPY cube must have height x width x bands dimensions')
    if suffix == '.npz':
        archive = np.load(str(path), allow_pickle=False)
        array = archive[archive.files[0]]
        if array.ndim == 3:
            return array.astype(np.float32)
        raise ValueError('NPZ cube must have height x width x bands dimensions')
    raise HTTPException(status_code=422, detail='Unsupported cube format; use .mat, ENVI (.hdr/.img), GeoTIFF, .npy or .npz')


def load_real_labels(path_value: str | None, height: int, width: int) -> np.ndarray | None:
    if not path_value:
        return None
    path = Path(path_value)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f'Labels file not found: {path_value}')
    suffix = path.suffix.lower()
    if suffix == '.mat':
        from scipy.io import loadmat
        mat = loadmat(str(path))
        for key, value in mat.items():
            if key.startswith('__') or not isinstance(value, np.ndarray):
                continue
            labels = np.asarray(value).squeeze()
            if labels.ndim == 2 and labels.shape == (height, width):
                return labels.astype(np.int64)
    if suffix in {'.npy', '.npz'}:
        if suffix == '.npy':
            labels = np.load(str(path), allow_pickle=False)
        else:
            archive = np.load(str(path), allow_pickle=False)
            if not archive.files:
                raise HTTPException(status_code=422, detail='NPZ labels file is empty')
            name = archive.files[0]
            labels = archive[name]
        labels = np.asarray(labels).squeeze()
        if labels.shape == (height, width):
            return labels.astype(np.int64)
    if suffix == '.json':
        document = json.loads(path.read_text(encoding='utf-8'))
        if isinstance(document, dict) and isinstance(document.get('labels'), list):
            labels = np.asarray(document['labels'], dtype=np.int64)
            if labels.shape == (height, width):
                return labels
    if suffix in {'.tif', '.tiff'}:
        import rasterio
        with rasterio.open(str(path)) as source:
            labels = source.read(1)
        if labels.shape == (height, width):
            return labels.astype(np.int64)
    raise HTTPException(status_code=422, detail='Labels must be a height x width .mat/.npy/.npz/.tif/.tiff or JSON labels array')


class TwoDCNN(nn.Module):
    def __init__(self, bands: int, class_count: int = 3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(bands, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(16, class_count)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(inputs)
        return self.classifier(features.flatten(1))


class ThreeDCNN(nn.Module):
    def __init__(self, class_count: int = 3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )
        self.classifier = nn.Linear(8, class_count)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(inputs)
        return self.classifier(features.flatten(1))


@app.get('/health')
def health() -> dict[str, Any]:
    return {'status': 'ok', 'service': 'hsi-ml-service'}


@app.post('/inspect')
def inspect_dataset(payload: InspectRequest) -> dict[str, Any]:
    band_matrix = load_real_cube(payload.cubePath)
    if band_matrix is None:
        if not payload.allowSynthetic:
            raise HTTPException(status_code=422, detail='A real cubePath is required for inspection')
        rng = np.random.default_rng(payload.bands)
        band_matrix = rng.random((payload.height, payload.width, payload.bands))
    if band_matrix.ndim != 3:
        raise HTTPException(status_code=422, detail='Hyperspectral cube must have height x width x bands dimensions')

    height, width, bands = band_matrix.shape
    finite = np.isfinite(band_matrix)
    nan_count = int(np.size(band_matrix) - np.count_nonzero(finite))
    invalid_fraction = 1 - finite.reshape(-1, bands).mean(axis=0)
    noisy_bands = np.flatnonzero(invalid_fraction > 0.05).astype(int).tolist()
    class_histogram: dict[str, int] = {}
    if payload.labelsPath:
        labels = load_real_labels(payload.labelsPath, height, width)
        values, counts = np.unique(labels, return_counts=True)
        class_histogram = {str(int(label)): int(count) for label, count in zip(values, counts)}
    finite_values = band_matrix[finite]
    return {
        'shape': {'height': int(height), 'width': int(width), 'bands': int(bands)},
        'band_statistics': {
            'mean': float(np.mean(finite_values)) if finite_values.size else 0.0,
            'std': float(np.std(finite_values)) if finite_values.size else 0.0,
        },
        'quality_report': {
            'nanCount': nan_count,
            'noisyBands': noisy_bands,
            'classHistogram': class_histogram,
            'invalidPixels': int(np.count_nonzero(~np.all(finite, axis=2))),
        },
        'status': 'done',
    }


@app.post('/visualize')
def visualize_cube(payload: VisualizeRequest) -> dict[str, Any]:
    cube = load_real_cube(payload.cubePath)
    if cube is None:
        raise HTTPException(status_code=422, detail='A real cubePath is required for visualization')
    if cube.ndim != 3 or min(cube.shape) <= 0:
        raise HTTPException(status_code=422, detail='Cube must have height x width x bands dimensions')
    height, width, bands = cube.shape
    channels = [payload.redBand, payload.greenBand, payload.blueBand]
    if any(channel < 0 or channel >= bands for channel in channels):
        raise HTTPException(status_code=422, detail='Selected color band is outside the cube band range')
    if not 0 <= payload.x < width or not 0 <= payload.y < height:
        raise HTTPException(status_code=422, detail='Selected pixel is outside the cube dimensions')

    color_channels = []
    for band in channels:
        channel = cube[:, :, band]
        valid = np.isfinite(channel)
        values = channel[valid]
        if not values.size:
            raise HTTPException(status_code=422, detail=f'Color band {band} contains no finite pixels')
        low, high = np.percentile(values, [2, 98])
        if high <= low:
            scaled = np.zeros(channel.shape, dtype=np.uint8)
        else:
            scaled = np.clip((np.nan_to_num(channel, nan=low, posinf=high, neginf=low) - low) / (high - low), 0, 1)
            scaled = (scaled * 255).astype(np.uint8)
        scaled[~valid] = 0
        color_channels.append(scaled)
    rgb = np.stack(color_channels, axis=-1)
    spectrum = [
        float(value) if np.isfinite(value) else None
        for value in cube[payload.y, payload.x, :]
    ]
    return {
        'width': int(width),
        'height': int(height),
        'bands': int(bands),
        'pixel': {'x': payload.x, 'y': payload.y},
        'rgbBase64': base64.b64encode(rgb.tobytes()).decode('ascii'),
        'spectrum': spectrum,
    }


@app.post('/preprocess')
def preprocess(payload: PreprocessRequest) -> dict[str, Any]:
    sample = load_real_cube(payload.cubePath)
    if sample is None:
        if not payload.allowSynthetic:
            raise HTTPException(status_code=422, detail='A real cubePath is required for preprocessing')
        rng = np.random.default_rng(13)
        sample = rng.random((32, 32, 100), dtype=np.float32)
    invalid_fraction = 1 - np.isfinite(sample).mean(axis=(0, 1))
    sample = np.nan_to_num(sample, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if sample.ndim != 3:
        raise HTTPException(status_code=422, detail='Cube must have height x width x bands dimensions')
    if payload.remove_noisy_bands:
        valid_bands = invalid_fraction <= 0.05
        sample = sample[:, :, valid_bands]
    if sample.shape[2] == 0:
        raise HTTPException(status_code=422, detail='Preprocessing removed all spectral bands')
    if payload.normalize:
        if payload.method != 'minmax':
            raise HTTPException(status_code=422, detail='Only minmax normalization is supported')
        minimum = sample.min(axis=(0, 1), keepdims=True)
        maximum = sample.max(axis=(0, 1), keepdims=True)
        sample = (sample - minimum) / (maximum - minimum + 1e-8)
    if payload.keep_bands is not None:
        if payload.keep_bands > sample.shape[2]:
            raise HTTPException(status_code=422, detail='Band selection exceeds the number of available bands')
        band_variance = np.var(sample, axis=(0, 1))
        selected_bands = np.sort(np.argsort(band_variance)[-payload.keep_bands:])
        sample = sample[:, :, selected_bands]

    if payload.pca_components is None:
        projected_cube = sample
        component_count = 0
        explained_variance = 1.0
    else:
        component_count = min(payload.pca_components, sample.shape[2], sample.shape[0] * sample.shape[1])
        pca = PCA(n_components=component_count)
        projected = pca.fit_transform(sample.reshape(-1, sample.shape[2]))
        projected_cube = projected.reshape(sample.shape[0], sample.shape[1], component_count).astype(np.float32)
        explained_variance = float(np.sum(pca.explained_variance_ratio_))
    cache_dir = Path(__file__).resolve().parent / 'cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f'prepared-cube-{uuid.uuid4().hex}.npz'
    np.savez_compressed(
        cache_path,
        cube=projected_cube,
        explained_variance=np.array([explained_variance], dtype=np.float32),
    )
    return {
        'prepared_cube': {'shape': list(projected_cube.shape), 'dtype': 'float32'},
        'cache': {'format': 'npz', 'path': str(cache_path)},
        'preview': {'mean': float(np.mean(sample)), 'std': float(np.std(sample))},
        'pca': {'components': component_count, 'explainedVariance': explained_variance},
        'cached': True,
    }


@app.post('/train')
def train(payload: TrainRequest) -> dict[str, Any]:
    if payload.task != 'classification':
        raise HTTPException(status_code=501, detail='Detection model training is not implemented for this dataset format')
    if payload.model not in {'svm', 'cnn2d', 'cnn3d'}:
        raise HTTPException(status_code=422, detail=f'Unsupported classification model: {payload.model}')

    rng = np.random.default_rng(payload.seed)
    real_cube = load_real_cube(payload.cubePath)
    if real_cube is not None:
        height, width, bands = real_cube.shape
        if min(height, width, bands) <= 0:
            raise HTTPException(status_code=422, detail='Cube dimensions must all be positive')
        base = np.nan_to_num(real_cube, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        real_labels = load_real_labels(payload.labelsPath, height, width)
        if real_labels is None:
            raise HTTPException(status_code=422, detail='A matching labels file is required to train a real classification cube')
        unique_labels = np.unique(real_labels)
        if 0 in unique_labels:
            unique_labels = unique_labels[unique_labels != 0]
        if unique_labels.size < 2:
            raise HTTPException(status_code=422, detail='At least two labeled classes are required')
        class_map = np.full(real_labels.shape, -1, dtype=np.int64)
        for index, label in enumerate(unique_labels):
            class_map[real_labels == label] = index
        labeled_mask = class_map >= 0
    elif payload.allowSynthetic:
        height, width, bands = 32, 32, 25
        base = rng.normal(0, 1, (height, width, bands)).astype(np.float32)
        class_map = ((np.indices((height, width))[0] // 8) + (np.indices((height, width))[1] // 8)) % 3
        for band in range(bands):
            base[:, :, band] += class_map * (0.08 + band * 0.002)
        labeled_mask = np.ones((height, width), dtype=bool)
    else:
        raise HTTPException(status_code=422, detail='A real cubePath and matching labelsPath are required for training')

    class_count = int(np.max(class_map) + 1)
    if class_count < 2 or class_count > np.iinfo(np.uint16).max:
        raise HTTPException(status_code=422, detail='Training requires between two and 65535 classes')
    block_size = 8 if payload.model in {'cnn2d', 'cnn3d'} else 4
    train_mask, val_mask, test_mask = spatial_split(
        height,
        width,
        payload.seed,
        payload.trainRatio,
        payload.valRatio,
        payload.testRatio,
        block_size,
    )
    train_mask &= labeled_mask
    test_mask &= labeled_mask
    artifact_dir = Path(__file__).resolve().parent / 'artifacts'
    run_directory = artifact_dir / uuid.uuid4().hex
    run_directory.mkdir(parents=True, exist_ok=False)
    train_y, train_x = np.where(train_mask)
    test_y, test_x = np.where(test_mask)
    if not len(train_y) or not len(test_y):
        raise HTTPException(status_code=422, detail='Spatial split has no labeled training or test samples')
    if payload.model in {'cnn2d', 'cnn3d'}:
        epochs = payload.hyperparameters.get('epochs', 8)
        if isinstance(epochs, bool) or not isinstance(epochs, int) or not 1 <= epochs <= 100:
            raise HTTPException(status_code=422, detail='CNN epochs must be an integer between 1 and 100')
        from scipy.ndimage import binary_erosion
        footprint = np.ones((5, 5), dtype=bool)
        train_y, train_x = np.where(binary_erosion(train_mask, structure=footprint, border_value=0))
        test_y, test_x = np.where(binary_erosion(test_mask, structure=footprint, border_value=0))
        if not len(train_y) or not len(test_y):
            raise HTTPException(status_code=422, detail='Spatial blocks are too small for 5x5 CNN patches')
    max_train = 600 if payload.model in {'cnn2d', 'cnn3d'} else 30000
    max_test = 300 if payload.model in {'cnn2d', 'cnn3d'} else 100000
    if len(train_y) > max_train:
        selected = rng.choice(len(train_y), max_train, replace=False)
        train_y, train_x = train_y[selected], train_x[selected]
    if len(test_y) > max_test:
        selected = rng.choice(len(test_y), max_test, replace=False)
        test_y, test_x = test_y[selected], test_x[selected]
    train_labels = class_map[train_y, train_x]
    truth = class_map[test_y, test_x]
    if np.unique(train_labels).size < 2:
        raise HTTPException(status_code=422, detail='Spatial training split must contain at least two labeled classes')
    training_started = time.perf_counter()
    if payload.model in {'cnn2d', 'cnn3d'}:
        torch.manual_seed(payload.seed)
        padded = np.pad(base, ((2, 2), (2, 2), (0, 0)), mode='edge')
        train_patches = np.stack([
            padded[y:y + 5, x:x + 5, :].transpose(2, 0, 1)
            for y, x in zip(train_y, train_x)
        ])
        test_patches = np.stack([
            padded[y:y + 5, x:x + 5, :].transpose(2, 0, 1)
            for y, x in zip(test_y, test_x)
        ])

        if payload.model == 'cnn2d':
            model = TwoDCNN(bands, class_count)
            train_x_tensor = torch.from_numpy(train_patches).float()
            test_x_tensor = torch.from_numpy(test_patches).float()
        else:
            model = ThreeDCNN(class_count)
            train_x_tensor = torch.from_numpy(train_patches[:, None, ...]).float()
            test_x_tensor = torch.from_numpy(test_patches[:, None, ...]).float()

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        loss_fn = nn.CrossEntropyLoss()
        train_y_tensor = torch.from_numpy(train_labels).long()
        for _ in range(epochs):
            optimizer.zero_grad()
            loss = loss_fn(model(train_x_tensor), train_y_tensor)
            loss.backward()
            optimizer.step()
        train_seconds = time.perf_counter() - training_started
        inference_started = time.perf_counter()
        with torch.no_grad():
            predictions = model(test_x_tensor).argmax(dim=1).cpu().numpy()
        inference_seconds = time.perf_counter() - inference_started
        model_path = run_directory / f'{payload.model}.pt'
        torch.save({
            'state_dict': model.state_dict(),
            'bands': bands,
            'class_count': class_count,
            'model': payload.model,
        }, model_path)
    else:
        try:
            C = float(payload.hyperparameters.get('C', 10))
            gamma = payload.hyperparameters.get('gamma', 'scale')
        except (TypeError, ValueError):
            raise HTTPException(status_code=422, detail='SVM C must be a positive number')
        if isinstance(payload.hyperparameters.get('C', 10), bool) or not np.isfinite(C) or C <= 0:
            raise HTTPException(status_code=422, detail='SVM C must be a positive number')
        valid_gamma = (
            isinstance(gamma, str) and gamma in {'scale', 'auto'}
        ) or (
            isinstance(gamma, (int, float)) and not isinstance(gamma, bool)
            and np.isfinite(gamma) and gamma > 0
        )
        if not valid_gamma:
            raise HTTPException(status_code=422, detail='SVM gamma must be positive or either "scale" or "auto"')
        classifier = SVC(
            C=C,
            gamma=gamma,
            random_state=payload.seed,
        )
        classifier.fit(base[train_y, train_x], train_labels)
        train_seconds = time.perf_counter() - training_started
        inference_started = time.perf_counter()
        predictions = classifier.predict(base[test_y, test_x])
        inference_seconds = time.perf_counter() - inference_started
        model_path = run_directory / 'svm.joblib'
        joblib.dump({
            'classifier': classifier,
            'class_count': class_count,
            'dataset': payload.dataset,
        }, model_path)
    prediction_map = np.zeros((height, width), dtype=np.uint16)
    prediction_map[test_y, test_x] = predictions.astype(np.uint16)
    prediction_path = run_directory / 'prediction-map.npy'
    np.save(prediction_path, prediction_map)
    truth_map = np.full((height, width), -1, dtype=np.int32)
    truth_map[test_y, test_x] = truth.astype(np.int32)
    truth_path = run_directory / 'heldout-truth.npy'
    np.save(truth_path, truth_map)

    oa = float(accuracy_score(truth, predictions))
    precision, recall, f1, _ = precision_recall_fscore_support(truth, predictions, average='macro', zero_division=0)
    kappa = float(cohen_kappa_score(truth, predictions))
    class_ids = list(range(class_count))
    matrix = confusion_matrix(truth, predictions, labels=class_ids).tolist()
    per_class_precision, per_class_recall, per_class_f1, support = precision_recall_fscore_support(
        truth, predictions, labels=class_ids, zero_division=0,
    )
    return {
        'model': payload.model,
        'task': 'classification',
        'metrics': {
            'OA': oa,
            'AA': float(np.mean(per_class_recall)),
            'Kappa': kappa,
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
        },
        'confusion_matrix': matrix,
        'per_class': {
            'precision': per_class_precision.tolist(),
            'recall': per_class_recall.tolist(),
            'f1': per_class_f1.tolist(),
            'support': support.tolist(),
        },
        'split': {
            'strategy': 'spatial',
            'train': int(len(train_labels)),
            'validation': int(np.count_nonzero(val_mask & labeled_mask)),
            'test': int(len(truth)),
        },
        'timings': {'train': train_seconds, 'infer': inference_seconds},
        'artifacts': {
            'model_path': str(model_path),
            'map_path': str(prediction_path),
            'truth_map_path': str(truth_path),
        },
        'status': 'done',
    }


@app.post('/train/stream')
def train_stream(payload: TrainRequest) -> StreamingResponse:
    def events():
        yield '{"event":"started","progress":5}\n'
        result = train(payload)
        yield '{"event":"progress","progress":80}\n'
        yield json.dumps({'event': 'done', 'progress': 100, 'result': result}) + '\n'

    return StreamingResponse(events(), media_type='application/x-ndjson')


@app.post('/sweep')
def sweep(payload: SweepRequest) -> dict[str, Any]:
    leaderboard = []
    for trial in range(max(1, min(payload.trials, 20))):
        C = [0.1, 1.0, 10.0, 100.0][trial % 4]
        gamma = [0.001, 0.01, 'scale'][trial % 3]
        result = train(TrainRequest(
            model='svm',
            dataset='Sweep',
            seed=42 + trial,
            cubePath=payload.cubePath,
            labelsPath=payload.labelsPath,
            hyperparameters={'C': C, 'gamma': gamma},
        ))
        oa = float(result['metrics']['OA'])
        leaderboard.append({'trial': trial + 1, 'C': C, 'gamma': gamma, 'OA': oa})
    leaderboard.sort(key=lambda item: item['OA'], reverse=True)
    return {'leaderboard': leaderboard, 'status': 'done'}


@app.post('/sam')
def spectral_angle_mapper(referenceBand: int = 0, cubePath: str | None = None) -> dict[str, Any]:
    cube = load_real_cube(cubePath)
    if cube is None:
        raise HTTPException(status_code=422, detail='A real cubePath is required for spectral angle mapping')
    band = int(np.clip(referenceBand, 0, cube.shape[2] - 1))
    reference = cube[:, :, band]
    denominator = (np.linalg.norm(cube, axis=2) * np.linalg.norm(reference[:, :, None], axis=2) + 1e-8)
    cosine = np.clip(np.sum(cube * reference[:, :, None], axis=2) / denominator, -1.0, 1.0)
    sam = np.arccos(cosine)
    return {
        'referenceBand': band,
        'map': {'shape': list(sam.shape), 'min': float(np.min(sam)), 'max': float(np.max(sam))},
    }


@app.post('/detect')
def detect(payload: DetectRequest) -> dict[str, Any]:
    if not payload.hasObjectAnnotations:
        return {
            'detectionsSupported': False,
            'detections': [],
            'message': 'No objects detected: dataset has no object-level annotations',
        }
    if payload.modelPath and payload.imagePath:
        try:
            from ultralytics import YOLO
            model = YOLO(payload.modelPath)
            inference = model.predict(source=payload.imagePath, conf=payload.confidenceThreshold, verbose=False)
            detections = []
            for result in inference:
                names = result.names
                for box in result.boxes:
                    x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    detections.append({
                        'className': str(names.get(class_id, class_id)),
                        'confidence': confidence,
                        'box': [x1, y1, x2 - x1, y2 - y1],
                    })
            return {
                'detectionsSupported': True,
                'detections': detections,
                'verificationAvailable': False,
                'message': 'Predictions returned; IoU and mAP require matching held-out ground-truth annotations.',
            }
        except ImportError:
            raise HTTPException(status_code=501, detail='YOLO support requires the ultralytics package')
    if not payload.annotationsPath:
        raise HTTPException(status_code=422, detail='A trained modelPath and input imagePath are required')
    raise HTTPException(
        status_code=422,
        detail='Ground-truth annotations cannot be treated as predictions; provide modelPath and imagePath for inference',
    )


@app.post('/predict')
def predict(payload: PredictRequest) -> dict[str, Any]:
    cube = load_real_cube(payload.cubePath)
    if cube is None or not payload.modelPath:
        raise HTTPException(status_code=422, detail='A real cubePath and trained modelPath are required')
    model_path = Path(payload.modelPath)
    if not model_path.is_file():
        raise HTTPException(status_code=404, detail='Trained model file not found')
    cube = np.nan_to_num(cube, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    height, width, bands = cube.shape
    if payload.model == 'svm':
        stored_model = joblib.load(model_path)
        classifier = stored_model.get('classifier') if isinstance(stored_model, dict) else stored_model
        prediction = classifier.predict(cube.reshape(-1, bands)).reshape(height, width)
    elif payload.model in {'cnn2d', 'cnn3d'}:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
        if checkpoint.get('bands') != bands or checkpoint.get('model') != payload.model:
            raise HTTPException(status_code=422, detail='Model architecture or band count does not match the input cube')
        class_count = int(checkpoint['class_count'])
        model = TwoDCNN(bands, class_count) if payload.model == 'cnn2d' else ThreeDCNN(class_count)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        padded = np.pad(cube, ((2, 2), (2, 2), (0, 0)), mode='edge')
        coordinates = [(y, x) for y in range(height) for x in range(width)]
        predictions = []
        with torch.no_grad():
            for start in range(0, len(coordinates), 512):
                batch_coords = coordinates[start:start + 512]
                patches = np.stack([
                    padded[y:y + 5, x:x + 5, :].transpose(2, 0, 1)
                    for y, x in batch_coords
                ])
                tensor = torch.from_numpy(patches[:, None, ...] if payload.model == 'cnn3d' else patches).float()
                predictions.extend(model(tensor).argmax(dim=1).cpu().numpy().tolist())
        prediction = np.asarray(predictions, dtype=np.uint16).reshape(height, width)
    else:
        raise HTTPException(status_code=422, detail=f'Unsupported prediction model: {payload.model}')

    artifact_dir = Path(__file__).resolve().parent / 'artifacts' / uuid.uuid4().hex
    artifact_dir.mkdir(parents=True, exist_ok=False)
    map_path = artifact_dir / 'prediction-map.npy'
    np.save(map_path, prediction.astype(np.uint16))
    return {
        'dataset': payload.dataset,
        'model': payload.model,
        'classification_map': {'format': 'npy', 'shape': [height, width], 'dtype': 'uint16'},
        'prediction_count': int(prediction.size),
        'artifacts': {'prediction_array': str(map_path)},
    }


@app.post('/evaluate')
def evaluate(payload: EvaluateRequest) -> dict[str, Any]:
    truth_path = Path(payload.truthPath)
    prediction_path = Path(payload.predictionPath)
    if not truth_path.is_file() or not prediction_path.is_file():
        raise HTTPException(status_code=404, detail='Truth or prediction map file not found')
    truth = np.asarray(np.load(truth_path, allow_pickle=False)).squeeze()
    prediction = np.asarray(np.load(prediction_path, allow_pickle=False)).squeeze()
    if truth.ndim != 2 or prediction.shape != truth.shape:
        raise HTTPException(status_code=422, detail='Truth and prediction maps must be matching 2D arrays')
    valid = truth >= 0
    truth = truth[valid].astype(np.int64)
    prediction = prediction[valid].astype(np.int64)
    if truth.size == 0:
        raise HTTPException(status_code=422, detail='No labeled test pixels were found')
    class_ids = np.union1d(truth, prediction).astype(int).tolist()
    precision_by_class, recall_by_class, f1_by_class, support = precision_recall_fscore_support(
        truth, prediction, labels=class_ids, zero_division=0,
    )
    precision, recall, f1, _ = precision_recall_fscore_support(
        truth, prediction, average='macro', zero_division=0,
    )
    return {
        'metrics': {
            'OA': float(accuracy_score(truth, prediction)),
            'AA': float(np.mean(recall_by_class)),
            'Kappa': float(cohen_kappa_score(truth, prediction)),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
        },
        'per_class': {
            'labels': class_ids,
            'precision': precision_by_class.tolist(),
            'recall': recall_by_class.tolist(),
            'f1': f1_by_class.tolist(),
            'support': support.tolist(),
        },
        'confusion_matrix': confusion_matrix(truth, prediction, labels=class_ids).tolist(),
    }


@app.post('/render-maps')
def render_maps(payload: RenderMapsRequest) -> dict[str, Any]:
    truth_path = Path(payload.truthPath)
    prediction_path = Path(payload.predictionPath)
    if not truth_path.is_file() or not prediction_path.is_file():
        raise HTTPException(status_code=404, detail='Truth or prediction map file not found')
    truth = np.asarray(np.load(truth_path, allow_pickle=False)).squeeze()
    prediction = np.asarray(np.load(prediction_path, allow_pickle=False)).squeeze()
    if truth.ndim != 2 or prediction.shape != truth.shape:
        raise HTTPException(status_code=422, detail='Truth and prediction maps must be matching 2D arrays')
    valid = truth >= 0
    if not np.any(valid):
        raise HTTPException(status_code=422, detail='No held-out pixels were found in the truth map')
    palette = np.asarray([
        [31, 119, 180], [255, 127, 14], [44, 160, 44], [214, 39, 40],
        [148, 103, 189], [140, 86, 75], [227, 119, 194], [127, 127, 127],
        [188, 189, 34], [23, 190, 207], [57, 197, 172], [224, 144, 95],
    ], dtype=np.uint8)

    def encode_map(values: np.ndarray, mask: np.ndarray) -> str:
        rgb = np.full((*values.shape, 3), 255, dtype=np.uint8)
        for label in np.unique(values[mask]):
            rgb[mask & (values == label)] = palette[int(label) % len(palette)]
        buffer = BytesIO()
        import matplotlib.pyplot as plt
        plt.imsave(buffer, rgb, format='png')
        return base64.b64encode(buffer.getvalue()).decode('ascii')

    errors = np.zeros((*truth.shape, 3), dtype=np.uint8)
    errors[valid & (prediction == truth)] = [95, 122, 85]
    errors[valid & (prediction != truth)] = [181, 82, 43]
    buffer = BytesIO()
    import matplotlib.pyplot as plt
    plt.imsave(buffer, errors, format='png')
    return {
        'shape': list(truth.shape),
        'truth': encode_map(truth, valid),
        'prediction': encode_map(prediction, valid),
        'error': base64.b64encode(buffer.getvalue()).decode('ascii'),
        'legend': [
            {'classId': label, 'color': '#%02x%02x%02x' % tuple(palette[label % len(palette)])}
            for label in np.unique(truth[valid]).astype(int).tolist()
        ],
    }


@app.post('/report/pdf')
def report_pdf(payload: ReportRequest) -> StreamingResponse:
    buffer = BytesIO()
    pdf = pdf_canvas.Canvas(buffer)
    pdf.setTitle(f"HSI Experiment {payload.experimentId}")
    pdf.drawString(72, 760, 'HSI Studio Experiment Report')
    pdf.drawString(72, 730, f"Experiment: {payload.experimentId}")
    pdf.drawString(72, 710, f"Dataset: {payload.dataset}")
    pdf.drawString(72, 690, f"Model: {payload.model}")
    pdf.drawString(72, 670, f"Split: {payload.split.get('strategy', 'spatial')}")
    pdf.drawString(72, 650, f"Seed: {payload.split.get('seed', 42)}")
    y = 620
    for name, value in payload.metrics.items():
        pdf.drawString(72, y, f"{name}: {value}")
        y -= 20
    pdf.drawString(72, 400, 'Limitations:')
    pdf.drawString(72, 380, 'Results are dataset and sensor specific and may not generalise.')
    pdf.save()
    buffer.seek(0)
    return StreamingResponse(buffer, media_type='application/pdf', headers={
        'Content-Disposition': f'attachment; filename="{payload.experimentId}-report.pdf"',
    })
