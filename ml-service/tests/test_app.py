from fastapi.testclient import TestClient
import json
import base64
from pathlib import Path
import numpy as np

from app.main import app


client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_inspect_synthetic_cube():
    response = client.post("/inspect", json={
        "name": "Synthetic",
        "bands": 16,
        "height": 32,
        "width": 32,
        "allowSynthetic": True,
    })
    assert response.status_code == 200
    shape = response.json()["shape"]
    assert shape["bands"] == 16
    assert shape["height"] == 32


def test_detection_feature_gate():
    unsupported = client.post("/detect", json={
        "dataset": "Indian Pines",
        "hasObjectAnnotations": False,
    })
    assert unsupported.status_code == 200
    assert unsupported.json()["detectionsSupported"] is False


def test_predict_returns_map_metadata():
    response = client.post("/predict", json={
        "dataset": "IndianPines",
        "model": "svm",
    })
    assert response.status_code == 422
    assert "real cubePath and trained modelPath" in response.json()["detail"]


def test_training_rejects_missing_cube_without_explicit_synthetic_mode():
    response = client.post("/train", json={
        "model": "svm",
        "task": "classification",
        "dataset": "IndianPines",
    })
    assert response.status_code == 422
    assert "real cubePath" in response.json()["detail"]


def test_inspection_requires_a_real_cube_by_default():
    response = client.post("/inspect", json={"name": "IndianPines"})
    assert response.status_code == 422
    assert "real cubePath" in response.json()["detail"]


def test_visualize_returns_cube_rgb_and_pixel_spectrum(tmp_path):
    cube_path = tmp_path / "cube.npy"
    cube = np.arange(6 * 5 * 4, dtype=np.float32).reshape(6, 5, 4)
    np.save(cube_path, cube)
    response = client.post("/visualize", json={
        "cubePath": str(cube_path),
        "redBand": 0,
        "greenBand": 1,
        "blueBand": 2,
        "x": 3,
        "y": 2,
    })
    assert response.status_code == 200
    result = response.json()
    assert (result["width"], result["height"], result["bands"]) == (5, 6, 4)
    assert len(base64.b64decode(result["rgbBase64"])) == 5 * 6 * 3
    assert result["spectrum"] == cube[2, 3, :].tolist()


def test_render_maps_returns_truth_prediction_and_error_pngs(tmp_path):
    truth_path = tmp_path / "truth.npy"
    prediction_path = tmp_path / "prediction.npy"
    np.save(truth_path, np.array([[0, 1, -1], [1, 0, 1]], dtype=np.int32))
    np.save(prediction_path, np.array([[0, 0, 0], [1, 1, 1]], dtype=np.uint16))

    response = client.post("/render-maps", json={
        "truthPath": str(truth_path),
        "predictionPath": str(prediction_path),
    })
    assert response.status_code == 200
    result = response.json()
    assert result["shape"] == [2, 3]
    assert [entry["classId"] for entry in result["legend"]] == [0, 1]
    for name in ("truth", "prediction", "error"):
        assert base64.b64decode(result[name]).startswith(b"\x89PNG\r\n\x1a\n")


def test_render_maps_rejects_mismatched_map_shapes(tmp_path):
    truth_path = tmp_path / "truth.npy"
    prediction_path = tmp_path / "prediction.npy"
    np.save(truth_path, np.zeros((2, 2), dtype=np.int32))
    np.save(prediction_path, np.zeros((3, 3), dtype=np.uint16))

    response = client.post("/render-maps", json={
        "truthPath": str(truth_path),
        "predictionPath": str(prediction_path),
    })
    assert response.status_code == 422


def test_spatial_split_keeps_all_partitions_nonempty_at_extreme_ratios():
    from app.main import spatial_split

    train, validation, test = spatial_split(1, 3, 42, 0.98, 0.01, 0.01, block_size=4)
    assert train.any()
    assert validation.any()
    assert test.any()
    assert not np.any(train & validation)
    assert not np.any(train & test)
    assert not np.any(validation & test)
    assert np.all(train | validation | test)


def test_preprocess_cache_shape_matches_reported_projected_cube(tmp_path):
    cube_path = tmp_path / "cube.npy"
    np.save(cube_path, np.random.default_rng(42).random((8, 8, 4), dtype=np.float32))
    response = client.post("/preprocess", json={
        "cubePath": str(cube_path),
        "normalize": True,
        "remove_noisy_bands": True,
        "pca_components": 3,
    })
    assert response.status_code == 200
    body = response.json()
    cache_path = Path(body["cache"]["path"])
    try:
        assert body["prepared_cube"]["shape"] == [8, 8, 3]
        with np.load(cache_path) as cache:
            assert cache["cube"].shape == (8, 8, 3)
    finally:
        cache_path.unlink(missing_ok=True)


def test_preprocessed_cube_flows_into_real_training(tmp_path):
    rows, columns = np.indices((16, 16))
    labels = 1 + ((rows // 4 + columns // 4) % 2)
    cube = np.stack([labels + np.random.default_rng(band).normal(0, 0.05, labels.shape) for band in range(6)], axis=2)
    cube_path = tmp_path / "cube.npy"
    labels_path = tmp_path / "labels.npy"
    np.save(cube_path, cube.astype(np.float32))
    np.save(labels_path, labels.astype(np.int64))

    prepared = client.post("/preprocess", json={
        "cubePath": str(cube_path),
        "normalize": True,
        "remove_noisy_bands": True,
        "keep_bands": 4,
        "pca_components": 3,
    })
    assert prepared.status_code == 200
    cache_path = Path(prepared.json()["cache"]["path"])
    try:
        trained = client.post("/train", json={
            "model": "svm",
            "task": "classification",
            "dataset": "PipelineTest",
            "seed": 42,
            "cubePath": str(cache_path),
            "labelsPath": str(labels_path),
        })
        assert trained.status_code == 200
        assert 0 <= trained.json()["metrics"]["OA"] <= 1
    finally:
        cache_path.unlink(missing_ok=True)


def test_detection_confidence_must_be_in_unit_interval():
    response = client.post("/detect", json={
        "dataset": "M2SODAI",
        "confidenceThreshold": 1.5,
    })
    assert response.status_code == 422


def test_svm_spatial_training_returns_real_metrics():
    response = client.post("/train", json={
        "model": "svm",
        "task": "classification",
        "dataset": "Synthetic",
        "seed": 42,
        "allowSynthetic": True,
    })
    assert response.status_code == 200
    body = response.json()
    assert body["split"]["strategy"] == "spatial"
    assert 0 <= body["metrics"]["OA"] <= 1
    assert len(body["confusion_matrix"]) == 3


def test_2d_cnn_training_runs():
    response = client.post("/train", json={
        "model": "cnn2d",
        "task": "classification",
        "dataset": "Synthetic",
        "seed": 42,
        "allowSynthetic": True,
    })
    assert response.status_code == 200
    assert response.json()["status"] == "done"


def test_real_cube_and_label_map_training(tmp_path):
    cube = np.zeros((16, 16, 8), dtype=np.float32)
    labels = np.zeros((16, 16), dtype=np.int64)
    tile_classes = 1 + ((np.indices((16, 16))[0] // 4 + np.indices((16, 16))[1] // 4) % 2)
    labels[:, :] = tile_classes
    cube[:, :, :] = tile_classes[:, :, None] * 0.3
    cube_path = tmp_path / "cube.npy"
    labels_path = tmp_path / "labels.npy"
    np.save(cube_path, cube)
    np.save(labels_path, labels)

    response = client.post("/train", json={
        "model": "svm",
        "task": "classification",
        "dataset": "Uploaded",
        "seed": 42,
        "cubePath": str(cube_path),
        "labelsPath": str(labels_path),
    })
    assert response.status_code == 200
    assert "confusion_matrix" in response.json()


def test_ground_truth_annotations_are_not_reported_as_predictions(tmp_path):
    annotations = {
        "categories": [{"id": 1, "name": "ship"}],
        "annotations": [
            {"category_id": 1, "bbox": [10, 20, 30, 40], "confidence": 0.9, "iou": 0.82}
        ],
    }
    path = tmp_path / "annotations.json"
    path.write_text(json.dumps(annotations), encoding="utf-8")
    response = client.post("/detect", json={
        "dataset": "M2SODAI",
        "hasObjectAnnotations": True,
        "confidenceThreshold": 0.5,
        "annotationsPath": str(path),
    })
    assert response.status_code == 422
    assert "cannot be treated as predictions" in response.json()["detail"]


def test_pdf_report_generation():
    response = client.post("/report/pdf", json={
        "experimentId": "exp-test",
        "dataset": "Synthetic",
        "model": "svm",
        "split": {"strategy": "spatial", "train": 0.6, "val": 0.2, "test": 0.2, "seed": 42},
        "metrics": {"OA": 0.94},
    })
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/pdf")
