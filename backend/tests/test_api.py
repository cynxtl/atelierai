"""Integration tests for FastAPI endpoints."""

from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "AtelierAI" in response.json()["message"]


def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["app_name"] == "AtelierAI"
    assert "cuda_available" in data


def test_insights_endpoint():
    response = client.get("/api/insights")
    assert response.status_code == 200
    data = response.json()
    assert "classes" in data
    assert len(data["classes"]) == 10
    assert "cgan" in data
    assert "vae" in data


def test_unsupported_generate_request():
    payload = {
        "prompt": "Show me 3 diamond rings",
        "model_type": "cgan",
    }
    response = client.post("/api/generate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is False
    assert "Unsupported item" in data["error"] or "ring" in (data.get("unsupported_requested") or "").lower()


def test_generate_with_disclosed_quantity_adjustment():
    payload = {
        "prompt": "Synthesize 15 casual trousers",
        "model_type": "cgan",
    }
    response = client.post("/api/generate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["count"] == 8
    assert data["warning"] is not None
    assert "exceeds" in data["warning"].lower()


def test_generate_with_invalid_negative_quantity():
    payload = {
        "prompt": "Generate -4 summer dresses",
        "model_type": "cgan",
    }
    response = client.post("/api/generate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is False
    assert "quantity" in data["error"].lower()


def test_compare_endpoint():
    payload = {
        "prompt": "Show me 2 winter coats",
        "seed": 42,
    }
    response = client.post("/api/compare", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["display_label"] == "Coat"
    assert "cgan" in data and data["cgan"]["success"] is True
    assert "vae" in data and data["vae"]["success"] is True
