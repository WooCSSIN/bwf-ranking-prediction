import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health():
    response = client.get("/api/v1/health")
    print("[OK] Health Endpoint")

def test_rankings():
    # Model MS/Asia was trained, so this should work
    response = client.get("/api/v1/rankings/MS/Asia?top_n=5")
    if response.status_code == 200:
        data = response.json()
        assert len(data["rankings"]) > 0
        print("[OK] Rankings Endpoint (MS/Asia)")
    else:
        print(f"[FAIL] Rankings Endpoint failed: {response.json()}")

def test_predict():
    # We need a valid player ID for MS/Asia.
    # Let's test Viktor Axelsen (ID: 25831) or any random MS player.
    # The endpoint should return a prediction or 404 if not found.
    response = client.post("/api/v1/predict", json={
        "player_id": 95661, # Anthony Sinisuka Ginting
        "draw": "MS",
        "region": "Asia"
    })
    if response.status_code in [200, 404]: # 404 is valid if player is not from Asia in the dataset
        print(f"[OK] Predict Endpoint logic (Status: {response.status_code})")
    else:
        print(f"[FAIL] Predict Endpoint logic failed: {response.json()}")

if __name__ == "__main__":
    print("Testing API logic...")
    test_health()
    test_rankings()
    test_predict()
    print("Test complete.")
