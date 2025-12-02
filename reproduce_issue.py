
import requests
import pandas as pd
import io
import zipfile

PHISHTANK_JSON = "http://data.phishtank.com/data/online-valid.json"
TRANCO_ZIP = "https://tranco-list.eu/top-1m.csv.zip"

def test_phishtank():
    print("Testing PhishTank...")
    try:
        r = requests.get(PHISHTANK_JSON, timeout=15)
        r.raise_for_status()
        data = r.json()
        print(f"PhishTank: Success, got {len(data)} entries")
    except Exception as e:
        print(f"PhishTank: Failed - {e}")

def test_tranco():
    print("Testing Tranco...")
    try:
        r = requests.get(TRANCO_ZIP, timeout=15)
        r.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            with z.open("top-1m.csv") as f:
                top = pd.read_csv(f, header=None, names=["rank", "domain"])
        print(f"Tranco: Success, got {len(top)} entries")
    except Exception as e:
        print(f"Tranco: Failed - {e}")

if __name__ == "__main__":
    test_phishtank()
    test_tranco()
