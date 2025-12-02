
import pickle
import os
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError

MODEL_PATH = "artifacts/model_v1.pkl"

def inspect_model():
    if not os.path.exists(MODEL_PATH):
        print(f"No model found at {MODEL_PATH}")
        return

    try:
        with open(MODEL_PATH, "rb") as f:
            payload = pickle.load(f)
        
        vect = payload.get("vectorizer")
        models = payload.get("models")
        
        print("Payload keys:", payload.keys())
        
        if vect:
            print(f"Vectorizer type: {type(vect)}")
            try:
                check_is_fitted(vect)
                print("Vectorizer is fitted.")
            except NotFittedError as e:
                print(f"Vectorizer is NOT fitted: {e}")
            except Exception as e:
                print(f"Error checking vectorizer: {e}")
        else:
            print("No vectorizer found in payload.")

        if models:
            for name, model in models.items():
                print(f"Checking model: {name}")
                try:
                    check_is_fitted(model)
                    print(f"  {name} is fitted.")
                except NotFittedError as e:
                    print(f"  {name} is NOT fitted: {e}")
    except Exception as e:
        print(f"Failed to load pickle: {e}")

if __name__ == "__main__":
    inspect_model()
