import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

def train_and_save_model():
    data = load_iris()
    X, y = data.data, data.target

    model = RandomForestClassifier()
    model.fit(X, y)

    joblib.dump(model, 'model.pkl')
    print("Modelo entrenado y guardado como model.pkl")

if __name__ == "__main__":
    train_and_save_model()
