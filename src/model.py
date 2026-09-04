from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import pandas as pd
import os

def get_models():
    """
    Définit les trois algorithmes à tester.
    """
    models = {
        "Random_Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(kernel='linear', probability=True, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5)
    }
    return models

def train_all_models(X_train, y_train):
    """
    Entraîne les trois modèles et les stocke dans un dictionnaire.
    """
    models = get_models()
    trained_models = {}
    
    for name, model in models.items():
        print(f"Entraînement de {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
        
    return trained_models

def save_best_model(model, name):
    """ Sauvegarde le modèle choisi """
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(models_dir, exist_ok=True)
    filename = os.path.join(models_dir, f'{name}_model.pkl')
    joblib.dump(model, filename)
    print(f"Modèle {name} sauvegardé !")

if __name__ == "__main__":
    # Charger les données
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'sonar.all-data.csv')
    data = pd.read_csv(data_path, header=None)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Entraîner les modèles
    trained_models = train_all_models(X_train, y_train)
    
    # Évaluer et sauvegarder le meilleur
    best_model = None
    best_score = 0
    for name, model in trained_models.items():
        y_pred = model.predict(X_test)
        score = accuracy_score(y_test, y_pred)
        print(f"{name} Accuracy: {score}")
        if score > best_score:
            best_score = score
            best_model = (model, name)
    
    if best_model:
        save_best_model(best_model[0], best_model[1])