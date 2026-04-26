import sys
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib

# Pour importer preprocess
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocess import preprocess_data

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
    print(f"Modèle {name} sauvegardé à {filename}")

if __name__ == "__main__":
    # Charger et prétraiter les données depuis preprocess.py
    print("Chargement et prétraitement des données...")
    X_train, X_test, y_train, y_test = preprocess_data()
    print(f"Données préchargées - Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Entraîner les modèles
    print("\nEntraînement des modèles...")
    trained_models = train_all_models(X_train, y_train)
    
    # Évaluer et sauvegarder le meilleur
    print("\nÉvaluation des modèles...")
    best_model = None
    best_score = 0
    for name, model in trained_models.items():
        y_pred = model.predict(X_test)
        score = accuracy_score(y_test, y_pred)
        print(f"{name} Accuracy: {score:.4f}")
        if score > best_score:
            best_score = score
            best_model = (model, name)
    
    if best_model:
        print(f"\nMeilleur modèle: {best_model[1]} avec une accuracy de {best_score:.4f}")
        save_best_model(best_model[0], best_model[1])
    else:
        print("Erreur: Aucun modèle n'a pu être créé.")