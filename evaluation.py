import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from preprocess import preprocess_data
import numpy as np

def run_evaluation(model_filename):
    """
    Charge le modèle, effectue des prédictions et affiche les analyses de performance.
    """
    # 1. Récupération des données (via le script de Preprocessing)
    _, X_test, _, y_test = preprocess_data()

    # 2. Chargement du modèle
    base_dir = os.path.dirname(__file__)
    # On cherche dans le dossier 'models' à la racine du projet
    model_path = os.path.normpath(os.path.join(base_dir, 'models', model_filename))
    
    if not os.path.exists(model_path):
        print(f"Erreur : Le fichier {model_filename} est introuvable dans {model_path}")
        return

    model = joblib.load(model_path)
    print(f"\n✅ Modèle chargé : {model_filename}")
    print(f"📊 Nombre d'échantillons de test : {len(y_test)}")

    # 3. Prédictions
    y_pred = model.predict(X_test)

    # 4. Calcul du score de précision
    accuracy = accuracy_score(y_test, y_pred)
    print(f"📈 Précision Globale (Accuracy) : {accuracy:.2%}")

    # 5. Rapport détaillé (Precision, Recall, F1)
    # Rappel : 0 = Mine (M), 1 = Roche (R) selon le Preprocessing
    print("\n📝 Rapport de Classification :")
    print(classification_report(y_test, y_pred, target_names=['Mine (M)', 'Roche (R)']))

    # 6. Matrice de Confusion
    cm = confusion_matrix(y_test, y_pred)
    
    # Analyse de l'erreur grave : Mine (0) prédite comme Roche (1)
    # Dans la matrice cm, c'est l'indice [0][1]
    mines_ratees = cm[0][1]
    
    print("-" * 30)
    if mines_ratees > 0:
        print(f"⚠️  ALERTE SÉCURITÉ : {mines_ratees} mine(s) non détectée(s) !")
        print("Le sonar a confondu une mine avec une simple roche. Risque élevé.")
    else:
        print("🛡️  SÉCURITÉ : Aucune mine n'a été manquée par le système.")
    print("-" * 30)

    # 7. Visualisation graphique
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Prédit Mine', 'Prédit Roche'],
                yticklabels=['Réel Mine', 'Réel Roche'])
    plt.title(f'Matrice de Confusion\nModèle: {model_filename}')
    plt.xlabel('Prédiction de l\'IA')
    plt.ylabel('Réalité (Terrain)')
    
    # Sauvegarde automatique du résultat pour le rapport
    plt.savefig(os.path.join(base_dir, 'confusion_matrix_result.png'))
    print(f"🖼️  Graphique sauvegardé : confusion_matrix_result.png")
    plt.show()

if __name__ == "__main__":
    print("="*50)
    print("🔬 MODULE D'ÉVALUATION - CONTRÔLEUR QUALITÉ")
    print("="*50)
    
    try:
        # Recherche automatique du meilleur modèle KNN ou Random Forest dans le dossier models
        models_dir = os.path.join(os.path.dirname(__file__), 'models')
        if not os.path.exists(models_dir):
            print("❌ Erreur : Le dossier 'models' n'existe pas.")
        else:
            available_models = [f for f in os.listdir(models_dir) if f.endswith('_model.pkl')]
            if not available_models:
                print("⚠️ Aucun modèle trouvé dans 'models/'. Lancez d'abord model.py.")
            else:
                # On prend le premier modèle trouvé (celui sauvegardé par model.py)
                target_model = available_models[0]
                run_evaluation(target_model)

    except Exception as e:
        print(f"💥 Erreur critique lors de l'évaluation : {e}")