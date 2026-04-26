# Projet_IA - Classification Sonar

## Description
Projet d'apprentissage automatique pour la classification de signaux sonar (minéraux vs roches) utilisant l'ensemble de données Sonar.

## Structure du projet
```
Projet_IA/
├── data/
│   └── sonar.all-data.csv          # Dataset sonar avec 60 attributs
├── models/                          # Modèles entraînés (générés)
├── src/
│   └── model.py                     # Script d'entraînement des modèles
├── preprocess.py                    # Script de prétraitement des données
├── README.md                        # Ce fichier
└── .gitignore                       # Fichiers ignorés par Git
```

## Données
- **Dataset**: sonar.all-data.csv
- **Attributs**: 60 colonnes numériques
- **Target**: Colonne 61 (M = minéral, R = roche)
- **Instances**: 208 enregistrements

## Prétraitement (preprocess.py)
Le script `preprocess.py` effectue les étapes suivantes:
1. Chargement du CSV
2. Séparation des features (X) et target (y)
3. Encodage des labels (M/R → 0/1)
4. Normalisation StandardScaler
5. Split train/test (80/20)

**Utilisation**:
```python
from preprocess import preprocess_data
X_train, X_test, y_train, y_test = preprocess_data()
```

## Modèles (src/model.py)
Le script `src/model.py` entraîne et évalue 3 classifieurs:
1. **Random Forest** (100 estimateurs)
2. **SVM** (kernel linéaire)
3. **KNN** (k=5)

**Utilisation**:
```bash
python src/model.py
```

Le script:
- Utilise les données prétraitées de `preprocess.py`
- Entraîne les 3 modèles
- Évalue leur performance sur l'ensemble test
- Sauvegarde le meilleur modèle dans `models/`

## Installation
```bash
pip install pandas scikit-learn joblib
```

## Dépendances
- pandas
- scikit-learn
- joblib

## Résultats attendus
Le script affiche l'accuracy de chaque modèle et sauvegarde le meilleur en tant que `.pkl`.