"""
SONAR - Étape 1: Exploration des données (EDA)
Auteur: Membre "Exploration & Data"
Rôle: Charger et analyser le dataset sonar
"""

# ==================== IMPORTS ====================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==================== CONFIGURATION ====================
# Pour voir toutes les colonnes
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

print("="*60)
print("🔍 SONAR - EXPLORATION DES DONNÉES (EDA)")
print("="*60)

# ==================== 1. CHARGEMENT DES DONNÉES ====================
print("\n📂 1. CHARGEMENT DES DONNÉES...")

# Chemin vers le fichier de données
fichier_data = "data/sonar.all-data.csv"

# Vérifier si le fichier existe
if not os.path.exists(fichier_data):
    print(f"❌ ERREUR: Le fichier {fichier_data} n'existe pas!")
    print("   Assurez-vous d'avoir téléchargé les données dans le dossier data/")
    exit(1)

# Charger le fichier CSV (pas d'en-tête, car les colonnes n'ont pas de noms)
df = pd.read_csv(fichier_data, header=None)

print(f"✅ Données chargées avec succès!")
print(f"   📊 Dimensions: {df.shape[0]} lignes × {df.shape[1]} colonnes")

# ==================== 2. APERÇU DES DONNÉES ====================
print("\n👀 2. APERÇU DES DONNÉES:")
print("-" * 40)
print("Voici les 5 premières lignes:")
print(df.head())

print("\nVoici les 5 dernières lignes:")
print(df.tail())

# ==================== 3. INFORMATIONS GÉNÉRALES ====================
print("\nℹ️ 3. INFORMATIONS GÉNÉRALES:")
print("-" * 40)
print(f"Nombre total d'échantillons: {df.shape[0]}")
print(f"Nombre de caractéristiques: {df.shape[1] - 1}")  # -1 car la dernière colonne est la cible
print(f"La dernière colonne (colonne 60) est la cible: M (Mine) ou R (Roche)")
print(f"Les 60 premières colonnes sont les fréquences sonar")

print("\n📋 Types de données:")
print(df.dtypes.value_counts())

# ==================== 4. STATISTIQUES DESCRIPTIVES ====================
print("\n📊 4. STATISTIQUES DESCRIPTIVES:")
print("-" * 40)
print("Voici un résumé statistique des 60 caractéristiques:")
print(df.describe())

# ==================== 5. ANALYSE DE LA COLONNE CIBLE ====================
print("\n🎯 5. ANALYSE DE LA COLONNE CIBLE (dernière colonne):")
print("-" * 40)

# La dernière colonne est la cible (M ou R)
cible = df[60]
print("Distribution des classes:")
print(cible.value_counts())

# Compter le nombre de mines et de roches
nb_mines = (cible == 'M').sum()
nb_roches = (cible == 'R').sum()
total = len(cible)

print(f"\n🔴 Mines (M): {nb_mines} échantillons ({nb_mines/total*100:.1f}%)")
print(f"🔵 Roches (R): {nb_roches} échantillons ({nb_roches/total*100:.1f}%)")

# Vérification que le total est correct
print(f"\n📊 Total: {total} échantillons")

# ==================== 6. VÉRIFICATIONS SUPPLÉMENTAIRES ====================
print("\n🔍 6. VÉRIFICATIONS SUPPLÉMENTAIRES:")
print("-" * 40)

# Vérifier les valeurs manquantes
print("Valeurs manquantes par colonne:")
print(df.isnull().sum().sum())
if df.isnull().sum().sum() == 0:
    print("✅ Aucune valeur manquante détectée!")
else:
    print("⚠️ Des valeurs manquantes ont été détectées!")

# Vérifier les doublons
print(f"\nLignes dupliquées: {df.duplicated().sum()}")

# ==================== 7. VISUALISATION ====================
print("\n📈 7. CRÉATION DU GRAPHIQUE...")
print("-" * 40)

# Créer une figure avec deux sous-graphiques
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Graphique 1: Diagramme en barres
ax1 = axes[0]
cible.value_counts().plot(kind='bar', ax=ax1, color=['red', 'blue'], edgecolor='black')
ax1.set_title('Distribution des classes - Mines vs Roches', fontsize=14)
ax1.set_xlabel('Classe', fontsize=12)
ax1.set_ylabel("Nombre d'échantillons", fontsize=12)
ax1.set_xticklabels(['Mines (M)', 'Roches (R)'], rotation=0)

# Ajouter les valeurs sur les barres
for i, v in enumerate(cible.value_counts().values):
    ax1.text(i, v + 2, str(v), ha='center', fontsize=12, fontweight='bold')

# Graphique 2: Camembert
ax2 = axes[1]
cible.value_counts().plot(kind='pie', ax=ax2, autopct='%1.1f%%', 
                          colors=['red', 'blue'], startangle=90,
                          labels=['Mines (M)', 'Roches (R)'])
ax2.set_title('Répartition en pourcentage', fontsize=14)
ax2.set_ylabel('')  # Enlever le label y

# Ajuster la mise en page
plt.tight_layout()

# Sauvegarder le graphique
plt.savefig('distribution_classes_sonar.png', dpi=150)
print("✅ Graphique sauvegardé: distribution_classes_sonar.png")

# Afficher le graphique
plt.show()

# ==================== 8. RÉSUMÉ ====================
print("\n📋 8. RÉSUMÉ DE L'ANALYSE:")
print("="*60)
print(f"📊 Dataset: SONAR (Mines vs Roches)")
print(f"   • {total} échantillons au total")
print(f"   • {nb_mines} mines ({nb_mines/total*100:.1f}%)")
print(f"   • {nb_roches} roches ({nb_roches/total*100:.1f}%)")
print(f"   • 60 caractéristiques de fréquences sonar")
print(f"   • Pas de valeurs manquantes")
print(f"   • {df.duplicated().sum()} doublons")
print("="*60)

print("\n✅ ÉTAPE 1 TERMINÉE AVEC SUCCÈS!")
print("   Le fichier eda.py a bien analysé les données.")