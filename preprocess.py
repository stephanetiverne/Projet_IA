import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


def preprocess_data():

    # Charger le dataset
    df = pd.read_csv("data/sonar.all-data.csv", header=None)

    # Séparer les données
    X = df.iloc[:, 0:60]
    y = df.iloc[:, 60]

    # Transformer M et R en nombres
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    # Normaliser les données
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Séparer train et test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test


# Test du script
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = preprocess_data()

    print("Train:", X_train.shape)
    print("Test:", X_test.shape)