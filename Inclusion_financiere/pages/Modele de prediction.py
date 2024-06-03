import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


# Charger le DataFrame à partir d'un fichier pickle
def load_dataframe(filepath):
    with open(filepath, 'rb') as file:
        dataframe = pickle.load(file)
    return dataframe


# Entraîner les modèles
def train_models(data):
    # Sélectionner les caractéristiques et la cible
    X = data[
        ['age_of_respondent', 'gender_of_respondent', 'relationship_with_head', 'marital_status', 'education_level',
         'job_type', 'location']]
    y = data['bank_account']

    # Encodage des variables catégorielles
    le = LabelEncoder()
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = le.fit_transform(X[col])

    # Diviser les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entraîner le modèle SVM
    model_svm = SVC(probability=True)
    model_svm.fit(X_train, y_train)

    # Entraîner le modèle Random Forest
    model_rf = RandomForestClassifier()
    model_rf.fit(X_train, y_train)

    # Entraîner le modèle de régression logistique
    model_lr = LogisticRegression()
    model_lr.fit(X_train, y_train)

    return model_svm, model_rf, model_lr


# Charger les modèles
def load_models():
    model_svm = pickle.load(open("model_svm.pkl", 'rb'))
    model_rf = pickle.load(open("model_rf.pkl", 'rb'))
    model_lr = pickle.load(open("model_lr.pkl", 'rb'))
    return model_svm, model_rf, model_lr


# Sauvegarder les modèles
def save_models(model_svm, model_rf, model_lr):
    pickle.dump(model_svm, open("model_svm.pkl", 'wb'))
    pickle.dump(model_rf, open("model_rf.pkl", 'wb'))
    pickle.dump(model_lr, open("model_lr.pkl", 'wb'))


# Fonction pour prédire avec SVM
def predict_svm(model, features):
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # Effectuer la prédiction avec le modèle SVM
    prediction = model.predict_proba(scaled_features)[:, 1]
    return prediction


# Fonction pour prédire avec Random Forest
def predict_random_forest(model, features):
    imputer = SimpleImputer(strategy='mean')
    filled_features = imputer.fit_transform(features)
    prediction = model.predict_proba(filled_features)[:, 1]
    return prediction


# Fonction pour prédire avec la régression logistique
def predict_logistic_regression(model, features):
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    prediction = model.predict_proba(scaled_features)[:, 1]
    return prediction


# Définir le titre de l'application
st.title("Prédiction d'inclusion financière en Afrique")

# Charger le DataFrame à partir du fichier pickle
data = load_dataframe("Data_Frame.pkl")

# Entraîner et sauvegarder les modèles (à exécuter une seule fois)
# model_svm, model_rf, model_lr = train_models(data)
# save_models(model_svm, model_rf, model_lr)

# Charger les modèles sauvegardés
model_svm, model_rf, model_lr = load_models()

# Liste des villes
cities = ["Abidjan", "Bondoukou", "Abengourou", "Adzopé", "Soubré", "Gagnoa", "Yamoussoukro", "San-Pédro",
          "Bouna", "Korogho", "Man", "Katiola", "Ferkessédougou", "Vavoua", "Guiglo", "Méagui", "Akoupé", "Danané"]

# Créer une colonne latérale pour les paramètres
with st.sidebar:
    model_choice = st.selectbox("Modèle", ["SVM", "Random Forest", "Régression Logistique"])

    if model_choice == "SVM":
        kernel = st.selectbox("Noyau", ["linear", "poly", "rbf", "sigmoid"])
        C = st.number_input("C", min_value=0.1, max_value=10.0, step=0.1)
        gamma = st.number_input("Gamma", min_value=0.1, max_value=10.0, step=0.1)
    elif model_choice == "Random Forest":
        n_estimators = st.number_input("Nombre d'arbres", min_value=1, max_value=1000, step=1)
        max_depth = st.number_input("Profondeur maximale", min_value=1, max_value=100, step=1)
    elif model_choice == "Régression Logistique":
        penalty = st.selectbox("Régularisation", ["l1", "l2"])
        C = st.number_input("C", min_value=0.1, max_value=10.0, step=0.1)

    precision = st.number_input("Précision des probabilités", min_value=0, max_value=5, step=1)

with st.sidebar:
    st.header("Paramètres")
    age = st.number_input("Âge")
    education = st.selectbox("Niveau d'éducation", ["Primaire", "Secondaire", "Université"])
    monthly_income = st.number_input("Revenu mensuel (fcfa)")
    internet_access = st.radio("Accès à Internet", ["Oui", "Non"])
    occupation = st.selectbox("Occupation", ["Employé", "Indépendant", "Étudiant", "Sans emploi", "Autre"])
    marital_status = st.selectbox("Situation matrimoniale", ["Célibataire", "Marié(e)", "Divorcé(e)", "Veuf(ve)"])
    dependents = st.number_input("Nombre de personnes à charge")
    location = st.selectbox("Localisation géographique", cities)

    if st.button("Prédire"):
        features = []

        # Âge
        age = int(age)
        features.append(age)

        # Niveau d'éducation
        if education == "Primaire":
            education_encoded = 0
        elif education == "Secondaire":
            education_encoded = 1
        else:
            education_encoded = 2
        features.append(education_encoded)

        # Revenu mensuel
        monthly_income = float(monthly_income)
        features.append(monthly_income)

        # Accès à Internet
        if internet_access == "Oui":
            internet_access_encoded = 1
        else:
            internet_access_encoded = 0
        features.append(internet_access_encoded)

        # Occupation
        if occupation == "Employé":
            occupation_encoded = 0
        elif occupation == "Indépendant":
            occupation_encoded = 1
        elif occupation == "Étudiant":
            occupation_encoded = 2
        elif occupation == "Sans emploi":
            occupation_encoded = 3
        else:
            occupation_encoded = 4
        features.append(occupation_encoded)

        # Situation matrimoniale
        if marital_status == "Célibataire":
            marital_status_encoded = 0
        elif marital_status == "Marié(e)":
            marital_status_encoded = 1
        elif marital_status == "Divorcé(e)":
            marital_status_encoded = 2
        else:
            marital_status_encoded = 3
        features.append(marital_status_encoded)

        # Nombre de personnes à charge
        dependents = int(dependents)
        features.append(dependents)

        # Localisation géographique
        if location in cities:
            location_encoded = cities.index(location)
        else:
            st.error("La ville sélectionnée n'est pas valide.")
            location_encoded = -1
        features.append(location_encoded)

        if location_encoded != -1:
            # Convertir la liste en tableau NumPy
            features = np.array(features)

            # Restructurer le tableau pour correspondre aux attentes du modèle
            features = features.reshape(1, -1)

            # Prédiction en fonction du modèle choisi
            if model_choice == "SVM":
                prediction = predict_svm(model_svm, features)
            elif model_choice == "Random Forest":
                prediction = predict_random_forest(model_rf, features)
            elif model_choice == "Régression Logistique":
                prediction = predict_logistic_regression(model_lr, features)
            else:
                raise ValueError("Modèle non reconnu. Veuillez sélectionner un modèle valide.")

            # Afficher le résultat de la prédiction avec la précision souhaitée
            st.write(f"Probabilité d'avoir un compte bancaire : {prediction[0]:.{precision}f}")
