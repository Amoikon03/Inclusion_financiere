# Import des librairies
import streamlit as st
import pandas as pd


# Personnalisation de l'affichage
st.title("Exploration des données")

# Charger le fichier CSV
@st.cache_data
def load_data():
    return pd.read_csv("Inclusion.csv")

df = load_data()

# Afficher la forme du DataFrame (nombre de lignes et de colonnes)
st.write("### Forme du DataFrame : ")
st.write(f"Le nombre de lignes dans le DataFrame est : {df.shape[0]}")
st.write(f"Le nombre de colonnes dans le DataFrame est : {df.shape[1]}")

# Afficher les premières lignes du DataFrame
st.write("### Premières lignes du DataFrame : ")
st.write(df.head())

# Afficher les informations sur les (type de données, valeurs manquantes, etc.)
st.write(df.info())

# Vérification des valeurs manquantes
st.write("### **Valeurs manquantes par colonne :**")
missing_values = df.isnull().sum()
st.write(missing_values[missing_values > 0])

# Vérification des valeurs aberrantes
st.write("### **Les valeurs aberrantes :**")
for column in df.select_dtypes(include=['int64', 'float64']).columns:
    outliers = df[column][((df[column] - df[column].mean()) / df[column].std()).abs() > 3]
    if not outliers.empty:
        st.write(f"Valeurs aberrantes dans la colonne '{column}':")
        st.write(outliers)

# Afficher les statistiques descriptives pour les colonnes numériques
st.write("### **Statistiques descriptives pour les colonnes numériques :**")
st.write(df.describe())

# Lien vers les autres pages ou sections
st.subheader("Des hyperliens vers d'autres pages ou sections")
st.write("""
- [Informations](http://localhost:8501/Information)
- [Manipulation des données](http://localhost:8501/Manipulation_des_donn%C3%A9es)
- [Modèle de prédiction](http://localhost:8501/Modele_de_prediction)
- [Nous Contacter](http://localhost:8501/A_propos_de_nous)
- [A Propos De Nous](http://localhost:8501/A_propos_de_nous)
""")
