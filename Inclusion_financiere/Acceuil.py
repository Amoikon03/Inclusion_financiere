# Import des librairies
import streamlit as st
from PIL import Image

# Configuration de la page
st.set_page_config(
    page_title="Inclusion financière en Afrique",
    page_icon="☏"
)

# Fonction pour afficher une section avec titre et contenu
def section(titre, contenu):
    st.header(titre)
    st.write(contenu)

# Fonction pour afficher une image avec un titre en dessous
def image_with_caption(image_path, caption):
    img = Image.open(image_path)
    st.image(img, caption=caption, use_column_width=True)

# Fonction pour afficher un paragraphe justifié
def paragraphe(texte):
    st.write(f"<div style='text-align: justify'>{texte}</div>", unsafe_allow_html=True)

# Titre de page
st.title("Inclusion financière en Afrique")

# Image illustrative de l'application
image_with_caption("Inclusion-FINANCIERE.jpeg", " ")

# Description de l'application
paragraphe("""

Ce point de contrôle porte sur l'ensemble de données "Inclusion financière en Afrique", fourni dans le cadre de l'initiative d'inclusion financière en Afrique et hébergé par la plateforme «indi». L'ensemble de données comprend des informations démographiques sur environ 33 600 personnes en Afrique de l'Est, ainsi que les services financiers qu'elles utilisent.

L'objectif du modèle de machine learning est de prédire quels individus sont les plus susceptibles d'avoir ou d'utiliser un compte bancaire. L'inclusion financière vise à garantir que les individus et les entreprises ont accès à des produits et services financiers utiles et abordables, tels que les transactions, les paiements, les économies, le crédit et l'assurance, livrés de manière responsable et durable.

""")

# Fonctionnalités de l'application
section("Fonctionnalités de l'application", "")

# Informations sur les données
section("Informations sur les données", "Information sur les données")

# Exploration des données
section("Exploration des données", "Exploration des données")

# Manipulation des données
section("Manipulation des données", "Manipulations des données")

# Modélisation
section("Modélisation", "Création des modèles de machine learning et de deep learning")

# Contactez-nous
section("Contactez-Nous", "Prendre contact pour plus d'information")

# À Propos de Nous
section("À Propos De Nous", "Qui sommes nous et comment nous rejoindre")
