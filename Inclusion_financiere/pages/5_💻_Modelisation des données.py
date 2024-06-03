import streamlit as st
import pandas as pd
import pickle as pk
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score


def afficher_data_modelisation():
    # Warnings
    st.set_option('deprecation.showfileUploaderEncoding', False)

    # Titre de la page
    st.title("Modélisation des données")

    # Chargement des données
    df = pk.load(open("data_frame.pkl", "rb"))

    # Créer un bouton pour afficher un message d'information
    if st.checkbox("**Cliquez ici pour masquer l'information**", value=True):
        # display the text if the checkbox returns True value
        st.info("**Sur cette page, vous avez accès à trois algorithmes de machine learning et de deep learning.**")
        # Utilisation de st.success() avec des balises HTML pour personnaliser le style
        st.markdown("""
        <div style='background-color:#00353F; padding:10px'>
          <h3 style='color:white'>En bas de cette section, vous trouverez les résultats issus du modèle sélectionné. </h3>
          <h3 style='color:white'>Cette page vous permet d'effectuer des prédictions ou des classifications.</h3>
        </div>
        """, unsafe_allow_html=True)

        st.write("")
        st.write("")

        def paragraphe(texte):
            st.write(f"<div style='text-align: justify'>{texte}</div>", unsafe_allow_html=True)

        st.markdown("<strong>La Régression Logistique</strong>", unsafe_allow_html=True)
        paragraphe("""
        Elle sera utilisée pour établir une relation entre les caractéristiques démographiques et financières des individus et la probabilité qu'ils possèdent un compte bancaire en Afrique de l'Est. En d'autres termes, elle servirait à modéliser comment ces caractéristiques influencent la présence ou l'absence d'un compte bancaire chez les individus de la région.
         """)

        st.write("")

        st.markdown("<strong>Le Modèle Random Forest</strong>", unsafe_allow_html=True)
        paragraphe("""
         Il sera utilisé pour prédire la probabilité qu'un individu ait un compte bancaire en se basant sur un ensemble de caractéristiques démographiques et financières. Contrairement à la régression logistique qui cherche une relation linéaire entre les caractéristiques et la probabilité d'avoir un compte bancaire, Random Forest utilise un ensemble d'arbres de décision pour capturer des relations non linéaires et des interactions complexes entre les caractéristiques. Ainsi, il pourrait être plus efficace pour modéliser des relations complexes et non linéaires dans les données, ce qui pourrait conduire à des prédictions plus précises sur l'inclusion financière des individus en Afrique de l'Est.
        """)

        st.write("")

        st.markdown("<strong>Le Modèle SVM(Support Vector Machine)</strong>", unsafe_allow_html=True)
        paragraphe("""
        Il sera utilisé pour prédire la probabilité qu'un individu ait un compte bancaire en fonction de ses caractéristiques démographiques et financières. Contrairement à la régression logistique et au Random Forest, SVM est un modèle qui cherche à trouver l'hyperplan qui sépare au mieux les différentes classes dans l'espace des caractéristiques. Il utilise des vecteurs de support pour déterminer cet hyperplan, ce qui lui permet de trouver des frontières de décision complexes, même dans des espaces de grande dimension. Ainsi, SVM pourrait être efficace pour modéliser des relations non linéaires et des interactions complexes entre les caractéristiques, ce qui en fait un autre choix potentiellement puissant pour prédire l'inclusion financière en Afrique de l'Est.
        """)


    # Les modèles disponibles
    model = st.sidebar.selectbox("Sélectionnez un modèle.",
                                 ["Regression Logistique", "Random Forest", "Support Vector Machine"])

    # ✂️ Selection et découpage des données
    seed = 123

    def select_split(dataframe, model):

        if model == "Regression Logistique":
            x = dataframe[['age_of_respondent', 'gender_of_respondent', 'relationship_with_head', 'marital_status', 'education_level', 'job_type']]
            y = dataframe["bank_account"]

        elif model == "Random Forest":
            x = dataframe[['age_of_respondent', 'gender_of_respondent', 'relationship_with_head', 'marital_status', 'education_level', 'job_type']]
            y = dataframe["bank_account"]

        elif model == "Support Vector Machine":
            x = dataframe[['age_of_respondent', 'gender_of_respondent', 'relationship_with_head', 'marital_status', 'education_level', 'job_type']]
            y = dataframe["bank_account"]

        else:
            raise ValueError(
                "Modèle non reconnu. Veuillez choisir parmi 'Regression Logistique', 'Random Forest' ou 'Support Vector Machine'.")

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=seed)
        return x_train, x_test, y_train, y_test

    # Création des variables d'entrainement et test
    # Modifier cet appel pour inclure le modèle sélectionné
    x_train, x_test, y_train, y_test = select_split(dataframe=df, model=model)

    # Conversion des séries pandas en tableaux bidimensionnels
    x_train = np.array(x_train).reshape(-1, 1)
    x_test = np.array(x_test).reshape(-1, 1)
    y_train = np.array(y_train).reshape(-1, 1)
    y_test = np.array(y_test).reshape(-1, 1)

    # ✏️ Afficher les graphiques de performance sans try et après avec try except
    def plot_perf(graphes):
        if "Confucsion matrix" in graphes:
            st.subheader("Matrice de confusion")
            try:
                ConfusionMatrixDisplay.from_estimator(model, x_test, y_test)
                st.pyplot()
            except Exception as e:
                st.warning(f"La Matrice de confusion ne peut pas être représenté avec les données du modèle: {str(e)}")

        if "ROC Curve" in graphes:
            st.subheader("La courbe ROC (Receiver Operating Characteristic)")
            try:
                RocCurveDisplay.from_estimator(model, x_test, y_test)
                st.pyplot()
                st.info("Une courbe ROC idéale se rapproche du coin supérieur gauche du graphique, "
                        "ce qui indique un modèle avec une sensibilité élevée et un faible taux de faux positifs.")
            except Exception as e:
                st.warning(f"La courbe ROC ne peut pas être représenté avec les données du modèle: {str(e)}")

        if "Precision_Recall Curve" in graphes:
            st.subheader("La courbe de Précision Recall)")
            try:
                PrecisionRecallDisplay.from_estimator(model, x_test, y_test)
                st.pyplot()
                st.info("La courbe PR met l'accent sur la capacité du modèle à bien classer les échantillons positifs, "
                        "ce qui est important dans les cas où les classes sont très déséquilibrées. ")

                st.write("")

                st.write("""Une AUC-PR proche de 1 indique un modèle idéal, 
                    où chaque prédiction positive est correcte et chaque prédiction négative est incorrecte. 
                    Proximité du coin supérieur droit : Comme pour la courbe ROC, une courbe PR idéale tend 
                    à se rapprocher du coin supérieur droit du graphique. Cela signifie que le modèle atteint 
                    à la fois une précision élevée et un rappel élevé pour un seuil de classification donné.
                    Croissance rapide de la courbe : Une courbe PR idéale augmente rapidement à partir de 
                    l'origine, ce qui signifie qu'elle atteint une haute précision pour un rappel relativement faible. 
                    Cela indique que le modèle est capable de bien classer les échantillons positifs 
                    dès le début de la prédiction. Pas de "dents de scie" : Une courbe PR idéale est lisse, 
                    sans "dents de scie" ou de fortes variations. Cela signifie que le modèle maintient 
                    une précision élevée même lorsqu'il rappelle un grand nombre d'échantillons positifs.
                    """)
            except Exception as e:
                st.warning(
                    f"La courbe de Précision Recall ne peut pas être représenté avec les données du modèle: {str(e)}")

            # 3️⃣ Regression logistique
            if model == "Regression logistique":
                st.sidebar.subheader("Les hyperparamètres du modèle")

                hyp_c = st.sidebar.number_input("Choisir la valeur du paramètre de régularisation", 0.01, 10.0)

                n_max_iter = st.sidebar.number_input("Choisir le nombre maximal d'itération", 100, 1000, step=10)

                graphes_perf = st.sidebar.multiselect(
                    "Choisir un ou des graphiques de performance du modèle à afficher",
                    ("Confusion matrix", "ROC Curve", "Precision_Recall Curve"))

                if st.sidebar.button("Prédire", key="logistic_regression"):
                    st.subheader("Résultat de la Regression logistique")

                    # Initialiser le modèle
                    model = LogisticRegression(C=hyp_c, max_iter=n_max_iter, random_state=seed)

                    # Entrainer le modèle
                    model.fit(x_train, y_train)

                    # Prédiction du modèle
                    y_pred = model.predict(x_test)

                    # Calcul des metrics de performances

                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, pos_label="oui")
                    recall = recall_score(y_test, y_pred, pos_label="oui")

                    # Afficher les métriques
                    st.write("Exactitude du modèle :", accuracy)
                    st.write("Précision du modèle :", precision)
                    st.write("Recall du modèle :", recall)

                    # Afficher les graphiques de performances
                    plot_perf(graphes_perf)


            # 4️⃣ Random Forest
            elif model == "Random Forest":
                st.sidebar.subheader("Hyperparameters for the Random Forest model")

                n_estimators = st.sidebar.number_input("Nombre d'estimateurs", 1, 1000, step=1)
                max_depth = st.sidebar.slider("Max depth of each tree", 1, 20, 10)

                graphes_perf = st.sidebar.multiselect("Select one or more performance graphs to display",
                                                      ["Confusion matrix", "ROC Curve", "Precision_Recall Curve"])

                if st.sidebar.button("Predict", key="random_forest"):
                    st.subheader("Random Forest Model Results")

                    # Initialize the Random Forest model
                    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=seed)

                    # Train the model
                    model.fit(x_train, y_train)

                    # Make predictions
                    y_pred = model.predict(x_test)

                    # Calculate performance metrics
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, average='micro')
                    recall = recall_score(y_test, y_pred, average='micro')

                    # Display metrics
                    st.write("Model Accuracy:", accuracy)
                    st.write("Model Precision:", precision)
                    st.write("Model Recall:", recall)


            # 5️⃣ Support Vector Machine
            elif model == "Support Vector Machine":
                st.sidebar.subheader("Les hyperparamètres du modèle")

                hyp_c = st.sidebar.number_input("Choisir la valeur du paramètre de régularisation", 0.01, 10.0)

                kernel = st.sidebar.radio("Choisir le noyau", ("rbf", "linear", "poly", "sigmoid"))

                gamma = st.sidebar.radio("Gamma", ("scale", "auto"))

                graphes_perf = "Confusion matrix"

                if st.sidebar.button("Prédire", key="classifivation multiclasse"):
                    st.subheader("Résultat de Support Vecteur Machine (SVM)")

                    # Initialiser le modèle svc pour la classification
                    model = SVC(C=hyp_c, kernel=kernel, gamma=gamma, decision_function_shape='ovo')

                    # Entrainer le modèle
                    model.fit(x_train, y_train)

                    # Prédiction du modèle
                    y_pred = model.predict(x_test)

                    # Calcul des metrics de performances
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred)
                    recall = recall_score(y_test, y_pred)

                    # Afficher les métriques
                    st.write("Exactitude du modèle :", accuracy)
                    st.write("Précision du modèle :", precision)
                    st.write("Recall du modèle :", recall)

                    # Afficher les graphiques de performances
                    plot_perf(graphes_perf)
                    st.write()


# Appel de la fonction afficher_data_modelisation pour exécuter l'application
afficher_data_modelisation()
