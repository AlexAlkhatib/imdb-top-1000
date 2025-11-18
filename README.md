# 🎬 IMDB Movie Recommendation System

### *Système de recommandation de films basé sur les similarités de contenus (Content-Based Filtering)*


## 📌 Description du Projet

Ce projet a pour objectif de recommander des films similaires à partir du dataset **IMDB Top 1000 Movies (Kaggle)**.
Le système repose sur un **modèle de recommandation content-based**, utilisant les attributs suivants :

* 🎭 **Genres**
* 📅 **Année de sortie**
* ⏱️ **Durée (runtime)**
* ⭐ **Notes IMDB & Metascore**
* 🗳️ **Nombre de votes**
* 💰 **Box-office (Gross)**
* 🎬 **Réalisateur**
* 🎭 **Acteurs principaux (Star1–Star4)**

L’ensemble du pipeline inclut :

✔️ Prétraitement des données
✔️ Encodage des variables catégorielles
✔️ Vectorisation multi-label des genres
✔️ Normalisation
✔️ Construction d’un espace vectoriel de films
✔️ Recommandation via **K-Nearest Neighbors**


## 📂 Dataset

**Dataset utilisé :** *Top 1000 IMDB Movies*
👉 Source : Kaggle (Inductiveanks)

### Colonnes principales utilisées :

| Colonne       | Description                                   |
| ------------- | --------------------------------------------- |
| Genre         | Liste de genres (Action, Drama, Thriller…)    |
| Released_Year | Année de sortie (avec correction d’anomalies) |
| Runtime       | Durée du film (ex : “142 min”)                |
| IMDB_Rating   | Note IMDB                                     |
| Meta_score    | Score Metacritic                              |
| No_of_Votes   | Nombre de votes                               |
| Gross         | Revenus au box-office                         |
| Director      | Réalisateur                                   |
| Star1–Star4   | Acteurs principaux                            |


## 🧹 Prétraitement des Données

### 🛠️ Étapes appliquées

#### ✔️ Nettoyage des colonnes

* **Runtime** : suppression du “min”, conversion en entier
* **Gross** : suppression des virgules, conversion en float
* **Released_Year** : correction d’anomalie (`"PG"` → `1995`)

#### ✔️ Encodage des Genres

* Transformation en liste
* **MultiLabelBinarizer** pour obtenir un encodage multi-hot

#### ✔️ Encodage des variables catégorielles

Colonnes encodées :
`Director`, `Star1`, `Star2`, `Star3`, `Star4`

Méthode utilisée :
➡️ **OneHotEncoder (handle_unknown="ignore")**

#### ✔️ Construction de la matrice finale

Le dataset final contient **3901 features**, combinant :

* Variables numériques
* Genres encodés
* Réalisateurs et acteurs encodés


## 🤖 Modèle de Recommandation

### 📌 Approche : Content-Based Filtering

Le système recommande des films ayant **des caractéristiques similaires** au film demandé.

### 🔧 Algorithme utilisé :

➡️ **K-Nearest Neighbors (KNN)**
Métriques : *cosine* ou *euclidienne*

### 🔍 Fonctionnement

1. Chaque film est converti en **vecteur de 3901 dimensions**
2. Calcul des distances entre films
3. Retour des **k films les plus similaires**


## 📊 Exemple d’Utilisation

```python
def recommend(title, model, matrix, movies, n=5):
    idx = movies[movies["Series_Title"] == title].index[0]
    distances, indices = model.kneighbors(matrix[idx], n_neighbors=n+1)
    return movies.iloc[indices[0][1:]]
```

**Demande :**

```python
recommend("Inception", knn_model, df_final, df)
```


## 🚀 Technologies Utilisées

* Python 3.x
* pandas
* numpy
* scikit-learn
* matplotlib / seaborn
* jupyter notebook


## ▶️ Lancer le Projet

### 1️⃣ Cloner le dépôt

```bash
git clone https://github.com/username/imdb-recommender.git
cd imdb-recommender
```

### 2️⃣ Installer les dépendances

```bash
pip install -r requirements.txt
```

### 3️⃣ Exécuter le notebook ou le script d’entraînement

```bash
jupyter notebook IMDB_Recommender.ipynb
```

ou

```bash
python train_recommender.py
```

### 4️⃣ Tester une recommandation

```bash
python recommend.py --title "The Matrix"
```


## ✨ Améliorations Futures

* Ajout d’un modèle **TF-IDF** sur *Overview* (description des films)
* Recommandation **hybride** (contenu + collaborative filtering)
* Interface Web via **Streamlit / FastAPI**
* Visualisation des clusters (PCA, t-SNE)
* Pondération dynamique des features (votes, notes, genres…)


## 👤 Auteur

**Alex Alkhatib**
*Projet Machine Learning — Système de Recommandation IMDB*


## 📄 Licence
MIT License
Copyright (c) 2025 Alex Alkhatib
