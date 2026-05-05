import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import silhouette_score, davies_bouldin_score
from model import preprocessing, clustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


st.title("Clustering App")

file = st.file_uploader("Upload your dataset", type=["csv"])

if file is not None:
    df = pd.read_csv(file)
else:
    st.stop()

st.subheader("Dataset Preview")
st.dataframe(df.head())

features = st.multiselect(
    "Select features for clustering",
    df.columns.tolist()
)

if len(features) < 2:
    st.warning("Select at least 2 features")
    st.stop()

n_clusters = st.slider("Number of clusters", 2, 10, 3)
linkage = st.selectbox("Select linkage", ["ward", "complete", "single"])

X = preprocessing(df, features)
model, labels = clustering(X, n_clusters, linkage)

df["Cluster"] = labels


# =========================
# VISUALIZACIÓN CON PCA
# =========================

fig, ax = plt.subplots()

# tomar SOLO variables numéricas seleccionadas
numeric_features = [f for f in features if pd.api.types.is_numeric_dtype(df[f])]

if len(numeric_features) >= 2:

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[numeric_features])

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    scatter = ax.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=df["Cluster"],
        cmap="viridis"
    )

    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_title("Clusters con PCA")

    plt.colorbar(scatter, ax=ax)

    st.pyplot(fig)

else:
    st.warning("Selecciona al menos 2 variables numéricas para usar PCA")

# =========================

st.subheader("Clustered Data")
st.dataframe(df)