import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, davies_bouldin_score
from model import preprocessing, clustering

st.title("Clustering App")

file = st.file_uploader("Upload your dataset",type=["csv"])
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

if len(features)<2:
    st.warning("Select at least 2 features")
    st.stop()

n_clusters = st.slider("Number of clusters",2,10,3)
linkage = st.selectbox("Select linkage",["ward","complete","single"])

X = preprocessing(df,features)
model, labels = clustering(X,n_clusters,linkage)

df["Cluster"] = labels
if len(features) == 2 and all(pd.api.types.is_numeric_dtype(df[f]) for f in features):
    fig, ax = plt.subplots()
    ax.scatter(
        df[features[0]],
        df[features[1]],
        c=df["Cluster"]
    )
    ax.set_xlabel(features[0])
    ax.set_ylabel(features[1])

    st.pyplot(fig)

elif len(features) == 2:
    st.warning("Las dos variables deben ser numéricas para graficar")

else:
    st.info("Select exactly 2 features for visualization")
    
st.subheader("Clustered Data")
st.dataframe(df)


