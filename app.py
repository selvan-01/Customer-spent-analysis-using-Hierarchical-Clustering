# ============================================
# Customer Spend Analysis - Streamlit App
# Hierarchical Clustering Dashboard
# ============================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as clus
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import AgglomerativeClustering

# ---------- Page Config ----------
st.set_page_config(page_title="Customer Segmentation", layout="wide")

# ---------- Custom CSS ----------
st.markdown("""
    <style>
    .main {
        background: linear-gradient(to right, #1e3c72, #2a5298);
        color: white;
    }
    h1 {
        text-align: center;
        color: #ffffff;
    }
    .stButton>button {
        background-color: #ff4b2b;
        color: white;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# ---------- Title ----------
st.title("🧠 Customer Spend Analysis using Hierarchical Clustering")

# ---------- Sidebar ----------
st.sidebar.header("⚙️ Controls")

uploaded_file = st.sidebar.file_uploader("📁 Upload Dataset", type=["csv"])

n_clusters = st.sidebar.slider("Select Number of Clusters", 2, 10, 5)

# ---------- Load Data ----------
if uploaded_file is not None:
    dataset = pd.read_csv(uploaded_file)

    st.subheader("📊 Dataset Preview")
    st.dataframe(dataset.head())

    # ---------- Preprocessing ----------
    if 'Gender' in dataset.columns:
        le = LabelEncoder()
        dataset['Gender'] = le.fit_transform(dataset['Gender'])

    st.success("✅ Data Preprocessed Successfully")

    # ---------- Dendrogram ----------
    st.subheader("🌳 Dendrogram (Cluster Selection)")

    fig1 = plt.figure(figsize=(10, 5))
    clus.dendrogram(clus.linkage(dataset, method='ward'))
    plt.title("Dendrogram")
    plt.xlabel("Customers")
    plt.ylabel("Distance")

    st.pyplot(fig1)

    # ---------- Model ----------
    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric='euclidean',
        linkage='average'
    )

    y_means = model.fit_predict(dataset)

    # ---------- Cluster Visualization ----------
    st.subheader("📌 Customer Segmentation")

    # Select Income & Spending columns (change if needed)
    X = dataset.iloc[:, [3, 4]].values

    fig2 = plt.figure(figsize=(8, 6))

    colors = ['purple', 'orange', 'red', 'green', 'blue', 'cyan', 'yellow', 'pink', 'brown', 'gray']

    for i in range(n_clusters):
        plt.scatter(
            X[y_means == i, 0],
            X[y_means == i, 1],
            s=50,
            c=colors[i],
            label=f'Cluster {i+1}'
        )

    plt.title("Income vs Spending Clusters")
    plt.xlabel("Income")
    plt.ylabel("Spending Score")
    plt.legend()

    st.pyplot(fig2)

    # ---------- Insights ----------
    st.subheader("💡 Insights")

    st.info("""
    ✔️ Customers are segmented based on income and spending behavior  
    ✔️ Helps businesses target specific customer groups  
    ✔️ Useful for marketing strategies and personalization  
    """)

else:
    st.warning("⚠️ Please upload a dataset to proceed.")

# ---------- Footer ----------
st.markdown("---")
st.markdown("🚀 Built with Streamlit | Hierarchical Clustering Project")