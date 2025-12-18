import streamlit as st
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# -----------------------------
# App Title
# -----------------------------
st.title("🛒 Market Basket Analysis (Apriori)")
st.write("Find frequent itemsets and association rules")

# -----------------------------
# Upload File
# -----------------------------
uploaded_file = st.file_uploader("groceries.excel", type=["xlsx"])

if uploaded_file is not None:

    # Load dataset
    df = pd.read_excel(uploaded_file, header=None)

    st.subheader("Raw Dataset Preview")
    st.dataframe(df.head())

    # -----------------------------
    # Convert to Transactions
    # -----------------------------
    transactions = []
    for i in range(len(df)):
        transactions.append(
            df.iloc[i].dropna().tolist()
        )

    st.subheader("Sample Transactions")
    st.write(transactions[:5])

    # -----------------------------
    # User Inputs
    # -----------------------------
    st.subheader("🔧 Apriori Parameters")

    min_support = st.slider(
        "Minimum Support",
        min_value=0.01,
        max_value=0.5,
        value=0.05,
        step=0.01
    )

    min_confidence = st.slider(
        "Minimum Confidence",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.1
    )

    # -----------------------------
    # Transaction Encoding
    # -----------------------------
    te = TransactionEncoder()
    te_array = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_array, columns=te.columns_)

    # -----------------------------
    # Apply Apriori
    # -----------------------------
    frequent_itemsets = apriori(
        df_encoded,
        min_support=min_support,
        use_colnames=True
    )

    st.subheader("📦 Frequent Itemsets")
    st.dataframe(frequent_itemsets)

    # -----------------------------
    # Association Rules
    # -----------------------------
    rules = association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=min_confidence
    )

    st.subheader("🔗 Association Rules")
    st.dataframe(
        rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
    )

else:
    st.warning("⬆️ Please upload groceries.xlsx file")
