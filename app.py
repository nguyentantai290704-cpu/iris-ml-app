import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO

# Load model
model = joblib.load("iris_model.pkl")
classes = ["Setosa", "Versicolor", "Virginica"]

st.title("🌸 Ứng dụng dự đoán loài hoa Iris")

# --- 1. Dự đoán nhập tay ---
st.header("🔹 Dự đoán với dữ liệu nhập tay")
sepal_length = st.number_input("Sepal length", 0.0, 10.0, 5.1)
sepal_width = st.number_input("Sepal width", 0.0, 10.0, 3.5)
petal_length = st.number_input("Petal length", 0.0, 10.0, 1.4)
petal_width = st.number_input("Petal width", 0.0, 10.0, 0.2)

if st.button("🔍 Dự đoán 1 mẫu"):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(features)[0]
    st.success(f"Kết quả: {classes[prediction]}")

# --- 2. Upload file để dự đoán nhiều mẫu ---
st.header("🔹 Dự đoán với file Excel/CSV")
uploaded_file = st.file_uploader("Tải lên file dữ liệu (CSV/Excel)", type=["csv", "xlsx"])

if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.write("📊 Dữ liệu tải lên:")
        st.dataframe(df.head())

        # Dự đoán
        predictions = model.predict(df.values)
        df["Prediction"] = [classes[p] for p in predictions]

        st.write("✅ Kết quả dự đoán:")
        st.dataframe(df)

        # Xuất file Excel
        output = BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            df.to_excel(writer, index=False, sheet_name="Results")
        st.download_button(
            label="⬇️ Tải file kết quả (.xlsx)",
            data=output.getvalue(),
            file_name="iris_predictions.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        # Biểu đồ
        st.subheader("📈 Biểu đồ phân bố dự đoán")
        fig, ax = plt.subplots()
        df["Prediction"].value_counts().plot(kind="bar", ax=ax)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Lỗi khi xử lý file: {e}")
