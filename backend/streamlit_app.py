import streamlit as st
import pandas as pd
import numpy as np
import requests       

st.title("SmartHire AI Dashboard")

# Example data
st.write(pd.DataFrame(np.random.randn(5, 3), columns=["A", "B", "C"]))

# Optional: Fetch data from FastAPI


data = requests.get("http://127.0.0.1:8000/").json()

st.subheader("Backend Status")
st.success(data["message"])
