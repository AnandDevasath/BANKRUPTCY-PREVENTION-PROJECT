import os
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Bankruptcy Prediction System",
    page_icon="📊",
    layout="wide"
)

MODEL_PATH = r"C:\Users\D.Anand\Desktop\bankruptcy\bankruptcy_model.pkl"

# -----------------------------
# CUSTOM STYLE
# -----------------------------
st.markdown("""
    <style>
    .main-title {
        font-size: 40px;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-text {
        font-size: 18px;
        color: #b0b0b0;
        margin-bottom: 1rem;
    }
    .status-ok {
        color: #22c55e;
        font-weight: 600;
    }
    .status-bad {
        color: #ef4444;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    try:
        if not os.path.exists(MODEL_PATH):
            return None, f"Model file not found: {MODEL_PATH}"
        model = joblib.load(MODEL_PATH)
        return model, None
    except Exception as e:
        return None, str(e)

model, load_error = load_model()

# -----------------------------
# SESSION STATE
# -----------------------------
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

# -----------------------------
# HEADER
# -----------------------------
st.markdown('<div class="main-title">📊 Bankruptcy Prediction System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-text">Predict whether a company is likely to face Bankruptcy or Non-Bankruptcy based on risk indicators.</div>',
    unsafe_allow_html=True
)

col_status1, col_status2 = st.columns([2, 3])

with col_status1:
    if model is not None:
        st.markdown('<p class="status-ok">✅ Model Status: Loaded Successfully</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p class="status-bad">❌ Model Status: Not Loaded</p>', unsafe_allow_html=True)
        st.error(f"Error loading model: {load_error}")

with col_status2:
    st.info("Risk Level Guide: 0 = Low, 0.5 = Medium, 1 = High")

# -----------------------------
# SIDEBAR INPUTS
# -----------------------------
st.sidebar.header("Enter Risk Details")

industrial_risk = st.sidebar.selectbox("Industrial Risk", [0, 0.5, 1], index=0)
management_risk = st.sidebar.selectbox("Management Risk", [0, 0.5, 1], index=0)
financial_flexibility = st.sidebar.selectbox("Financial Flexibility", [0, 0.5, 1], index=0)
credibility = st.sidebar.selectbox("Credibility", [0, 0.5, 1], index=0)
competitiveness = st.sidebar.selectbox("Competitiveness", [0, 0.5, 1], index=0)
operating_risk = st.sidebar.selectbox("Operating Risk", [0, 0.5, 1], index=0)

st.sidebar.markdown("---")
predict_btn = st.sidebar.button("🔍 Predict")
clear_btn = st.sidebar.button("🗑️ Clear History")

if clear_btn:
    st.session_state.prediction_history = []
    st.sidebar.success("Prediction history cleared.")

# -----------------------------
# INPUT DATAFRAME
# -----------------------------
input_df = pd.DataFrame({
    "industrial_risk": [industrial_risk],
    "management_risk": [management_risk],
    "financial_flexibility": [financial_flexibility],
    "credibility": [credibility],
    "competitiveness": [competitiveness],
    "operating_risk": [operating_risk]
})

# -----------------------------
# MAIN LAYOUT
# -----------------------------
left_col, right_col = st.columns([1.1, 1])

with left_col:
    st.subheader("Selected Input Features")
    st.dataframe(input_df, use_container_width=True)

    st.markdown("### Sample CSV Download")
    sample_csv = input_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Current Input as CSV",
        data=sample_csv,
        file_name="bankruptcy_input_sample.csv",
        mime="text/csv"
    )

with right_col:
    st.subheader("Input Summary")
    st.metric("Industrial Risk", industrial_risk)
    st.metric("Management Risk", management_risk)
    st.metric("Financial Flexibility", financial_flexibility)
    st.metric("Credibility", credibility)
    st.metric("Competitiveness", competitiveness)
    st.metric("Operating Risk", operating_risk)

# -----------------------------
# PREDICTION SECTION
# -----------------------------
if predict_btn:
    st.markdown("---")
    st.subheader("Prediction Result")

    if model is None:
        st.warning("⚠️ Model not loaded. Please check the PKL file.")
    else:
        try:
            prediction = model.predict(input_df)[0]

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(input_df)[0]
                confidence = max(proba)
            else:
                confidence = None

            if prediction == 0:
                result_text = "Bankruptcy"
                st.error(f"🚨 Prediction: {result_text}")
            else:
                result_text = "Non-Bankruptcy"
                st.success(f"✅ Prediction: {result_text}")

            if confidence is not None:
                st.write(f"**Confidence Score:** {confidence:.2%}")
                st.progress(float(confidence))

            # Save history
            history_row = input_df.copy()
            history_row["Prediction"] = result_text
            history_row["Confidence"] = f"{confidence:.2%}" if confidence is not None else "N/A"

            st.session_state.prediction_history.append(history_row)

        except Exception as e:
            st.error(f"Prediction error: {e}")

# -----------------------------
# HISTORY SECTION
# -----------------------------
if st.session_state.prediction_history:
    st.markdown("---")
    st.subheader("Prediction History")

    history_df = pd.concat(st.session_state.prediction_history, ignore_index=True)
    st.dataframe(history_df, use_container_width=True)

    history_csv = history_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Prediction History",
        data=history_csv,
        file_name="prediction_history.csv",
        mime="text/csv"
    )

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
with st.expander("⚙️ Run Commands"):
    st.code("pip install streamlit scikit-learn pandas joblib")
    st.code("python -m streamlit run app.py")

with st.expander("ℹ️ About This App"):
    st.write("""
    This application predicts whether a company is likely to face **Bankruptcy** or **Non-Bankruptcy**
    using a machine learning model trained on risk-related business features:
    
    - Industrial Risk
    - Management Risk
    - Financial Flexibility
    - Credibility
    - Competitiveness
    - Operating Risk
    """)