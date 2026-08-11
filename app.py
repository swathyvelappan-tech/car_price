import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Car Price Category Prediction",
    page_icon="🚗",
    layout="wide"
)

# =====================================================
# LOAD MODEL
# =====================================================

model = joblib.load("car_price_model.pkl")

# =====================================================
# LOAD DATASET
# =====================================================

df = pd.read_csv("car_price.csv")

# Same cleaning as training
df = df.drop_duplicates()

df = df.fillna(
    df.median(numeric_only=True)
)

df = df.fillna(
    df.mode().iloc[0]
)

# =====================================================
# ENCODING
# =====================================================

encoders = {}

for c in df.select_dtypes("object"):

    le = LabelEncoder()

    df[c] = le.fit_transform(df[c])

    encoders[c] = le

# =====================================================
# FEATURE SELECTION
# =====================================================

corr = df.corr()["Price_Category"].abs()

features = (
    corr[corr > 0.1]
    .index
    .drop("Price_Category")
    .tolist()
)

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        135deg,
        #eef2ff,
        #f8fafc
    );
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    padding: 10px;
}

.subtitle {
    text-align: center;
    color: #64748b;
    font-size: 18px;
    margin-bottom: 25px;
}

.card {
    background: white;
    padding: 25px;
    border-radius: 20px;
    box-shadow: 0px 5px 20px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}

.result {
    background: white;
    padding: 30px;
    border-radius: 20px;
    text-align: center;
    box-shadow: 0px 5px 20px rgba(0,0,0,0.10);
    margin-top: 25px;
}

.result-title {
    font-size: 20px;
    color: #64748b;
}

.result-value {
    font-size: 36px;
    font-weight: bold;
    margin-top: 10px;
}

.footer {
    text-align: center;
    color: #64748b;
    padding: 20px;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# HEADER
# =====================================================

st.markdown(
    '<div class="title">🚗 Car Price Category Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Decision Tree Machine Learning System</div>',
    unsafe_allow_html=True
)

st.divider()

# =====================================================
# INPUT SECTION
# =====================================================

st.markdown(
    '<div class="card">',
    unsafe_allow_html=True
)

st.subheader("🚘 Enter Car Details")

input_values = {}

# =====================================================
# CREATE INPUTS BASED ON ACTUAL FEATURES
# =====================================================

for feature in features:

    # -----------------------------------------------
    # CATEGORICAL FEATURE
    # -----------------------------------------------

    if feature in encoders:

        options = list(
            encoders[feature].classes_
        )

        input_values[feature] = st.selectbox(
            f"🔹 {feature.replace('_', ' ')}",
            options
        )

    # -----------------------------------------------
    # NUMERIC FEATURE
    # -----------------------------------------------

    else:

        min_value = float(df[feature].min())
        max_value = float(df[feature].max())
        default_value = float(df[feature].median())

        input_values[feature] = st.number_input(
            f"🔹 {feature.replace('_', ' ')}",
            min_value=min_value,
            max_value=max_value,
            value=default_value
        )

st.markdown(
    "</div>",
    unsafe_allow_html=True
)

# =====================================================
# PREDICTION BUTTON
# =====================================================

if st.button(
    "🔮 PREDICT PRICE CATEGORY",
    use_container_width=True
):

    # Create input DataFrame
    input_df = pd.DataFrame(
        [input_values]
    )

    # -----------------------------------------------
    # Encode categorical values
    # -----------------------------------------------

    for column in input_df.columns:

        if column in encoders:

            input_df[column] = encoders[
                column
            ].transform(
                input_df[column]
            )

    # -----------------------------------------------
    # Correct feature order
    # -----------------------------------------------

    input_df = input_df[features]

    # -----------------------------------------------
    # Prediction
    # -----------------------------------------------

    prediction = model.predict(
        input_df
    )[0]

    # -----------------------------------------------
    # Convert prediction to original category
    # -----------------------------------------------

    target_encoder = encoders.get(
        "Price_Category"
    )

    if target_encoder:

        prediction = target_encoder.inverse_transform(
            [prediction]
        )[0]

    # =================================================
    # RESULT
    # =================================================

    st.markdown(
        f"""
        <div class="result">

        <div class="result-title">
        🎉 Prediction Completed
        </div>

        <div class="result-value">
        🚗 {prediction}
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.success(
        "✅ Car price category predicted successfully!"
    )

    st.balloons()

# =====================================================
# SHOW SELECTED FEATURES
# =====================================================

with st.expander("🔍 View Model Features"):

    st.write(
        "Features used by Decision Tree:"
    )

    st.write(features)

# =====================================================
# FOOTER
# =====================================================

st.divider()

st.markdown(
    """
    <div class="footer">

    🚗 <b>Car Price Category Prediction</b>
    <br>
    Powered by Decision Tree Classifier

    </div>
    """,
    unsafe_allow_html=True
)