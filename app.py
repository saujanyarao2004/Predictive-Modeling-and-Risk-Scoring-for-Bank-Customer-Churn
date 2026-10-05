
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import streamlit.components.v1 as components

# ==================================================
# PAGE CONFIGURATION
# ==================================================
st.set_page_config(
    page_title="European Bank Customer Churn Analytics",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==================================================
# PROFESSIONAL BANKING THEME
# ==================================================
st.markdown("""
<style>
.stApp {
    background: #000000;
    color: #FFFFFF;
}

.main .block-container {
    padding: 1.2rem 2rem 2rem 2rem;
    max-width: 1600px;
}

html, body, [class*="css"] {
    font-family: "Segoe UI", Arial, sans-serif;
}

/* HEADER */
.hero-header {
    background: linear-gradient(135deg, #111111 0%, #181818 100%);
    border: 1px solid #2A2A2A;
    border-radius: 16px;
    padding: 20px 28px;
    margin-bottom: 16px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.20);
}

.hero-title {
    color: #FFFFFF;
    font-size: 29px;
    font-weight: 700;
    letter-spacing: 1px;
    margin: 0;
}

/* NAVIGATION */
.nav-button button {
    width: 100%;
}

/* SECTION */
.section-title {
    color: #F5C84C;
    font-size: 20px;
    font-weight: 700;
    margin: 5px 0 5px 0;
}

.section-caption {
    color: #BDBDBD;
    font-size: 13px;
    margin-bottom: 14px;
}

/* INPUTS */
label,
.stSelectbox label,
.stNumberInput label {
    color: #BDBDBD !important;
    font-weight: 500 !important;
}

div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div {
    background-color: #111111 !important;
    border: 1px solid #444444 !important;
    border-radius: 8px !important;
}

input {
    color: #FFFFFF !important;
}

div[data-baseweb="select"] * {
    color: #FFFFFF !important;
}

/* BUTTONS */
.stButton > button,
.stFormSubmitButton > button {
    background: #C9A227 !important;
    color: #000000 !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    min-height: 40px;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    background: #E0B82F !important;
}

/* METRICS */
div[data-testid="stMetric"] {
    background: #111111;
    border: 1px solid #2A2A2A;
    border-radius: 12px;
    padding: 14px 16px;
    min-height: 95px;
}

div[data-testid="stMetricLabel"] {
    color: #BDBDBD !important;
    font-size: 11px !important;
    font-weight: 600 !important;
}

div[data-testid="stMetricValue"] {
    color: #F5C84C !important;
}

/* FORM */
div[data-testid="stForm"] {
    background: #0D0D0D;
    border: 1px solid #2A2A2A;
    border-radius: 14px;
    padding: 18px;
}

/* DIVIDER */
hr {
    border: none;
    border-top: 1px solid #2A2A2A;
    margin: 18px 0;
}

/* TABLEAU */
.tableau-container {
    background: #000000;
    border: 1px solid #2A2A2A;
    border-radius: 12px;
    padding: 0;
    overflow: hidden;
}

/* HIDE STREAMLIT CHROME */
#MainMenu,
footer {
    visibility: hidden;
}

header[data-testid="stHeader"] {
    background: transparent;
}

/* ALERTS */
div[data-testid="stAlert"] {
    border-radius: 9px;
}

/* SCROLLBAR */
::-webkit-scrollbar {
    width: 7px;
    height: 7px;
}

::-webkit-scrollbar-track {
    background: #000000;
}

::-webkit-scrollbar-thumb {
    background: #444444;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

# ==================================================
# LOAD MODEL FILES
# ==================================================
model = joblib.load("gb_model.pkl")
scaler = joblib.load("scaler.pkl")
test_probabilities = joblib.load("test_probabilities.pkl")
feature_importance = pd.read_pickle("feature_importance.pkl")

# ==================================================
# HEADER
# ==================================================
st.markdown("""
<div class="hero-header">
    <div class="hero-title">EUROPEAN BANK CUSTOMER CHURN ANALYTICS</div>
</div>
""", unsafe_allow_html=True)

# ==================================================
# NAVIGATION
# ==================================================
if "active_page" not in st.session_state:
    st.session_state["active_page"] = "Risk Prediction"

nav1, nav2, nav3, nav4 = st.columns(4)

with nav1:
    if st.button("🎯 Risk Prediction", use_container_width=True):
        st.session_state["active_page"] = "Risk Prediction"

with nav2:
    if st.button("📊 Model Insights", use_container_width=True):
        st.session_state["active_page"] = "Model Insights"

with nav3:
    if st.button("🔄 What-If Simulator", use_container_width=True):
        st.session_state["active_page"] = "What-If Simulator"

with nav4:
    if st.button("📈 Tableau Dashboard", use_container_width=True):
        st.session_state["active_page"] = "Tableau Dashboard"

st.divider()

# ==================================================
# PAGE 1 — RISK PREDICTION
# ==================================================
if st.session_state["active_page"] == "Risk Prediction":

    st.markdown(
        '<div class="section-title">Customer Risk Prediction</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-caption">'
        'Enter customer details to estimate the probability of customer churn.'
        '</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        year = st.number_input("Year", 1900, 2100, 2016)
        credit_score = st.number_input("Credit Score", 300, 900, 650)
        age = st.number_input("Age", 18, 100, 40)
        tenure = st.number_input("Tenure", 0, 20, 5)

    with col2:
        balance = st.number_input(
            "Balance",
            min_value=0.0,
            value=50000.0,
            step=1000.0
        )

        num_products = st.number_input(
            "Number of Products",
            min_value=1,
            max_value=10,
            value=2
        )

        has_cr_card = st.selectbox(
            "Has Credit Card?",
            ["Yes", "No"]
        )

        active_member = st.selectbox(
            "Is Active Member?",
            ["Yes", "No"]
        )

    with col3:
        estimated_salary = st.number_input(
            "Estimated Salary",
            min_value=0.0,
            value=50000.0,
            step=1000.0
        )

        geography = st.selectbox(
            "Geography",
            ["France", "Germany", "Spain"]
        )

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

    has_cr_card_value = 1 if has_cr_card == "Yes" else 0
    active_member_value = 1 if active_member == "Yes" else 0
    geography_germany = 1 if geography == "Germany" else 0
    geography_spain = 1 if geography == "Spain" else 0
    gender_male = 1 if gender == "Male" else 0

    input_data = pd.DataFrame([[
        year,
        credit_score,
        age,
        tenure,
        balance,
        num_products,
        has_cr_card_value,
        active_member_value,
        estimated_salary,
        geography_germany,
        geography_spain,
        gender_male
    ]], columns=[
        "Year",
        "CreditScore",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
        "EstimatedSalary",
        "Geography_Germany",
        "Geography_Spain",
        "Gender_Male"
    ])

    st.write("")

    if st.button("🔍 Predict Churn Risk", use_container_width=True):

        input_scaled = scaler.transform(input_data)
        churn_probability = model.predict_proba(input_scaled)[0][1]

        st.session_state["original_churn_probability"] = churn_probability

        prediction = 1 if churn_probability >= 0.50 else 0

        st.markdown(
            '<div class="section-title">Prediction Result</div>',
            unsafe_allow_html=True
        )

        result1, result2 = st.columns(2)

        with result1:
            st.metric(
                "CHURN PROBABILITY",
                f"{churn_probability * 100:.2f}%"
            )

        with result2:
            if prediction == 1:
                st.error("⚠️ HIGH CHURN RISK")
            else:
                st.success("✅ LOW CHURN RISK")

# ==================================================
# PAGE 2 — MODEL INSIGHTS
# ==================================================
elif st.session_state["active_page"] == "Model Insights":

    st.markdown(
        '<div class="section-title">Model Insights</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-caption">'
        'Explore churn probability patterns and the features contributing most to model predictions.'
        '</div>',
        unsafe_allow_html=True
    )

    chart1, chart2 = st.columns(2, gap="medium")

    with chart1:

        st.markdown(
            '<div class="section-title" style="font-size:16px;">'
            'Churn Probability Distribution'
            '</div>',
            unsafe_allow_html=True
        )

        probability_percent = test_probabilities * 100

        fig = px.histogram(
            x=probability_percent,
            nbins=20,
            labels={
                "x": "Churn Probability (%)",
                "count": "Customers"
            }
        )

        fig.update_traces(
            marker_color="#C9A227",
            marker_line_color="#000000",
            marker_line_width=1
        )

        if "original_churn_probability" in st.session_state:

            current_probability = (
                st.session_state["original_churn_probability"] * 100
            )

            fig.add_vline(
                x=current_probability,
                line_dash="dash",
                line_color="#F5C84C",
                annotation_text=f"Customer: {current_probability:.1f}%",
                annotation_position="top"
            )

        fig.add_vline(
            x=50,
            line_dash="dot",
            line_color="#8E7CC3",
            annotation_text="50% Threshold",
            annotation_position="bottom"
        )

        fig.update_layout(
            paper_bgcolor="#000000",
            plot_bgcolor="#111111",
            font=dict(color="#FFFFFF"),
            xaxis_title="Churn Probability (%)",
            yaxis_title="Customers",
            height=480,
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis=dict(gridcolor="#2A2A2A"),
            yaxis=dict(gridcolor="#2A2A2A")
        )

        st.plotly_chart(fig, use_container_width=True)

    with chart2:

        st.markdown(
            '<div class="section-title" style="font-size:16px;">'
            'Feature Importance'
            '</div>',
            unsafe_allow_html=True
        )

        fig_importance = go.Figure(
            go.Bar(
                x=feature_importance["Importance"],
                y=feature_importance["Feature"],
                orientation="h",
                marker=dict(
                    color="#8E7CC3",
                    line=dict(
                        color="#000000",
                        width=1
                    )
                )
            )
        )

        fig_importance.update_layout(
            paper_bgcolor="#000000",
            plot_bgcolor="#111111",
            font=dict(color="#FFFFFF"),
            xaxis_title="Importance",
            yaxis_title="Feature",
            yaxis=dict(
                autorange="reversed",
                gridcolor="#2A2A2A"
            ),
            xaxis=dict(gridcolor="#2A2A2A"),
            height=480,
            margin=dict(l=20, r=20, t=20, b=20)
        )

        st.plotly_chart(
            fig_importance,
            use_container_width=True
        )

# ==================================================
# PAGE 3 — WHAT-IF SIMULATOR
# ==================================================
elif st.session_state["active_page"] == "What-If Simulator":

    st.markdown(
        '<div class="section-title">What-If Scenario Simulator</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-caption">'
        'Adjust customer characteristics to explore how predicted churn risk changes.'
        '</div>',
        unsafe_allow_html=True
    )

    with st.form("whatif_form"):

        col1, col2, col3 = st.columns(3)

        with col1:
            whatif_year = st.number_input(
                "Year",
                2010,
                2030,
                2016,
                key="whatif_year"
            )

            whatif_balance = st.number_input(
                "Balance",
                min_value=0.0,
                value=50000.0,
                step=1000.0,
                key="whatif_balance"
            )

            whatif_salary = st.number_input(
                "Estimated Salary",
                min_value=0.0,
                value=50000.0,
                step=1000.0,
                key="whatif_salary"
            )

            whatif_credit_score = st.number_input(
                "Credit Score",
                300,
                850,
                650,
                key="whatif_credit_score"
            )

        with col2:
            whatif_products = st.number_input(
                "Number of Products",
                1,
                4,
                2,
                key="whatif_products"
            )

            whatif_geography = st.selectbox(
                "Geography",
                ["France", "Germany", "Spain"],
                key="whatif_geography"
            )

            whatif_age = st.number_input(
                "Age",
                18,
                100,
                40,
                key="whatif_age"
            )

            whatif_card = st.selectbox(
                "Has Credit Card?",
                ["Yes", "No"],
                key="whatif_card"
            )

        with col3:
            whatif_gender = st.selectbox(
                "Gender",
                ["Male", "Female"],
                key="whatif_gender"
            )

            whatif_tenure = st.number_input(
                "Tenure",
                0,
                10,
                5,
                key="whatif_tenure"
            )

            whatif_active = st.selectbox(
                "Is Active Member?",
                ["Yes", "No"],
                key="whatif_active"
            )

        whatif_submit = st.form_submit_button(
            "🔄 Run What-If Scenario",
            use_container_width=True
        )

    if whatif_submit:

        whatif_customer = pd.DataFrame({
            "Year": [whatif_year],
            "Balance": [whatif_balance],
            "EstimatedSalary": [whatif_salary],
            "CreditScore": [whatif_credit_score],
            "NumOfProducts": [whatif_products],
            "Age": [whatif_age],
            "Tenure": [whatif_tenure],
            "HasCrCard": [
                1 if whatif_card == "Yes" else 0
            ],
            "IsActiveMember": [
                1 if whatif_active == "Yes" else 0
            ],
            "Gender_Male": [
                1 if whatif_gender == "Male" else 0
            ],
            "Geography_Germany": [
                1 if whatif_geography == "Germany" else 0
            ],
            "Geography_Spain": [
                1 if whatif_geography == "Spain" else 0
            ]
        })

        whatif_customer = whatif_customer.reindex(
            columns=scaler.feature_names_in_,
            fill_value=0
        )

        whatif_scaled = scaler.transform(
            whatif_customer
        )

        whatif_probability = model.predict_proba(
            whatif_scaled
        )[0][1]

        st.session_state["whatif_probability"] = whatif_probability

    if "whatif_probability" in st.session_state:

        whatif_probability = (
            st.session_state["whatif_probability"]
        )

        if "original_churn_probability" in st.session_state:

            original_probability = (
                st.session_state["original_churn_probability"]
            )

            probability_change = (
                whatif_probability -
                original_probability
            )

            st.markdown(
                '<div class="section-title">Scenario Results</div>',
                unsafe_allow_html=True
            )

            r1, r2, r3 = st.columns(3)

            with r1:
                st.metric(
                    "ORIGINAL",
                    f"{original_probability * 100:.2f}%"
                )

            with r2:
                st.metric(
                    "WHAT-IF",
                    f"{whatif_probability * 100:.2f}%"
                )

            with r3:
                st.metric(
                    "CHANGE",
                    f"{probability_change * 100:+.2f}%"
                )

            comparison_df = pd.DataFrame({
                "Scenario": ["Original", "What-If"],
                "Churn Probability": [
                    original_probability * 100,
                    whatif_probability * 100
                ]
            })

            fig_comparison = px.bar(
                comparison_df,
                x="Scenario",
                y="Churn Probability",
                text="Churn Probability"
            )

            fig_comparison.update_traces(
                marker_color=[
                    "#6B7280",
                    "#C9A227"
                ],
                texttemplate="%{text:.2f}%",
                textposition="outside"
            )

            fig_comparison.update_layout(
                paper_bgcolor="#000000",
                plot_bgcolor="#111111",
                font=dict(color="#FFFFFF"),
                yaxis_title="Churn Probability (%)",
                xaxis_title="Scenario",
                yaxis=dict(
                    range=[0, 100],
                    gridcolor="#2A2A2A"
                ),
                xaxis=dict(gridcolor="#2A2A2A"),
                height=370,
                margin=dict(l=20, r=20, t=20, b=20)
            )

            st.plotly_chart(
                fig_comparison,
                use_container_width=True
            )

        else:
            st.metric(
                "WHAT-IF CHURN PROBABILITY",
                f"{whatif_probability * 100:.2f}%"
            )

            st.info(
                "Run the original customer prediction first "
                "to compare it with the What-If scenario."
            )

# ============================================================
# TABLEAU DASHBOARD - RESPONSIVE EMBED
# ============================================================

import streamlit.components.v1 as components

st.markdown(
    """
    <div style="
        color:#C9A227;
        font-size:22px;
        font-weight:700;
        margin-top:10px;
        margin-bottom:12px;
    ">
        TABLEAU ANALYTICS DASHBOARD
    </div>
    """,
    unsafe_allow_html=True
)

tableau_url = (
    "https://public.tableau.com/views/"
    "EuropeanBankCustomerChurnAnalysis/"
    "EuropeanBankCustomerChurnAnalysis"
    "?:showVizHome=no"
    "&:display_count=n"
    "&:toolbar=no"
)

tableau_html = f"""
<!DOCTYPE html>
<html>
<head>
<style>
    * {{
        box-sizing: border-box;
    }}

    html, body {{
        margin: 0;
        padding: 0;
        width: 100%;
        background: #000000;
        overflow: hidden;
    }}

    #tableau-container {{
        width: 100%;
        overflow: hidden;
        position: relative;
    }}

    #tableau-wrapper {{
        position: relative;
        width: 1604px;
        height: 866px;
        transform-origin: top left;
    }}

    tableau-viz {{
        display: block;
        width: 1604px;
        height: 866px;
    }}
</style>
</head>

<body>

<div id="tableau-container">
    <div id="tableau-wrapper">
        <tableau-viz
            id="tableauViz"
            src="{tableau_url}"
            toolbar="hidden"
            hide-tabs>
        </tableau-viz>
    </div>
</div>

<script type="module"
    src="https://public.tableau.com/javascripts/api/tableau.embedding.3.latest.min.js">
</script>

<script>
function resizeTableau() {{

    const container = document.getElementById("tableau-container");
    const wrapper = document.getElementById("tableau-wrapper");

    if (!container || !wrapper) return;

    const originalWidth = 1604;
    const originalHeight = 866;

    const availableWidth = container.clientWidth;

    let scale = availableWidth / originalWidth;

    // Never enlarge the dashboard beyond its original size
    scale = Math.min(scale, 1);

    wrapper.style.transform = "scale(" + scale + ")";

    // Adjust the wrapper height after scaling
    container.style.height = (originalHeight * scale) + "px";
}}

window.addEventListener("load", resizeTableau);
window.addEventListener("resize", resizeTableau);

const observer = new ResizeObserver(resizeTableau);
observer.observe(document.body);
</script>

</body>
</html>
"""

components.html(
    tableau_html,
    height=700,
    scrolling=False
)
