import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import shap
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from PIL import Image

# ----------------- PAGE CONFIG -----------------
st.set_page_config(page_title="Steven Rock-Ward Portfolio", layout="wide")


# ----------------- SESSION STATE -----------------
if "df" not in st.session_state:
    try:
        default_df = pd.read_csv("loan.csv")  # must be in the same folder as the app
    except Exception:
        default_df = pd.DataFrame()  # fallback if file not found
    st.session_state.df = default_df.copy()
    st.session_state.df_filtered = default_df.copy()

# ----------------- LOAD ML MODEL -----------------
@st.cache_resource
def load_model():
    return joblib.load("C:/Users/esteb/PythonProjects/Portfolio/ml_project/models/loan_model_pipeline_compressed.pkl")

model_pipeline = load_model()

# ----------------- FUNCTIONS -----------------
def show_professional_summary():
    st.title("Steven Rock-Ward Portfolio")
    # GitHub link right below title
    st.markdown("GitHub: [Portfolio Repository](https://github.com/stevenrockward/Portfolio-App)")

    # ----------------- PROFILE + ABOUT ME -----------------
    col1, col2 = st.columns([2, 1])  # About Me wider, image smaller
    with col1:
        st.header("About Me")
        st.write("""
        I'm a data-driven problem solver passionate about turning complex data 
        into clear, actionable insights. My experience spans **financial analytics**, 
        **machine learning**, and **business strategy**, combining quantitative precision 
        with practical impact.

        I enjoy building models that *actually get used* — whether it’s predicting 
        loan defaults, optimizing campaigns, or improving operational efficiency.
        """)

        # ----------------- PROFESSIONAL SUMMARY -----------------
        st.header("Professional Summary")
        st.write("""
        - 🎓 **Education:** M.S. in Business Analytics, University of Utah (Dec 2022)  
        - 🎓 **Education:** B.S. in Finance, Utah Valley University (Aug 2020)  
        - 💼 **Business Analyst** – U.S. Government (Apr 2025 – Present)  
        - 💼 **Data Analyst** – Zions Bank Corporation (Jan 2023 – Mar 2025)  
        - 💼 **Sales Campaign Analyst** – SoFi Bank (Sept 2021 – Jan 2023)  
        - 💼 **Review Team Lead** – SoFi Bank (May 2017 – Sept 2021)  
        - 📍 **Location:** Washington Metropolitan Area  
        """)
    with col2:
        try:
            st.write("**Me and my niece at Volcanoes National Park**")
            image = Image.open("ml_project/images/volcano.jpg")
            # Rotate 90 degrees clockwise
            image = image.rotate(-90, expand=True)
            # Resize image proportionally
            scale_factor = 1.5  
            width, height = image.size
            new_size = (int(width * scale_factor), int(height * scale_factor))
            image = image.resize(new_size)
            
            st.image(image)
        
        except FileNotFoundError:
            st.warning("Profile image not found.")

    
    
    # ----------------- SKILLS & IKIGAI -----------------
    st.subheader("Skills & Professional Purpose")
    col1, col2 = st.columns([1, 1])

    # ----- LEFT COLUMN: Radar Chart -----
    with col1:
        skills = {
            "Python": 80,
            "Machine Learning": 75,
            "SQL / Data Engineering": 65,
            "Statistics": 85,
            "Problem Solving": 90,
            "Data Visualization": 85
        }
        labels = list(skills.keys())
        values = list(skills.values())
        labels.append(labels[0])
        values.append(values[0])

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=labels,
            fill='toself',
            fillcolor='rgba(249, 168, 38, 0.3)',
            line=dict(color='green'),
            name='Skills'
        ))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100])
            ),
            showlegend=False,
            autosize=True,
            margin=dict(l=30, r=30, t=30, b=30)
        )
        st.plotly_chart(fig, use_container_width=True)

    # ----- RIGHT COLUMN: Ikigai -----
    with col2:
        # Explanation goes above the chart
        st.write("""
        **Ikigai** (生き甲斐) represents the intersection of what you love, what you’re good at, 
        what the world needs, and what you can be paid for. Here's where my professional purpose aligns:
        """)
        
        # Ikigai chart
        st.markdown("""
        <style>
        .ikigai-container {
            position: relative;
            width: 100%;
            padding-top: 100%;
        }
        .circle {
            position: absolute;
            width: 40%;
            height: 40%;
            border-radius: 50%;
            font-size: 0.85rem;
            text-align: center;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            padding: 4px;
        }
        .love { top: 0%; left: 30%; background-color: rgba(249, 168, 38, 0.7); color: white; }
        .good { top: 30%; left: 0%; background-color: rgba(76, 175, 80, 0.7); color: white; }
        .need { top: 30%; right: 0%; background-color: rgba(129, 199, 132, 0.7); color: white; }
        .paid { bottom: 0%; left: 30%; background-color: rgba(255, 204, 128, 0.7); color: white; }

        .center {
            position: absolute;
            top: 35%;
            left: 35%;
            width: 30%;
            height: 30%;
            background-color: rgba(255,255,255,0.95);
            color: black;
            border-radius: 50%;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            text-align: center;
        }
        .center h3 { font-weight: bold; margin: 0 0 3px 0; }
        .circle h3 { font-weight: bold; margin: 0 0 2px 0; font-size: 0.9rem; }
        </style>

        <div class="ikigai-container">
        <div class="circle love">
            <h3>What I Love</h3>
            Finance<br>
            Exploration<br>
            Competition<br>
            Intellectual Curiosity<br>
            Socializing
        </div>
        <div class="circle good">
            <h3>What I'm Good At</h3>
            Analytics<br>
            Strategy<br>
            Problem Solving<br>
            Mastery Through Play
        </div>
        <div class="circle need">
            <h3>What the World Needs</h3>
            Stability<br>
            Free Exchange<br>
            Financial Health
        </div>
        <div class="circle paid">
            <h3>What I Can Be Paid For</h3>
            Finance<br>
            Analytics<br>
            Strategy<br>
            Policy
        </div>
        <div class="center">
            <h3>Ikigai</h3>
            Strategy<br>Exploration<br>Stability
        </div>
        </div>
        """, unsafe_allow_html=True)

# ----------------- DATA EXPLORER -----------------
def show_data_explorer():
    st.header("Data Explorer")
    st.write("""
    Upload a CSV file to explore the data, handle missing values, filter columns, and visualize relationships.
    You can also train simple machine learning models in the ML Playground tab.
    A Default dataset (loan.csv) is pre-loaded for demonstration.
    """)
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df.copy()
        st.session_state.df_filtered = df.copy()
    
    if st.session_state.df is None:
        st.info("Please upload a CSV file to begin.")
        return

    df = st.session_state.df_filtered

    # Overview
    st.subheader("Overview")
    st.write("Shape:", df.shape)
    st.dataframe(df.head())

    # Missing Value Handling
    st.subheader("Handle Missing Values")
    na_option = st.selectbox(
        "How do you want to handle missing values?",
        ["Do nothing", "Drop rows", "Fill with mean", "Fill with median", "Fill with mode"]
    )

    if st.button("Apply NA Handling"):
        if na_option == "Drop rows":
            df = df.dropna()
        elif na_option == "Fill with mean":
            for col in df.select_dtypes(include='number').columns:
                df[col] = df[col].fillna(df[col].mean())
        elif na_option == "Fill with median":
            for col in df.select_dtypes(include='number').columns:
                df[col] = df[col].fillna(df[col].median())
        elif na_option == "Fill with mode":
            for col in df.columns:
                df[col] = df[col].fillna(df[col].mode()[0])
        st.session_state.df_filtered = df
        st.success("Missing values handled!")

    st.subheader("Missing Values (after handling)")
    st.dataframe(df.isnull().sum())

    # Column selection / filtering
    selected_cols = st.multiselect("Columns to view", list(df.columns), default=list(df.columns))
    filtered_df = df[selected_cols].copy()

    for col in selected_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            min_val, max_val = float(df[col].min()), float(df[col].max())
            rng = st.slider(f"Range for {col}", min_val, max_val, (min_val, max_val))
            filtered_df = filtered_df[(filtered_df[col] >= rng[0]) & (filtered_df[col] <= rng[1])]
        else:
            filter_text = st.text_input(f"Filter {col} by keyword (comma separated)")
            if filter_text:
                keywords = [k.strip() for k in filter_text.split(",")]
                filtered_df = filtered_df[filtered_df[col].astype(str).str.contains("|".join(keywords), na=False)]

    st.session_state.df_filtered = filtered_df

    st.subheader("Filtered Data")
    st.dataframe(filtered_df)

    # Correlation Matrix
    numeric_cols = filtered_df.select_dtypes(include="number").columns.tolist()
    if numeric_cols:
        st.subheader("Correlation Matrix")
        fig, ax = plt.subplots(figsize=(3, 2))
        sns.heatmap(
            filtered_df[numeric_cols].corr(),
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            ax=ax,
            annot_kws={"size": 6}  # annotation font size
        )
        ax.tick_params(axis='x', labelrotation=45, labelsize=6)
        ax.tick_params(axis='y', labelrotation=0, labelsize=6)
        st.pyplot(fig, use_container_width=False)

    # Visualizations
    cat_cols = filtered_df.select_dtypes(include="object").columns.tolist()
    st.subheader("Visualizations")
    chart_type = st.selectbox("Choose plot type", ["Histogram", "Boxplot", "Scatterplot", "Bar Chart", "Boxplot by Category", "Count Plot"])
    if chart_type == "Histogram" and numeric_cols:
        col = st.selectbox("Column", numeric_cols)
        fig, ax = plt.subplots(figsize=(3, 2))
        sns.histplot(filtered_df[col], kde=True, ax=ax)
        ax.set_xlabel(col, fontsize=6)
        ax.set_ylabel("Count", fontsize=6)
        ax.tick_params(axis='both', labelsize=6)
        st.pyplot(fig, use_container_width=False)
    elif chart_type == "Boxplot" and numeric_cols:
        col = st.selectbox("Column", numeric_cols)
        fig, ax = plt.subplots(figsize=(3, 2))
        sns.boxplot(x=filtered_df[col], ax=ax)
        ax.set_xlabel(col, fontsize=6)
        ax.tick_params(axis='both', labelsize=6)
        st.pyplot(fig, use_container_width=False)
    elif chart_type == "Scatterplot" and len(numeric_cols) >= 2:
        x_col = st.selectbox("X-axis", numeric_cols)
        y_col = st.selectbox("Y-axis", numeric_cols)
        fig, ax = plt.subplots(figsize=(3, 2))
        sns.scatterplot(x=filtered_df[x_col], y=filtered_df[y_col], ax=ax)
        ax.set_xlabel(x_col, fontsize=6)
        ax.set_ylabel(y_col, fontsize=6)
        ax.tick_params(axis='both', labelsize=6)
        st.pyplot(fig, use_container_width=False)
    elif chart_type == "Bar Chart" and cat_cols and numeric_cols:
        cat_col = st.selectbox("Categorical Column", cat_cols)
        num_col = st.selectbox("Numeric Column", numeric_cols)
        agg_func = st.selectbox("Aggregate", ["mean", "sum"])
        grouped = filtered_df.groupby(cat_col)[num_col].agg(agg_func).reset_index()
        grouped = grouped.sort_values(by=num_col, ascending=False).head(20)
        fig, ax = plt.subplots(figsize=(3.5, 2.5))
        sns.barplot(x=cat_col, y=num_col, data=grouped, ax=ax)
        ax.set_xlabel(cat_col, fontsize=6)
        ax.set_ylabel(num_col, fontsize=6)
        plt.xticks(rotation=45, fontsize=6)
        plt.yticks(fontsize=6)
        st.pyplot(fig, use_container_width=False)
    elif chart_type == "Boxplot by Category" and cat_cols and numeric_cols:
        cat_col = st.selectbox("Categorical Column", cat_cols)
        num_col = st.selectbox("Numeric Column", numeric_cols)
        fig, ax = plt.subplots(figsize=(3, 2))
        sns.boxplot(x=cat_col, y=num_col, data=filtered_df, ax=ax)
        plt.xticks(rotation=45, fontsize=6)
        plt.yticks(fontsize=6)
        ax.set_xlabel(cat_col, fontsize=6)
        ax.set_ylabel(num_col, fontsize=6)
        st.pyplot(fig, use_container_width=False)
    elif chart_type == "Count Plot" and cat_cols:
        cat_col = st.selectbox("Categorical Column", cat_cols)
        fig, ax = plt.subplots(figsize=(3.5, 2.5))
        sns.countplot(x=cat_col, data=filtered_df, ax=ax)
        ax.set_xlabel(cat_col, fontsize=6)
        ax.set_ylabel("Count", fontsize=6)
        plt.xticks(rotation=45, fontsize=6)
        plt.yticks(fontsize=6)
        st.pyplot(fig, use_container_width=False)

# ----------------- ML PLAYGROUND -----------------
def show_ml_playground():
    st.header("Machine Learning Playground")
    st.caption(
        "Tip: For the default Lending Club dataset, use **'loan_status'** "
        "as the target column (1 = default, 0 = paid)."
    )
    if st.session_state.df_filtered is None:
        st.info("Please upload and process a CSV in Data Explorer first.")
        return

    df_ml = st.session_state.df_filtered.copy()
    for col in df_ml.select_dtypes(include='number').columns:
        df_ml[col] = df_ml[col].fillna(df_ml[col].median())
    for col in df_ml.select_dtypes(include='object').columns:
        df_ml[col] = df_ml[col].fillna(df_ml[col].mode()[0])

    target = st.selectbox("Target", list(df_ml.columns))
    features = [col for col in df_ml.columns if col != target]
    X = pd.get_dummies(df_ml[features], drop_first=True)
    y = df_ml[target]

    test_size = st.slider("Test size (%)", 10, 50, 20) / 100
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    clf_name = st.selectbox("Classifier", ["Logistic Regression", "Random Forest", "KNN"])
    if clf_name == "Logistic Regression":
        clf = LogisticRegression(max_iter=1000)
    elif clf_name == "Random Forest":
        n_estimators = st.slider("n_estimators", 50, 300, 100)
        clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    else:
        k = st.slider("k (neighbors)", 3, 15, 5)
        clf = KNeighborsClassifier(n_neighbors=k)

    if st.button("Train Model"):
        if clf_name in ["Logistic Regression", "KNN"]:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        st.subheader("Metrics")
        st.text(classification_report(y_test, y_pred))
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(3, 2))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, annot_kws={"size": 6})
        ax.set_xlabel("Predicted", fontsize=6)
        ax.set_ylabel("Actual", fontsize=6)
        ax.tick_params(axis='both', labelsize=6)
        st.pyplot(fig, use_container_width=False)

# ----------------- Loan Default Prediction -----------------
def show_loan_default_prediction():
    st.title("Loan Default Prediction")

    # --- Model Info ---
    model_version = "v1.3"
    last_trained = "2025-11-06"
    hyperparameters = {
        "n_estimators": 200,
        "max_depth": 6,
        "random_state": 42
    }
    st.markdown(f"**Model Version:** {model_version}")
    st.markdown(f"**Last Trained:** {last_trained}")
    st.markdown(f"**Hyperparameters:** {hyperparameters}")

    st.write(
        "This section uses a trained Random Forest model to predict "
        "whether a loan is likely to default based on your input data. "
        "The model was trained on Lending Club data from 2007 to 2018. "
        "You can view the model development notebook code on GitHub: "
        "[Portfolio Repository](https://github.com/stevenrockward/Portfolio-App/blob/main/ml_project/notebooks/ML%20Portfolio%20Project.ipynb)"
    )

    st.write("Input information to determine if a loan will default or be paid back in full.")

    # --- Numeric Inputs ---
    loan_amnt = st.number_input("Loan Amount", min_value=0, value=5000, step=100)
    last_fico_avg = st.number_input("Last FICO Avg", min_value=0, value=700, step=1)
    annual_inc = st.number_input("Annual Income", min_value=0, value=50000, step=1000)
    int_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=100.0, value=10.0, step=0.1)

    # --- Categorical Inputs ---
    term = st.selectbox(
        "Loan Term",
        [" 36 months", " 60 months"],
        help="The duration of the loan."
    )
    emp_length = st.selectbox(
        "Employment Length",
        ["0–3 years", "4–7 years", "8–9 years", "10+ years"],
        help="Length of time the applicant has been employed."
    )
    last_credit_pull_d = st.selectbox(
        "Last Credit Pull Period",
        ["2007–2010", "2011–2013", "2014–2016", "2017–2018", "2019"],
        help="The most recent period when the applicant's credit was pulled."
    )
    earliest_cr_line = st.selectbox(
        "Earliest Credit Line",
        ["1930–1970", "1971–1989", "1990–1999", "2000–2009", "2010–2015"],
        help="The period when the applicant's earliest credit line was opened."
    )
    purpose = st.selectbox(
        "Loan Purpose",
        ["debt_consolidation", "business", "home", "personal", "other", "auto"],
        help="Reason for the loan request."
    )
    issue_d = st.selectbox(
        "Issue Date",
        ["2007–2010", "2011–2013", "2014–2016", "2017–2018"],
        help="The period when the loan was issued."
    )
    verification_status = st.selectbox(
        "Verification Status",
        ["Not Verified", "Verified"],
        help="Whether the applicant's income was verified by the lender."
    )
    home_ownership = st.selectbox(
        "Home Ownership",
        ["OWN_HOME", "RENT", "OTHER"],
        help="Home ownership status of the applicant."
    )
    application_type = st.selectbox(
        "Application Type",
        ["Individual", "Joint App"],
        help="Whether the loan application is individual or joint."
    )
    initial_list_status = st.selectbox(
        "Initial List Status",
        ["w", "f"],
        help="Initial listing status of the loan ('w' = whole, 'f' = fractioned)."
    )

    # --- Build input DataFrame ---
    input_df = pd.DataFrame([{
        "loan_amnt": loan_amnt,
        "last_fico_avg": last_fico_avg,
        "annual_inc": annual_inc,
        "int_rate": int_rate,
        "term": term,
        "emp_length": emp_length,
        "last_credit_pull_d": last_credit_pull_d,
        "earliest_cr_line": earliest_cr_line,
        "purpose": purpose,
        "issue_d": issue_d,
        "verification_status": verification_status,
        "home_ownership": home_ownership,
        "application_type": application_type,
        "initial_list_status": initial_list_status
    }])

     # --- Prediction Button ---
    if st.button("Predict Loan Default"):
        try:
            prediction = model_pipeline.predict(input_df)[0]
            proba = model_pipeline.predict_proba(input_df)[0, 1]

            st.subheader("Prediction Results")
            st.write(f"Prediction: **{'Default' if prediction else 'Approved'}**")
            st.write(f"Probability of Default: **{proba:.2%}**")

            # Risk category
            if proba < 0.2:
                risk = "Low Risk"
            elif proba < 0.5:
                risk = "Moderate Risk"
            elif proba < 0.8:
                risk = "High Risk"
            else:
                risk = "Very High Risk"
            st.write(f"Calibrated Risk Category: **{risk}**")

        except Exception as e:
            st.error(f"Error making prediction: {e}")

    

# ----------------- FINANCIAL HEALTH -----------------
def show_financial_health():
    st.title("Your Financial Ratios")
    st.write(
        "These ratios are based on standard financial planning guidelines used by advisors to assess financial health. "
        "All inputs should be entered as **monthly values**."
    )

    # --- Inputs ---
    st.header("Enter Your Monthly Financials")
    gross_income = st.number_input("Gross Income (before tax)", min_value=0.0, step=100.0)
    retirement_pct = st.number_input("Retirement Contribution Rate (%)", min_value=0.0, max_value=50.0, step=1.0)
    take_home_pay = st.number_input("Take-Home Pay (after tax and retirement)", min_value=0.0, step=100.0)
    housing_costs = st.number_input("Housing Costs (mortgage + insurance + taxes)", min_value=0.0, step=50.0)
    auto_costs = st.number_input("Auto Costs (loan + insurance)", min_value=0.0, step=50.0)
    other_expenses = st.number_input("Other Living Expenses", min_value=0.0, step=50.0)
    debt_payments = st.number_input("Debt Payments (non-mortgage + non-auto)", min_value=0.0, step=50.0)
    cash_on_hand = st.number_input("Cash on Hand (checking + savings)", min_value=0.0, step=100.0)
    current_retirement = st.number_input("Current Retirement Accounts ($)", min_value=0.0, step=100.0)
    current_investments = st.number_input("Other Investment Accounts ($)", min_value=0.0, step=100.0)

    # --- Calculations ---
    retirement_contrib = gross_income * (retirement_pct / 100)
    total_expenses = housing_costs + auto_costs + other_expenses + debt_payments + retirement_contrib
    cash_surplus = take_home_pay - (housing_costs + auto_costs + other_expenses + debt_payments)
    savings_rate = (cash_surplus / take_home_pay * 100) if take_home_pay else 0
    housing_ratio = (housing_costs / gross_income * 100) if gross_income else 0
    auto_ratio = (auto_costs / gross_income * 100) if gross_income else 0
    monthly_expenses_excl_savings = housing_costs + auto_costs + other_expenses + debt_payments
    emergency_fund = ((cash_on_hand + current_retirement + current_investments) / monthly_expenses_excl_savings) if monthly_expenses_excl_savings else 0
    dti = (debt_payments / gross_income * 100) if gross_income else 0
    debt_disposable = (debt_payments / take_home_pay * 100) if take_home_pay else 0

    # --- Status helper ---
    def status_text(value, ratio_type):
        if ratio_type == "cash":
            return "🟢 Healthy" if value >= 0 else "🔴 Unhealthy"
        elif ratio_type == "housing":
            if value <= 28:
                return "🟢 Healthy"
            elif 28 < value <= 35:
                return "🟧 Warning"
            else:
                return "🔴 Unhealthy"
        elif ratio_type == "emergency":
            if value < 3:
                return "🔴 Unhealthy"
            elif 3 <= value <= 6:
                return "🟧 Warning"
            else:  # value > 6
                return "🟢 Healthy"
        elif ratio_type == "dti":
            return "🟢 Healthy" if value <= 36 else "🔴 Unhealthy"
        elif ratio_type == "debt_disposable":
            return "🟢 Healthy" if value <= 14 else "🔴 Unhealthy"
        elif ratio_type == "savings":
            return "🟢 Healthy" if value >= 10 else "🔴 Unhealthy"
        elif ratio_type == "retirement":
            if 10 <= value <= 15:
                return "🟢 Healthy"
            else:
                return "🟧 Check Contribution"
        return ""

    # --- KPI Cards ---
    st.header("Financial Health KPIs")

    def render_kpi(title, value, status, goal):
        st.markdown(f"""
            <div style="
                border: 2px solid #ccc; 
                border-radius: 10px; 
                padding: 15px; 
                margin-bottom: 10px;
            ">
                <div style="font-weight:bold; font-size:18px; margin-bottom:5px;">{title}</div>
                <div style="font-size:16px; margin-bottom:5px;">{value}</div>
                <div style="margin-bottom:5px;">Goal: {goal}</div>
                <div style="font-weight:bold;">{status}</div>
            </div>
        """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        render_kpi("Cash Surplus (Deficit)", f"${cash_surplus:,.2f}", status_text(cash_surplus, "cash"), "≥ $0")
        render_kpi("Basic Housing Ratio", f"{housing_ratio:.2f}%", status_text(housing_ratio, "housing"), "≤ 28%")
    with col2:
        render_kpi("Auto Ratio", f"{auto_ratio:.2f}%", status_text(auto_ratio, "auto"), "≤ 20%")
        render_kpi("Emergency Fund", f"{emergency_fund:.2f} months", status_text(emergency_fund, "emergency"), "≥ 6 months")
        render_kpi("Savings Rate", f"{savings_rate:.2f}%", status_text(savings_rate, "savings"), "≥ 10%")
    with col3:
        render_kpi("Debt-to-Income", f"{dti:.2f}%", status_text(dti, "dti"), "≤ 36%")
        render_kpi("Debt-to-Disposable", f"{debt_disposable:.2f}%", status_text(debt_disposable, "debt_disposable"), "≤ 14%")
        render_kpi("Retirement Contribution %", f"{retirement_pct:.2f}%", status_text(retirement_pct, "retirement"), "10–15%")


# ----------------- MAIN -----------------
pages = {
    "Professional Summary": show_professional_summary,
    "Data Explorer": show_data_explorer,
    "ML Playground": show_ml_playground,
    "Loan Default Prediction": show_loan_default_prediction,
    "Financial Health": show_financial_health
    
}

st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", list(pages.keys()))
pages[selection]()

import sklearn
print(sklearn.__version__)
