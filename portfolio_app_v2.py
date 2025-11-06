import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

st.set_page_config(page_title="Steven Rock-Ward Portfolio", layout="wide")


# ----------------- SESSION STATE -----------------
if "df" not in st.session_state:
    try:
        default_df = pd.read_csv("loan.csv")  # must be in the same folder as the app
    except Exception:
        default_df = pd.DataFrame()  # fallback if file not found
    st.session_state.df = default_df.copy()
    st.session_state.df_filtered = default_df.copy()

# ----------------- FUNCTIONS -----------------
def show_professional_summary():
    st.title("Steven Rock-Ward Portfolio")
    st.header("Professional Summary")
    st.write("""
    - 🎓 Education: B.S. in Finance, Utah Valley University  Aug. 2020
    - 🎓 Education: M.S. in Business Analytics, University of Utah  Dec. 2022
    - 💼 Experience: Data Analyst at Zions Bank Corporation Jan. 2023 to March 2025
    - 💼 Experience: Sales Campaign Analyst at SoFi Bank Sept. 2021 to Jan 2023
    - 💼 Experience: Review Team Lead at SoFi Bank May. 2017 to Sept. 2021
    - 🛠 Skills: Python, SQL, R, Machine Learning, Data Visualization, Streamlit, Statistics  
    - 📍 Location: Washington Metropolitan Area
    """)

    # Radar chart
    skills = {"Python": 80, "Machine Learning": 75, "SQL / Data Engineering": 55, "Statistics": 85,
              "Problem Solving": 90, "Data Visualization": 80}
    labels = list(skills.keys())
    values = list(skills.values())
    values += values[:1]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    col1, = st.columns([1])
    with col1:
        fig, ax = plt.subplots(figsize=(3, 3), subplot_kw=dict(polar=True))
        ax.plot(angles, values, 'o-', linewidth=.75, color='green', markersize=2.5)
        ax.fill(angles, values, color='#FF8C00', alpha=0.25)
        ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=6)
        ax.set_ylim(0, 100)
        ax.set_yticks([25, 50, 75, 100])
        ax.set_yticklabels([25, 50, 75, 100], fontsize=6, color='gray')
        ax.xaxis.grid(True, color='gray', linestyle='--', linewidth=0.5)
        ax.yaxis.grid(True, color='gray', linestyle='--', linewidth=0.5)
        label_offset = 5
        for angle, value in zip(angles, values):
            ax.text(angle, value + label_offset, str(value), fontsize=7,
                    horizontalalignment='center', verticalalignment='center')
        ax.set_title("Skills & Proficiency", size=10, y=1.05)
        st.pyplot(fig, use_container_width=False)

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
    "Financial Health": show_financial_health
}

st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", list(pages.keys()))
pages[selection]()
