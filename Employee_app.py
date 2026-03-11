import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Employee Attrition App", layout="wide", page_icon="📊")

# LOAD PKL FILES OR THE PRE-TRAINED ASSETS
@st.cache_resource
def load_assets():
    model = joblib.load('xgb_attrition_model.pkl')
    scaler = joblib.load('fitted_scaler.pkl')
    cols = joblib.load('training_columns.pkl')
    return model, scaler, cols

@st.cache_data
def load_data():
    df = pd.read_csv('employee_performance_workload_attrition1.csv')
    return df

xgb_model, scaler, feature_cols = load_assets()
df = load_data()

st.title("Employee Attrition Prediction")
st.markdown("This web application demonstrates how new employee data is processed and evaluated by our trained XGBoost model.")

# CREATE TABS
tab1, tab2 = st.tabs(["Analytics Dashboard", "Prediction Tool"])

# TAB 1: DASHBOARD
with tab1:
    st.header("Historical Attrition Overview")
    
    #KPI METRICS
    total_employees = len(df)
    attrition_rate = (df['attrition'] == 'Yes').mean() * 100
    avg_hours = df['avg_weekly_hours'].mean()
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Employees", f"{total_employees:,}")
    col2.metric("Historical Attrition Rate", f"{attrition_rate:.1f}%")
    col3.metric("Average Weekly Hours", f"{avg_hours:.1f} hrs")
    
    st.divider()
    
    # VISUALIZATIONS
    st.subheader("Key Drivers of Attrition")
    
    chart_col1, chart_col2 = st.columns(2)
    
    # CHART 1: JOB ATTRITION VS ATTRITION
    with chart_col1:
        st.write("**Attrition by Job Satisfaction**")
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        sns.countplot(data=df, x='job_satisfaction', hue='attrition', palette='viridis', ax=ax1)
        ax1.set_xlabel("Job Satisfaction (1-5)")
        ax1.set_ylabel("Number of Employees")
        st.pyplot(fig1)
    # CHART 2: WEEKLY HOURS VS ATTRITION  
    with chart_col2:
        st.write("**Average Weekly Hours by Attrition**")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.boxplot(data=df, x='attrition', y='avg_weekly_hours', palette='Set1', ax=ax2)
        ax2.set_xlabel("Left the Company?")
        ax2.set_ylabel("Weekly Hours")
        st.pyplot(fig2)

# TAB 2: PREDICTION TOOL
with tab2:
    st.header("1. Enter Employee Data")
    st.markdown("Use this tool to predict if an employee is at risk of leaving.")
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            department = st.number_input("Department ID (0-5)", min_value=0, max_value=5, value=1)
            role_level = st.number_input("Role Level ID (0-2)", min_value=0, max_value=2, value=1)
            salary = st.number_input("Monthly Salary ($)", value=45000)
            hours = st.slider("Avg Weekly Hours", 30, 80, 50)

        with col2:
            projects = st.number_input("Projects Handled", min_value=1, max_value=10, value=4)
            rating = st.slider("Performance Rating (1-5)", 1, 5, 3)
            absences = st.number_input("Absence Days", min_value=0, max_value=30, value=8)
            satisfaction = st.slider("Job Satisfaction (1-5)", 1, 5, 2)
            
        submitted = st.form_submit_button("Generate Prediction")

    if submitted:
        #CAPTURE RAW INPUT DATA INTO DATAFRAME
        new_employee = pd.DataFrame([{
            "department": department, "role_level": role_level, "monthly_salary": salary,
            "avg_weekly_hours": hours, "projects_handled": projects, 
            "performance_rating": rating, "absences_days": absences, "job_satisfaction": satisfaction
        }])

        # APPLY FEATURE ENGINEERING
        new_employee['projects_per_hour'] = new_employee['projects_handled'] / new_employee['avg_weekly_hours']
        new_employee["absence_rate"] = new_employee["absences_days"] / 365
        new_employee["performance_efficiency"] = new_employee["performance_rating"] / new_employee["avg_weekly_hours"]
        new_employee["stress_index"] = new_employee["avg_weekly_hours"] * new_employee["absences_days"]
        new_employee["satisfaction_score"] = new_employee["job_satisfaction"] * new_employee["performance_rating"]

        # ALIGN COLUMNS AND SCALE
        new_employee = pd.get_dummies(new_employee) #USE PD.GET_DUMMIES TO MATCH THE TRAINING DATA FORMAT
        for col in feature_cols:
            if col not in new_employee.columns:
                new_employee[col] = 0
                
        new_employee = new_employee[feature_cols]
        new_employee_scaled = scaler.transform(new_employee)

        # CREATE THE PREDICTION OUTPUT
        st.divider()
        st.header("2. Final Prediction Output")
        
        prediction = xgb_model.predict(new_employee_scaled)[0]
        probability = xgb_model.predict_proba(new_employee_scaled)[0][1]

        # INTERPRET THE RESULT
        if prediction == 1:
            st.error(f"Prediction: **YES (High Risk of Leaving)**")
        else:
            st.success(f"Prediction: **NO (Likely to Stay)**")
            
        st.metric(label="Calculated Probability of Attrition", value=f"{probability * 100:.2f}%")