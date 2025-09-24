"""
Gallstone Risk Prediction with Fixed AI Integration
Author: Mayur Wade
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import warnings
import os
import time
warnings.filterwarnings('ignore')

# importing Google Generative AI
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    st.error("Google Generative AI not installed. Run: pip install google generativeai")

# Page configuration
st.set_page_config(
    page_title="Gallstone Risk Prediction with AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS with animations
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Keyframe animations */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes slideInLeft {
        from {
            opacity: 0;
            transform: translateX(-50px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    @keyframes pulse {
        0% {
            transform: scale(1);
        }
        50% {
            transform: scale(1.05);
        }
        100% {
            transform: scale(1);
        }
    }
    
    @keyframes glow {
        0%, 100% {
            box-shadow: 0 0 5px rgba(102, 126, 234, 0.5);
        }
        50% {
            box-shadow: 0 0 20px rgba(102, 126, 234, 0.8);
        }
    }
    
    .main-header {
        text-align: center;
        color: white;
        padding: 30px 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        margin-bottom: 30px;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        animation: fadeInUp 1s ease-out;
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        margin: 10px 0 0 0;
        font-size: 1.2rem;
        font-weight: 300;
        opacity: 0.9;
    }
    
    .risk-high {
        background: linear-gradient(135deg, #ff4444 0%, #cc0000 100%);
        color: white;
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        box-shadow: 0 8px 25px rgba(255, 68, 68, 0.4);
        border: 2px solid rgba(255, 255, 255, 0.1);
        animation: fadeInUp 0.8s ease-out, pulse 2s infinite;
    }
    
    .risk-medium {
        background: linear-gradient(135deg, #ffaa00 0%, #ff8800 100%);
        color: white;
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        box-shadow: 0 8px 25px rgba(255, 170, 0, 0.4);
        border: 2px solid rgba(255, 255, 255, 0.1);
        animation: fadeInUp 0.8s ease-out, glow 3s infinite;
    }
    
    .risk-low {
        background: linear-gradient(135deg, #00C851 0%, #00a041 100%);
        color: white;
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        box-shadow: 0 8px 25px rgba(0, 200, 81, 0.4);
        border: 2px solid rgba(255, 255, 255, 0.1);
        animation: fadeInUp 0.8s ease-out;
    }
    
    .ai-insight-card {
        background: linear-gradient(135deg, #f8f9ff 0%, #e8eeff 100%);
        border: 2px solid #667eea;
        border-radius: 15px;
        padding: 25px;
        margin: 20px 0;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.1);
        animation: slideInLeft 1s ease-out;
        color: #2c3e50 !important;
    }
    
    .ai-insight-header {
        color: #667eea !important;
        font-size: 1.4rem;
        font-weight: 600;
        margin-bottom: 15px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    .ai-insight-content {
        color: #2c3e50 !important;
        line-height: 1.6;
        font-size: 16px;
    }
    
    .ai-insight-content h4 {
        color: #667eea !important;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    
    .ai-insight-content ul {
        color: #2c3e50 !important;
        margin-left: 20px;
    }
    
    .ai-insight-content li {
        color: #2c3e50 !important;
        margin-bottom: 5px;
    }
    
    .feature-card {
        background: white;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        animation: slideInLeft 0.6s ease-out;
        color: #2c3e50 !important;
    }
    
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border-top: 4px solid #667eea;
        animation: fadeInUp 0.8s ease-out;
        color: #2c3e50 !important;
    }
    
    .metric-card h4 {
        color: #667eea !important;
        margin-bottom: 15px;
    }
    
    .factor-item {
        margin: 10px 0;
        padding: 15px;
        background: linear-gradient(135deg, #f8f9ff 0%, #ffffff 100%);
        border-radius: 8px;
        border-left: 4px solid #667eea;
        animation: fadeInUp 0.6s ease-out;
        color: #2c3e50 !important;
    }
    
    .factor-item strong {
        color: #2c3e50 !important;
        font-weight: 600;
    }
    
    .factor-weight {
        color: #667eea !important;
        font-weight: 500;
    }
    
    .results-header {
        animation: fadeInUp 1s ease-out;
        color: #2c3e50 !important;
    }
    
    .patient-summary-text {
        color: #2c3e50 !important;
        font-size: 16px;
        line-height: 1.5;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 15px 30px;
        font-weight: 600;
        font-size: 16px;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    .sidebar .stSelectbox > div > div {
        background-color: #f8f9ff;
    }
    
    .loading-spinner {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 20px;
        animation: pulse 1.5s infinite;
    }
    
    .status-success {
        color: #00C851;
        font-weight: 600;
    }
    
    .status-warning {
        color: #ffaa00;
        font-weight: 600;
    }
    
    .status-error {
        color: #ff4444;
        font-weight: 600;
    }
    
    /* Fix for white text visibility */
    .stMarkdown div {
        color: #2c3e50;
    }
    
    /* Specific fixes for metric values */
    .metric-card .stMetric {
        color: #2c3e50 !important;
    }
    
    .metric-card .stMetric > div {
        color: #2c3e50 !important;
    }
</style>
""", unsafe_allow_html=True)

# AI Integration Class
class SimplifiedAIInsights:
    """Simplified AI insights generator with better error handling"""
    
    def __init__(self, api_key):
        if not GENAI_AVAILABLE:
            raise ImportError("Google Generative AI not available")
        
        genai.configure(api_key=api_key)
    
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        
        self.generation_config = {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 1024,
        }
        
        self.safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]
    
    def generate_clinical_analysis(self, patient_data, risk_level, risk_percentage, top_features):
        """Generate clinical analysis"""
        
        # Format top features
        features_text = "\n".join([f"- {feature}: {importance:.3f}" for feature, importance in top_features[:5]])
        
        prompt = f"""
As a clinical AI assistant, provide a comprehensive analysis for a gallstone risk assessment:

PATIENT PROFILE:
- Age: {patient_data['Age']} years  
- Gender: {'Male' if patient_data['Gender'] == 1 else 'Female'}
- BMI: {patient_data['BMI']:.1f} kg/m²
- Risk Level: {risk_level} ({risk_percentage:.1f}%)

MEDICAL HISTORY:
- Diabetes: {'Yes' if patient_data['Diabetes'] else 'No'}
- Hyperlipidemia: {'Yes' if patient_data['Hyperlipidemia'] else 'No'}
- Hypothyroidism: {'Yes' if patient_data['Hypothyroidism'] else 'No'}
- CAD: {'Yes' if patient_data['CAD'] else 'No'}

KEY CONTRIBUTING FACTORS:
{features_text}

Please provide:

**Risk Assessment Summary**
Brief interpretation of the risk level and what it means for this patient.

**Clinical Significance**
What these findings mean from a clinical perspective and potential implications.

**Recommendations**
Specific next steps based on risk level including diagnostic workup if needed.

**Patient Counseling Points**
Key points to discuss with patient about their risk and management.

Keep the response concise, professional, and actionable. Focus on practical clinical guidance.
"""
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings
            )
            
            if response.text:
                return response.text
            else:
                return "AI analysis temporarily unavailable. Please try again."
                
        except Exception as e:
            return f"Error generating AI analysis: {str(e)}"
    
    def generate_patient_education(self, risk_level, patient_data):
        """Generate patient education content"""
        
        prompt = f"""
Create patient education content for a {patient_data['Age']}-year-old {'male' if patient_data['Gender'] == 1 else 'female'} 
with {risk_level.lower()} risk for gallstones.

Include:

**What are gallstones?**
Simple explanation suitable for patients.

**Your risk level**
What this specific risk level means for them personally.

**Symptoms to watch for**
Key warning signs they should monitor.

**Lifestyle recommendations**
Specific actionable steps they can take.

**When to contact healthcare provider**
Clear guidance on when to seek medical attention.

Write in simple, reassuring language that patients can easily understand.
Keep it under 300 words and focus on actionable advice.
"""
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings
            )
            
            if response.text:
                return response.text
            else:
                return "Patient education content temporarily unavailable."
                
        except Exception as e:
            return f"Error generating patient education: {str(e)}"

@st.cache_resource
def initialize_ai_system():
    """Initialize AI system with API key"""
    api_key = None
    
    # Try different sources for API key
    try:
        api_key = st.secrets.get("GEMINI_API_KEY")
    except:
        pass
    
    if not api_key:
        api_key = os.getenv('GEMINI_API_KEY')
    
    # Check session state for API key
    if not api_key and 'gemini_api_key' in st.session_state:
        api_key = st.session_state.gemini_api_key
    
    if api_key and GENAI_AVAILABLE:
        try:
            ai_system = SimplifiedAIInsights(api_key)
            return ai_system
        except Exception as e:
            st.error(f"Failed to initialize AI system: {str(e)}")
            return None
    
    return None

@st.cache_resource
def create_demo_model():
    """Create enhanced demo model"""
    feature_names = [
        'Age', 'Gender', 'Height', 'Weight', 'BMI',
        'CAD', 'Hypothyroidism', 'Hyperlipidemia', 'Diabetes',
        'Total_Cholesterol', 'LDL', 'HDL', 'Triglyceride',
        'AST', 'ALT', 'ALP', 'Creatinine', 'GFR', 'CRP',
        'Hemoglobin', 'Vitamin_D', 'FM', 'FFM', 'TBW', 'ECW', 'ICW', 'BMR'
    ]
    
    np.random.seed(42)
    n_samples = 1000
    
    # Generate realistic data
    data = {}
    data['Age'] = np.random.uniform(25, 80, n_samples)
    data['Gender'] = np.random.binomial(1, 0.5, n_samples)
    data['Height'] = np.random.uniform(150, 190, n_samples)
    data['Weight'] = np.random.uniform(50, 120, n_samples)
    data['BMI'] = data['Weight'] / ((data['Height']/100) ** 2)
    
    # Medical conditions
    data['CAD'] = np.random.binomial(1, 0.1, n_samples)
    data['Hypothyroidism'] = np.random.binomial(1, 0.15, n_samples)
    data['Hyperlipidemia'] = np.random.binomial(1, 0.25, n_samples)
    data['Diabetes'] = np.random.binomial(1, 0.12, n_samples)
    
    # Lab values
    data['Total_Cholesterol'] = np.random.uniform(150, 300, n_samples)
    data['LDL'] = np.random.uniform(70, 200, n_samples)
    data['HDL'] = np.random.uniform(30, 80, n_samples)
    data['Triglyceride'] = np.random.uniform(80, 300, n_samples)
    data['AST'] = np.random.uniform(15, 50, n_samples)
    data['ALT'] = np.random.uniform(15, 55, n_samples)
    data['ALP'] = np.random.uniform(40, 120, n_samples)
    data['Creatinine'] = np.random.uniform(0.7, 1.5, n_samples)
    data['GFR'] = np.random.uniform(60, 120, n_samples)
    data['CRP'] = np.random.uniform(0.5, 10, n_samples)
    data['Hemoglobin'] = np.random.uniform(11, 17, n_samples)
    data['Vitamin_D'] = np.random.uniform(10, 50, n_samples)
    
    # Body composition
    data['FM'] = np.random.uniform(10, 40, n_samples)
    data['FFM'] = np.random.uniform(35, 70, n_samples)
    data['TBW'] = np.random.uniform(30, 50, n_samples)
    data['ECW'] = np.random.uniform(12, 25, n_samples)
    data['ICW'] = np.random.uniform(18, 30, n_samples)
    data['BMR'] = np.random.uniform(1200, 2200, n_samples)
    
    # Create features
    X = np.column_stack([data[feature] for feature in feature_names])
    
    # Create realistic target
    risk_score = (
        (data['Age'] > 50).astype(int) * 0.3 +
        (data['BMI'] > 30).astype(int) * 0.3 +
        (data['Gender'] == 0).astype(int) * 0.2 +
        data['Diabetes'].astype(int) * 0.2 +
        (data['Total_Cholesterol'] > 240).astype(int) * 0.1
    )
    
    y = np.random.binomial(1, np.clip(risk_score, 0.1, 0.8))
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    scaler = StandardScaler()
    scaler.fit(X)
    
    return model, scaler, feature_names

def create_enhanced_patient_form():
    """Create enhanced patient input form"""
    st.markdown("## 👤 Patient Information")
    
    # Demographics
    with st.container():
        st.markdown("### 📋 Demographics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            age = st.number_input("Age (years)", min_value=18, max_value=100, value=45, help="Patient's current age")
        
        with col2:
            gender = st.selectbox("Gender", ["Female", "Male"], help="Biological gender")
        
        with col3:
            height = st.number_input("Height (cm)", min_value=140, max_value=200, value=170, help="Height in centimeters")
        
        with col4:
            weight = st.number_input("Weight (kg)", min_value=40, max_value=150, value=70, help="Current weight")
    
    st.markdown("---")
    
    # Medical History
    with st.container():
        st.markdown("### 🏥 Medical History")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            cad = st.checkbox("Coronary Artery Disease", help="History of CAD")
        
        with col2:
            hypothyroidism = st.checkbox("Hypothyroidism", help="Thyroid dysfunction")
        
        with col3:
            hyperlipidemia = st.checkbox("Hyperlipidemia", help="High cholesterol/lipids")
        
        with col4:
            diabetes = st.checkbox("Diabetes Mellitus", help="Type 1 or 2 diabetes")
    
    st.markdown("---")
    
    # Laboratory Values
    with st.container():
        st.markdown("### 🧪 Laboratory Values")
        
        # Lipid Profile
        st.markdown("#### Lipid Profile")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_chol = st.number_input("Total Cholesterol (mg/dL)", min_value=100, max_value=400, value=200)
        
        with col2:
            ldl = st.number_input("LDL (mg/dL)", min_value=50, max_value=300, value=130)
        
        with col3:
            hdl = st.number_input("HDL (mg/dL)", min_value=20, max_value=100, value=50)
        
        with col4:
            triglycerides = st.number_input("Triglycerides (mg/dL)", min_value=50, max_value=500, value=150)
        
        # Liver Function
        st.markdown("#### Liver Function Tests")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            ast = st.number_input("AST (U/L)", min_value=10, max_value=200, value=25)
        
        with col2:
            alt = st.number_input("ALT (U/L)", min_value=10, max_value=200, value=30)
        
        with col3:
            alp = st.number_input("ALP (U/L)", min_value=40, max_value=200, value=75)
        
        # Other Labs
        st.markdown("#### Other Laboratory Values")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            creatinine = st.number_input("Creatinine (mg/dL)", min_value=0.5, max_value=3.0, value=1.0, step=0.1)
        
        with col2:
            gfr = st.number_input("GFR (mL/min/1.73m²)", min_value=30, max_value=150, value=90)
        
        with col3:
            crp = st.number_input("CRP (mg/L)", min_value=0.0, max_value=50.0, value=2.0, step=0.1)
        
        with col4:
            hemoglobin = st.number_input("Hemoglobin (g/dL)", min_value=8.0, max_value=18.0, value=14.0, step=0.1)
        
        with col5:
            vitamin_d = st.number_input("Vitamin D (ng/mL)", min_value=5, max_value=100, value=25)
    
    # Advanced Parameters
    with st.expander("🔬 Advanced Body Composition (Optional)", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            fm = st.number_input("Fat Mass (kg)", min_value=5.0, max_value=50.0, value=20.0)
            ffm = st.number_input("Fat-Free Mass (kg)", min_value=30.0, max_value=80.0, value=50.0)
            tbw = st.number_input("Total Body Water (L)", min_value=25.0, max_value=60.0, value=35.0)
        
        with col2:
            ecw = st.number_input("Extracellular Water (L)", min_value=10.0, max_value=30.0, value=15.0)
            icw = st.number_input("Intracellular Water (L)", min_value=15.0, max_value=35.0, value=20.0)
            bmr = st.number_input("BMR (kcal)", min_value=1000, max_value=2500, value=1500)
    
    # Calculate BMI
    bmi = weight / ((height/100) ** 2)
    
    # BMI Display
    col1, col2, col3 = st.columns([1, 1, 2])
    with col2:
        bmi_color = "🟢" if bmi < 25 else "🟡" if bmi < 30 else "🔴"
        st.metric("Calculated BMI", f"{bmi:.1f} kg/m²", help=f"BMI Status: {bmi_color}")
    
    # Compile patient data
    patient_data = {
        'Age': age, 'Gender': 1 if gender == "Male" else 0, 'Height': height,
        'Weight': weight, 'BMI': bmi, 'CAD': int(cad), 'Hypothyroidism': int(hypothyroidism),
        'Hyperlipidemia': int(hyperlipidemia), 'Diabetes': int(diabetes),
        'Total_Cholesterol': total_chol, 'LDL': ldl, 'HDL': hdl, 'Triglyceride': triglycerides,
        'AST': ast, 'ALT': alt, 'ALP': alp, 'Creatinine': creatinine, 'GFR': gfr,
        'CRP': crp, 'Hemoglobin': hemoglobin, 'Vitamin_D': vitamin_d,
        'FM': fm, 'FFM': ffm, 'TBW': tbw, 'ECW': ecw, 'ICW': icw, 'BMR': bmr
    }
    
    return patient_data

def make_enhanced_prediction(model, scaler, patient_data, feature_names):
    """Make enhanced prediction with feature importance"""
    patient_df = pd.DataFrame([patient_data])
    
    for feature in feature_names:
        if feature not in patient_df.columns:
            patient_df[feature] = 0
    
    patient_df = patient_df[feature_names]
    patient_scaled = scaler.transform(patient_df)
    
    prediction = model.predict(patient_scaled)[0]
    probability = model.predict_proba(patient_scaled)[0]
    
    # Feature importance
    feature_importance = list(zip(feature_names, model.feature_importances_))
    feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
    
    return prediction, probability, feature_importance

def create_enhanced_visualizations(probability, feature_importance):
    """Create enhanced visualizations"""
    risk_prob = probability[1] * 100
    
    # Enhanced gauge chart
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=risk_prob,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Gallstone Risk (%)", 'font': {'size': 20}},
        delta={'reference': 50, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue", 'thickness': 0.3},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 40], 'color': 'lightgreen'},
                {'range': [40, 70], 'color': 'yellow'},
                {'range': [70, 100], 'color': 'lightcoral'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    
    fig_gauge.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    
    # Enhanced feature importance plot
    features, importances = zip(*feature_importance[:10])
    
    fig_features = px.bar(
        x=importances,
        y=features,
        orientation='h',
        title="Top 10 Contributing Factors",
        labels={'x': 'Importance Score', 'y': 'Clinical Factors'},
        color=importances,
        color_continuous_scale='plasma'
    )
    
    fig_features.update_layout(
        height=500,
        yaxis={'categoryorder': 'total ascending'},
        margin=dict(l=150, r=50, t=50, b=50),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    
    return fig_gauge, fig_features

def display_enhanced_results(patient_data, prediction, probability, feature_importance, ai_system):
    """Display enhanced results with fixed AI integration and animations"""
    st.markdown("""
    <div class="results-header">
        <h2>🎯 Comprehensive Risk Assessment Results</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Add a progress bar animation
    progress_bar = st.progress(0)
    for i in range(100):
        time.sleep(0.01)
        progress_bar.progress(i + 1)
    progress_bar.empty()
    
    risk_prob = probability[1] * 100
    
    # Determine risk level and styling
    if risk_prob >= 70:
        risk_level = "HIGH RISK"
        risk_class = "risk-high"
        risk_emoji = "🔴"
    elif risk_prob >= 40:
        risk_level = "MEDIUM RISK"
        risk_class = "risk-medium"
        risk_emoji = "🟡"
    else:
        risk_level = "LOW RISK"
        risk_class = "risk-low"
        risk_emoji = "🟢"
    
    # Main results display
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Risk display with enhanced styling
        st.markdown(f"""
        <div class="{risk_class}">
            {risk_emoji} {risk_level}<br>
            <div style="font-size: 32px; margin-top: 10px;">
                {risk_prob:.1f}%
            </div>
            <div style="font-size: 16px; margin-top: 5px; opacity: 0.9;">
                Probability Score
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Patient summary card with fixed text visibility
        st.markdown("""
        <div class="metric-card">
            <h4 style="color: #667eea !important;">👤 Patient Summary</h4>
            <div class="patient-summary-text">
        """, unsafe_allow_html=True)
        
        col1a, col1b = st.columns(2)
        with col1a:
            st.metric("Age", f"{patient_data['Age']} years")
            st.metric("BMI", f"{patient_data['BMI']:.1f} kg/m²")
        
        with col1b:
            st.metric("Gender", "Male" if patient_data['Gender'] == 1 else "Female")
            comorbidities = sum([patient_data['Diabetes'], patient_data['Hyperlipidemia'], 
                               patient_data['CAD'], patient_data['Hypothyroidism']])
            st.metric("Comorbidities", f"{comorbidities}")
        
        st.markdown("</div></div>", unsafe_allow_html=True)
    
    with col2:
        # Enhanced gauge visualization
        fig_gauge, fig_features = create_enhanced_visualizations(probability, feature_importance)
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    st.markdown("---")
    
    # Feature importance section
    st.markdown("""
    <div class="results-header">
        <h2>📊 Model Explanation & Feature Analysis</h2>
    </div>
    """, unsafe_allow_html=True)
    
    col3, col4 = st.columns([2, 1])
    
    with col3:
        st.plotly_chart(fig_features, use_container_width=True)
    
    with col4:
        st.markdown("""
        <div class="feature-card">
            <h4 style="color: #667eea !important;">🎯 Top Contributing Factors</h4>
        </div>
        """, unsafe_allow_html=True)
        
        for i, (feature, importance) in enumerate(feature_importance[:8], 1):
            importance_percent = importance * 100
            st.markdown(f"""
            <div class="factor-item">
                <strong style="color: #2c3e50 !important;">{i}. {feature}</strong><br>
                <span class="factor-weight">Weight: {importance:.3f} ({importance_percent:.1f}%)</span>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # AI-Generated Insights Section (FIXED)
    if ai_system:
        st.markdown("""
        <div class="results-header">
            <h2>🤖 AI-Powered Clinical Insights</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Create tabs for different AI analyses
        tab1, tab2 = st.tabs(["Clinical Analysis", "Patient Education"])
        
        with tab1:
            with st.spinner("🧠 Generating clinical analysis..."):
                try:
                    clinical_analysis = ai_system.generate_clinical_analysis(
                        patient_data, risk_level, risk_prob, feature_importance[:5]
                    )
                    
                    # Display the analysis 
                    st.markdown(f"""
                    <div class="ai-insight-card">
                        <div class="ai-insight-header">
                            🏥 Clinical Assessment
                        </div>
                        <div class="ai-insight-content">
                            {format_ai_response(clinical_analysis)}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Error generating clinical analysis: {str(e)}")
                    display_fallback_analysis(risk_level, patient_data, feature_importance)
        
        with tab2:
            with st.spinner("📚 Generating patient education..."):
                try:
                    patient_education = ai_system.generate_patient_education(
                        risk_level.replace(" RISK", ""), patient_data
                    )
                    
                    # Display patient education
                    st.markdown(f"""
                    <div class="ai-insight-card">
                        <div class="ai-insight-header">
                            📖 Patient Education Materials
                        </div>
                        <div class="ai-insight-content">
                            {format_ai_response(patient_education)}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Error generating patient education: {str(e)}")
                    display_fallback_education(risk_level, patient_data)
    else:
        st.markdown("## ℹ️ AI Insights Unavailable")
        st.info("Configure your Gemini API key to enable AI-powered clinical insights")
        display_fallback_analysis(risk_level, patient_data, feature_importance)
    
    return risk_level, risk_prob

def format_ai_response(text):
    """Format AI response text for better HTML display"""
    # Clean and format the text
    formatted_text = text.replace('\n\n', '<br><br>')
    formatted_text = formatted_text.replace('\n', '<br>')
    formatted_text = formatted_text.replace('**', '<strong>').replace('**', '</strong>')
    formatted_text = formatted_text.replace('*', '').replace('##', '<h4>').replace('#', '</h4>')
    
    # Handle bullet points
    lines = formatted_text.split('<br>')
    formatted_lines = []
    for line in lines:
        line = line.strip()
        if line.startswith('- '):
            line = f'<li style="color: #2c3e50 !important; margin-bottom: 5px;">{line[2:]}</li>'
        elif line and not line.startswith('<'):
            line = f'<p style="color: #2c3e50 !important; margin-bottom: 10px;">{line}</p>'
        formatted_lines.append(line)
    
    return ''.join(formatted_lines)

def display_fallback_analysis(risk_level, patient_data, feature_importance):
    """Display fallback analysis when AI is unavailable"""
    st.markdown("""
    <div class="ai-insight-card">
        <div class="ai-insight-header">
            📋 Standard Clinical Analysis
        </div>
        <div class="ai-insight-content">
    """, unsafe_allow_html=True)
    
    # Risk assessment
    if "HIGH" in risk_level:
        st.markdown("""
        <h4 style="color: #ff4444 !important;">High Risk Assessment:</h4>
        <ul style="color: #2c3e50 !important;">
            <li>Immediate clinical evaluation recommended</li>
            <li>Consider imaging studies (ultrasound)</li>
            <li>Gastroenterology consultation</li>
            <li>Aggressive lifestyle modifications</li>
            <li>Close monitoring for symptoms</li>
        </ul>
        """, unsafe_allow_html=True)
    elif "MEDIUM" in risk_level:
        st.markdown("""
        <h4 style="color: #ffaa00 !important;">Moderate Risk Assessment:</h4>
        <ul style="color: #2c3e50 !important;">
            <li>Regular clinical monitoring</li>
            <li>Lifestyle modifications recommended</li>
            <li>Consider imaging if symptoms develop</li>
            <li>Risk factor management</li>
            <li>Follow-up in 6-12 months</li>
        </ul>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <h4 style="color: #00C851 !important;">Low Risk Assessment:</h4>
        <ul style="color: #2c3e50 !important;">
            <li>Continue routine preventive care</li>
            <li>Maintain healthy lifestyle</li>
            <li>Standard follow-up schedule</li>
            <li>Patient education on risk factors</li>
        </ul>
        """, unsafe_allow_html=True)
    
    # Top factors explanation
    st.markdown("""
    <h4 style="color: #667eea !important;">Key Contributing Factors:</h4>
    <ul style="color: #2c3e50 !important;">
    """, unsafe_allow_html=True)
    
    for i, (feature, importance) in enumerate(feature_importance[:5], 1):
        impact_direction = "increases" if importance > 0 else "decreases"
        st.markdown(f"""
        <li style="color: #2c3e50 !important; margin-bottom: 5px;">
            <strong>{feature}</strong>: {impact_direction} risk (importance: {importance:.3f})
        </li>
        """, unsafe_allow_html=True)
    
    st.markdown("</ul></div></div>", unsafe_allow_html=True)

def display_fallback_education(risk_level, patient_data):
    """Display fallback patient education"""
    st.markdown("""
    <div class="ai-insight-card">
        <div class="ai-insight-header">
            📚 Standard Patient Education
        </div>
        <div class="ai-insight-content">
            <h4 style="color: #667eea !important;">What are gallstones?</h4>
            <p style="color: #2c3e50 !important;">Gallstones are hardened deposits that form in your gallbladder, a small organ that stores bile.</p>
            
            <h4 style="color: #667eea !important;">Common symptoms to watch for:</h4>
            <ul style="color: #2c3e50 !important;">
                <li>Pain in upper right abdomen</li>
                <li>Nausea and vomiting</li>
                <li>Pain between shoulder blades</li>
                <li>Indigestion after fatty meals</li>
            </ul>
            
            <h4 style="color: #667eea !important;">Lifestyle recommendations:</h4>
            <ul style="color: #2c3e50 !important;">
                <li>Maintain healthy weight</li>
                <li>Eat regular, balanced meals</li>
                <li>Choose high-fiber, low-fat foods</li>
                <li>Stay physically active</li>
                <li>Limit refined sugars and processed foods</li>
            </ul>
            
            <h4 style="color: #667eea !important;">When to contact your healthcare provider:</h4>
            <ul style="color: #2c3e50 !important;">
                <li>Severe abdominal pain</li>
                <li>Persistent nausea or vomiting</li>
                <li>Fever with abdominal pain</li>
                <li>Yellowing of skin or eyes</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)

def setup_api_interface():
    """API key setup interface"""
    with st.sidebar:
        st.markdown("### 🤖 AI Insights Configuration")
        
        if not GENAI_AVAILABLE:
            st.error("❌ Google AI not installed")
            st.code("pip install google-generativeai")
            return False
        
        # Check current API key status
        current_key = None
        try:
            current_key = st.secrets.get("GEMINI_API_KEY")
        except:
            pass
        
        if not current_key:
            current_key = os.getenv('GEMINI_API_KEY')
        
        if not current_key and 'gemini_api_key' in st.session_state:
            current_key = st.session_state.gemini_api_key
        
        if current_key:
            st.success("✅ API Key Configured")
            if st.button("🔄 Clear API Key", help="Clear the stored API key"):
                if 'gemini_api_key' in st.session_state:
                    del st.session_state.gemini_api_key
                if 'GEMINI_API_KEY' in os.environ:
                    del os.environ['GEMINI_API_KEY']
                st.rerun()
            return True
        else:
            st.warning("⚠️ API Key Required for AI Insights")
            
            # API key input
            api_key = st.text_input(
                "Enter Gemini API Key",
                type="password",
                help="Get free API key from Google AI Studio",
                placeholder="Enter your API key here..."
            )
            
            if api_key:
                st.session_state.gemini_api_key = api_key
                os.environ['GEMINI_API_KEY'] = api_key
                st.success("✅ API Key Set!")
                st.rerun()
            
            # Instructions
            st.info("""
            **Get your free API key:**
            1. Visit: [Google AI Studio](https://aistudio.google.com/app/apikey)
            2. Sign in with Google account
            3. Create new API key
            4. Copy and paste above
            
            **Features with API:**
            - Clinical AI analysis
            - Patient education
            - Interactive chat
            """)
            
            return False

def create_chat_interface(ai_system):
    """Create chat interface for AI interaction"""
    st.markdown("""
    <div class="results-header">
        <h2>💬 Chat with AI Assistant</h2>
    </div>
    """, unsafe_allow_html=True)
    
    if not ai_system:
        st.warning("🔑 AI chat requires API key configuration")
        return
    
    # Initialize chat history
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_messages = []
        st.rerun()
    
    # Display chat history
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about gallstone risk, symptoms, or prevention..."):
        # Add user message
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate AI response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                try:
                    chat_prompt = f"""
You are a medical AI assistant specializing in gallstone disease. 
Provide helpful, accurate information about gallstones, risk factors, symptoms, and prevention.
Keep responses concise and medically accurate.
Always remind users to consult healthcare professionals for medical advice.

User question: {prompt}
"""
                    response = ai_system.model.generate_content(
                        chat_prompt,
                        generation_config=ai_system.generation_config,
                        safety_settings=ai_system.safety_settings
                    )
                    
                    if response.text:
                        st.markdown(response.text)
                        st.session_state.chat_messages.append(
                            {"role": "assistant", "content": response.text}
                        )
                    else:
                        st.error("Unable to generate response. Please try again.")
                        
                except Exception as e:
                    st.error(f"Chat error: {str(e)}")

def main():
    """main application"""
    # Header with animation
    st.markdown("""
    <div class="main-header">
        <h1>Gallstone Risk Prediction with AI Insights</h1>
        <p>Advanced Clinical Decision Support System</p>
        <p style="font-size: 14px; opacity: 0.8;">AI-Powered Analysis By Mayur Wade</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Setup API and initialize systems
    api_configured = setup_api_interface()
    ai_system = initialize_ai_system() if api_configured else None
    
    # Load model
    model, scaler, feature_names = create_demo_model()
    
    # Enhanced Sidebar
    with st.sidebar:
        st.markdown("## 📊 System Status")
        
        # Model status
        st.markdown("### 🔬 Prediction Model")
        st.success("✅ Random Forest Model Ready")
        st.info(f"Features: {len(feature_names)}")
        
        # AI status
        st.markdown("### 🤖 AI Assistant")
        if ai_system:
            st.success("✅ AI Insights Enabled")
            st.success("✅ Gemini 2.0 Flash Connected")
        else:
            st.warning("⚠️ AI Insights Disabled")
            st.info("Configure API key to enable")
        
        st.markdown("---")
        
        # Information
        st.markdown("## ℹ️ About")
        st.info("""
        This system provides:
        - **Risk Prediction**: ML-based gallstone risk assessment
        - **Explainable AI**: Feature importance analysis
        - **Clinical Insights**: AI-powered recommendations
        - **Patient Education**: Personalized guidance
        - **Interactive Chat**: Real-time AI assistant
        """)
        
        st.markdown("### 🎯 Risk Levels")
        st.markdown("""
        - **🟢 Low**: < 40%
        - **🟡 Medium**: 40-70%
        - **🔴 High**: > 70%
        """)
        
        st.markdown("---")
        st.markdown("**⚠️ Disclaimer:** For screening purposes only. Always consult healthcare professionals for medical decisions.")
    
    # Main content with tabs
    tab1, tab2, tab3 = st.tabs(["🏥 Risk Assessment", "💬 AI Chat", "📊 System Info"])
    
    with tab1:
        # Patient form
        patient_data = create_enhanced_patient_form()
        
        # Prediction button
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            predict_button = st.button(
                "🔮 Generate Risk Assessment", 
                type="primary", 
                use_container_width=True,
                help="Analyze patient data and generate comprehensive risk assessment"
            )
        
        if predict_button:
            with st.spinner("⚡ Analyzing patient data..."):
                # Make prediction
                prediction, probability, feature_importance = make_enhanced_prediction(
                    model, scaler, patient_data, feature_names
                )
                
                # Display results
                risk_level, risk_prob = display_enhanced_results(
                    patient_data, prediction, probability, feature_importance, ai_system
                )
                
                # Report generation
                st.markdown("---")
                st.markdown("""
                <div class="results-header">
                    <h2>📄 Generate Report</h2>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns([1, 1, 1])
                with col2:
                    report_content = f"""
COMPREHENSIVE GALLSTONE RISK ASSESSMENT REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
System: Enhanced AI-Powered Clinical Decision Support

PATIENT INFORMATION:
- Age: {patient_data['Age']} years
- Gender: {'Male' if patient_data['Gender'] else 'Female'}
- BMI: {patient_data['BMI']:.1f} kg/m²
- Height: {patient_data['Height']} cm
- Weight: {patient_data['Weight']} kg

RISK ASSESSMENT:
- Risk Level: {risk_level}
- Probability Score: {risk_prob:.1f}%
- Model: Random Forest Classifier
- Features Analyzed: {len(feature_names)}

TOP CONTRIBUTING FACTORS:
"""
                    for i, (feature, importance) in enumerate(feature_importance[:8], 1):
                        report_content += f"{i}. {feature}: {importance:.3f} ({importance*100:.1f}%)\n"
                    
                    report_content += f"""
MEDICAL HISTORY:
- Diabetes: {'Yes' if patient_data['Diabetes'] else 'No'}
- Hyperlipidemia: {'Yes' if patient_data['Hyperlipidemia'] else 'No'}
- Coronary Artery Disease: {'Yes' if patient_data['CAD'] else 'No'}
- Hypothyroidism: {'Yes' if patient_data['Hypothyroidism'] else 'No'}

LABORATORY VALUES:
- Total Cholesterol: {patient_data['Total_Cholesterol']} mg/dL
- HDL: {patient_data['HDL']} mg/dL
- LDL: {patient_data['LDL']} mg/dL
- Triglycerides: {patient_data['Triglyceride']} mg/dL
- AST: {patient_data['AST']} U/L
- ALT: {patient_data['ALT']} U/L
- ALP: {patient_data['ALP']} U/L
- Creatinine: {patient_data['Creatinine']} mg/dL
- GFR: {patient_data['GFR']} mL/min/1.73m²

{'='*60}
AI-POWERED INSIGHTS: {'ENABLED' if ai_system else 'DISABLED'}
{'='*60}

CLINICAL RECOMMENDATIONS:
{"AI-generated clinical insights available in full system report." if ai_system else "Standard risk assessment completed. Consider clinical correlation."}

{'='*60}
SYSTEM INFORMATION:
- Model: Random Forest with Feature Importance
- AI Integration: {'Google Gemini 1.5 Flash' if ai_system else 'Not Available'}
- Generated by: Enhanced Gallstone Risk Prediction System
{'='*60}

Disclaimer: This report is for screening and educational purposes only. 
Always consult qualified healthcare professionals for medical decisions.
Clinical correlation is essential for proper patient management.
"""
                    
                    st.download_button(
                        label="📥 Download Comprehensive Report",
                        data=report_content,
                        file_name=f"gallstone_assessment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain",
                        use_container_width=True,
                        help="Download detailed assessment report"
                    )
    
    with tab2:
        if ai_system:
            create_chat_interface(ai_system)
        else:
            st.markdown("""
            <div class="ai-insight-card">
                <div class="ai-insight-header">
                    🔐 AI Chat Unavailable
                </div>
                <div class="ai-insight-content">
                    <p>Configure your Gemini API key in the sidebar to enable:</p>
                    <ul>
                        <li>Interactive AI medical assistant</li>
                        <li>Real-time Q&A about gallstones</li>
                        <li>Personalized health guidance</li>
                        <li>Symptom assessment support</li>
                    </ul>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with tab3:
        st.markdown("""
        <div class="results-header">
            <h2>🔧 System Information</h2>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📦 Dependencies")
            st.code("""
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.15.0
scikit-learn>=1.3.0
google-generativeai>=0.3.0
            """)
            
            st.markdown("### 🎨 Features")
            st.success("✅ Animated UI Components")
            st.success("✅ Responsive Design")
            st.success("✅ Real-time Visualization")
            st.success("✅ Export Functionality")
            
        with col2:
            st.markdown("### ⚙️ Model Details")
            st.info(f"""
            **Algorithm**: Random Forest Classifier  
            **Features**: {len(feature_names)} clinical parameters  
            **Training**: 319 synthetic samples  
            **Performance**: Optimized for gallstone risk  
            **Explainability**: Feature importance analysis  
            """)
            
            st.markdown("### 🤖 AI Integration")
            if ai_system:
                st.success("**Model**: Google Gemini 2.0 Flash")
                st.success("**Status**: Connected & Ready")
                st.success("**Features**: Clinical Analysis, Education, Chat")
            else:
                st.warning("**Status**: Not Connected")
                st.info("**Required**: Gemini API Key")
        
        st.markdown("---")
        
        st.markdown("### 🚀 Features")
        st.markdown("""
        - **🔮 Machine Learning**: Random Forest-based risk prediction with feature importance
        - **🤖 AI Integration**: Google Gemini for clinical insights and patient education  
        - **💬 Interactive Chat**: Real-time AI medical assistant for Q&A
        - **📊 Visualizations**: Interactive gauge charts and feature importance plots
        - **🎨 Animations**: Smooth UI animations and progress indicators
        - **📱 Responsive**: Modern, mobile-friendly interface
        - **📄 Reports**: Comprehensive assessment reports with download
        - **🔒 Privacy**: Secure API key handling and data protection
        """)
        
        st.markdown("### 📈 Version History")
        st.markdown("""
        - **v3.0**: Fixed AI integration, added animations, improved text visibility
        - **v2.0**: Enhanced AI integration with chat interface  
        - **v1.5**: Added explainable AI and feature importance
        - **v1.0**: Basic gallstone risk prediction
        """)
        
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #f8f9ff 0%, #e8eeff 100%); border-radius: 10px;">
            <h4 style="color: #667eea;">🏥 Enhanced Gallstone Risk Prediction System</h4>
            <p style="color: #2c3e50;">Powered by Machine Learning & AI • Built for Healthcare Professionals By Mayur Wade</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()