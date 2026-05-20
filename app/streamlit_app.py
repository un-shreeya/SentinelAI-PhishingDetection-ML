import streamlit as st
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils import predict_sms, predict_url, load_models
from training.preprocess import extract_url_features
import plotly.graph_objects as go
import re

# Page config
st.set_page_config(
    page_title="SentinelAI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .title-main {
        font-size: 3em;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5em;
    }
    .subtitle {
        font-size: 1.1em;
        color: #666;
        margin-bottom: 2em;
    }
    .risk-high {
        color: #d32f2f;
        font-weight: bold;
    }
    .risk-medium {
        color: #f57c00;
        font-weight: bold;
    }
    .risk-low {
        color: #388e3c;
        font-weight: bold;
    }
    .metric-box {
        padding: 1em;
        border-radius: 0.5em;
        background: #f5f5f5;
        margin: 0.5em 0;
    }
    </style>
""", unsafe_allow_html=True)

# Header
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown('<div class="title-main">🛡️ SentinelAI</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Advanced Phishing & Spam Detection System</div>', unsafe_allow_html=True)
with col2:
    st.info("✓ Models Ready" if all(load_models()) else "⚠ Models Loading...")

st.divider()

# Main tabs
tab1, tab2, tab3 = st.tabs(["📧 SMS/Email/Message", "🔗 URL Analysis", "📊 About"])

# ==================== TAB 1: SMS/EMAIL/MESSAGE ====================
with tab1:
    st.subheader("🔍 Analyze Message Content")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        input_type = st.selectbox("Input Type", ["Email/SMS", "Message"], key="msg_type")
    
    with col2:
        st.write("")  # spacing
    
    message_text = st.text_area(
        "Paste your message, email, or SMS here",
        height=200,
        placeholder="Example: Click here immediately to verify your account! Your password is needed...",
        key="msg_input"
    )
    
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        analyze_btn = st.button("🔍 Analyze Message", key="msg_btn", use_container_width=True)
    
    if analyze_btn:
        if not message_text.strip():
            st.error("❌ Please paste a message to analyze")
        else:
            with st.spinner("🔄 Analyzing message..."):
                result = predict_sms(message_text)
            
            # Risk assessment display
            risk_color = "red" if result['probability'] > 0.7 else "orange" if result['probability'] > 0.4 else "green"
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Spam Probability", f"{result['probability']*100:.1f}%")
            with col2:
                st.metric("Risk Level", result['risk'])
            with col3:
                if result['probability'] >= 0.85:
                    classification = "🚨 HIGH RISK SPAM"

                elif result['probability'] >= 0.6:
                    classification = "⚠️ SUSPICIOUS"

                else:
                    classification = "✓ LEGITIMATE"

                st.metric("Classification", classification)
            
            st.divider()
            
            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=result['probability']*100,
                title="Spam Score",
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': risk_color},
                    'steps': [
                        {'range': [0, 40], 'color': "#e8f5e9"},
                        {'range': [40, 70], 'color': "#fff3e0"},
                        {'range': [70, 100], 'color': "#ffebee"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 70
                    }
                }
            ))
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            st.divider()
            
            # Explanation
            st.subheader("🔎 Analysis Details")
            st.markdown(f"**Detected Indicators:**\n\n{result['explanation']}")
            
            # Message preview
            with st.expander("📝 Message Preview"):
                st.text(message_text[:500] + ("..." if len(message_text) > 500 else ""))

# ==================== TAB 2: URL ANALYSIS ====================
with tab2:
    st.subheader("🔗 Analyze URL Security")
    
    url_input = st.text_input(
        "Paste the URL to analyze",
        placeholder="https://example.com or http://suspicious-domain.xyz",
        key="url_input"
    )
    
    col1, col2 = st.columns([1, 3])
    with col1:
        analyze_url_btn = st.button("🔍 Analyze URL", key="url_btn", use_container_width=True)
    
    if analyze_url_btn:
        if not url_input.strip():
            st.error("❌ Please paste a URL to analyze")
        else:
            with st.spinner("🔄 Analyzing URL..."):
                url_features = extract_url_features(url_input)
                result = predict_url(url_features)
            
            risk_color = "red" if result['probability'] > 0.7 else "orange" if result['probability'] > 0.4 else "green"
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Phishing Probability", f"{result['probability']*100:.1f}%")
            with col2:
                st.metric("Risk Level", result['risk'])
            with col3:
                st.metric("Classification", "⚠️ PHISHING" if result['label'] else "✓ SAFE")
            
            st.divider()
            
            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=result['probability']*100,
                title="Phishing Score",
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': risk_color},
                    'steps': [
                        {'range': [0, 40], 'color': "#e8f5e9"},
                        {'range': [40, 70], 'color': "#fff3e0"},
                        {'range': [70, 100], 'color': "#ffebee"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 70
                    }
                }
            ))
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            st.divider()
            
            # Explanation
            st.subheader("🔎 Analysis Details")
            st.markdown(f"**Detected Indicators:**\n\n{result['explanation']}")
            
            # URL features breakdown
            with st.expander("📊 URL Structure Analysis"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**URL Length:** {url_features['url_length']} chars")
                    st.write(f"**Number of Dots:** {url_features['num_dots']}")
                    st.write(f"**Contains @:** {'Yes ⚠️' if url_features['has_at'] else 'No ✓'}")
                with col2:
                    st.write(f"**Contains IP:** {'Yes ⚠️' if url_features['has_ip'] else 'No ✓'}")
                    st.write(f"**Domain Length:** {url_features['domain_len']} chars")
                    st.write(f"**Suspicious Keywords:** {url_features['suspicious_kw_count']}")

# ==================== TAB 3: ABOUT ====================
with tab3:
    st.subheader("📊 About SentinelAI")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🤖 How It Works
        
        SentinelAI uses **machine learning models** trained on thousands of real phishing attempts and spam messages to detect threats:
        
        **SMS/Email Analysis:**
        - TF-IDF text vectorization (3000 features, 1-2 grams)
        - Logistic Regression classification
        - Pattern analysis for urgency language, suspicious keywords, unusual formatting
        
        **URL Analysis:**
        - Random Forest classifier (150 trees)
        - URL structure analysis (length, dots, @ symbols, IP addresses)
        - Domain and TLD examination
        - Suspicious keyword detection
        """)
    
    with col2:
        st.markdown("""
        ### 🎯 Accuracy & Performance
        
        - **SMS/Email Model:** ~95% accuracy on validation set
        - **URL Model:** ~97% accuracy on validation set
        - Real-time analysis in milliseconds
        
        ### ⚠️ Disclaimer
        
        SentinelAI is a **detection tool** not a guarantee. Always:
        - Never click suspicious links
        - Don't share passwords in emails
        - Verify sender identity independently
        - Hover over links to see actual destination
        
        ### 🔒 Privacy
        Messages and URLs are analyzed locally and **not stored**.
        """)
    
    st.divider()
    
    st.info("""
    **🛡️ Best Practices:**
    
    1. **Enable 2FA** on all important accounts
    2. **Check sender email** carefully (look for spoofed domains)
    3. **Hover over links** before clicking to see real URL
    4. **Report phishing** to email providers
    5. **Never reply** with personal information
    6. **Use password managers** to avoid phishing tricks
    """)