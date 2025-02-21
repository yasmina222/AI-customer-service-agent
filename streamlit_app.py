import streamlit as st
import re
import os
import csv
import pandas as pd
from datetime import datetime
from agent import handle_user_query, log_escalation, append_to_feedback_log, analyze_sentiment
import plotly.express as px  

ESCALATION_CSV_PATH = "escalation_log.txt"
FEEDBACK_LOG_PATH = "feedback_log.csv"

#Negative complaint recorded for escalation
if "last_complaint" not in st.session_state:
    st.session_state.last_complaint = None

# Page settings
st.set_page_config(
    page_title="AI Sentiment Analysis Customer Service Agent",
    layout="wide",
    page_icon="🤖"
)

#Custom CSS for a modern theme
st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
    
    * {{
        font-family: 'Inter', sans-serif !important;
        color: #ffffff !important;
    }}
    
    .stApp {{
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%) !important;
    }}
    
    /* Changed input text color to black */
    .stTextInput input {{
        background: rgba(255,255,255,0.1) !important;
        border: 1px solid #4a40a3 !important;
        border-radius: 10px !important;
        color: #000000 !important;  /* BLACK TEXT */
        padding: 12px !important;
    }}
    
    /* Placeholder text styling */
    .stTextInput input::placeholder {{
        color: #666666 !important;
        opacity: 1 !important;
    }}
    
    /* Brighter button for better contrast */
    .stButton>button {{
        background: #7c6cf0 !important;
        color: white !important;
        border-radius: 12px !important;
        padding: 12px 24px !important;
        border: none !important;
        font-weight: 600 !important;
    }}

    /* Keep the rest of your existing CSS below... */
    .stMarkdown h1 {{
        color: #ffffff !important;
        font-size: 2.5rem !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }}
    
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px !important;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        background: rgba(108,92,231,0.2) !important;
        border-radius: 8px !important;
        padding: 12px 24px !important;
        margin: 0 4px !important;
    }}
    
    .stTabs [aria-selected="true"] {{
        background: #6c5ce7 !important;
    }}
    
    .stDataFrame {{
        background: rgba(255,255,255,0.05) !important;
        border-radius: 12px !important;
        border: 1px solid #4a40a3 !important;
    }}
    
    .ai-agent-header {{
        display: flex;
        align-items: center;
        gap: 2rem;
        margin-bottom: 2rem;
    }}
    
    .ai-agent-image {{
        width: 120px;
        border-radius: 50%;
        border: 3px solid #6c5ce7;
        box-shadow: 0 4px 15px rgba(108,92,231,0.3);
    }}
    </style>
""", unsafe_allow_html=True)

#Image
st.markdown("""
    <div class="ai-agent-header">
        <img src="https://img.icons8.com/?size=100&id=qfBYZBppgo9X&format=png&color=000000" class="ai-agent-image">
        <div>
            <h1>Sentiment Analysis AI Agent</h1>
            <p style="font-size: 1.1rem; color: #ffffff; max-width: 800px;">
                Your 24/7 AI Customer Service Solution for Enhanced Customer Experience and Reduced Internal Workload
            </p>
        </div>
    </div>
""", unsafe_allow_html=True)

#Benefits Section
col1, col2 = st.columns(2)
with col1:
    st.markdown("""
    ### ✨ Key Benefits
    - **Instant Response** to common customer inquiries
    - **Urgent Escalation** to human agents through sentiment analysis scoring
    - **24/7 Availability** with consistent service quality
    - **High Priority** cases prioritised for enhanced customer experience and company reputation.  
    - **Reduced Workload** for human agents by handling 70%+ 
    """)

with col2:
    st.markdown("""
    ### 🚀 Advanced Features
    - Real-time sentiment scoring & emotion detection
    - Automatic escalation logging with email capture for follow up within 24 hours.
    - Hybrid AI system (GPT-4 + fine-tuned models)
    - Continuous learning from customer interactions
    """)

#Example Queries
st.markdown("""
    ### 💡 This demo has been tailored for a online perfume store. Test it for yourself and enter a perfume-related query!!
    - When is my perfume coming back into stock?
    - I need to talk to someone else right now!
    - What is the best perfume for summer?
    - I'm not happy with this service!
    - How can i request a refund?
    - My order arrived broken
""")

#Main Tabs
tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat Interface", "📈 Escalations", "📋 Feedback Log", "📊 Analytics"])

with tab1:
    with st.form(key="query_form"):
        user_input = st.text_input("Enter your query here:", placeholder="Type your perfume-related question...")
        submit_button = st.form_submit_button(label="Submit Query")
        
        if submit_button and user_input:
            email_match = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", user_input)
            if email_match:
                user_email = email_match.group(0)
                if st.session_state.last_complaint:
                    log_escalation(st.session_state.last_complaint, "NEGATIVE", 0.95, f"Follow-up with Email: {user_email}")
                    st.success(" Thank you! Our team will contact you within 24 hours to resolve your issue.")
                    st.session_state.last_complaint = None
                else:
                    log_escalation(user_input, "NEGATIVE", 0.95, escalation_reason=f"Email provided: {user_email}")
                    st.info("Thank you! We'll reach out within the next 24 hours.")
            else:
                response = handle_user_query(user_input)
                sentiment, confidence = analyze_sentiment(user_input)
                
                st.markdown(f"""
                    <div style="background: rgba(108,92,231,0.15); padding: 1.5rem; border-radius: 12px; margin: 1rem 0;">
                        <h4 style="color: #ffffff; margin-bottom: 0.5rem;">AI Response</h4>
                        <p style="margin-bottom: 0; color: #ffffff;">{response}</p>
                        <div style="margin-top: 1rem; color: #ffffff;">
                            Sentiment: <strong>{sentiment}</strong> (Confidence: {round(confidence, 2)})
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                if response.strip().startswith("I'm really sorry"):
                    st.session_state.last_complaint = user_input

with tab2:
    st.header("🚨 Escalation Management")
    if os.path.exists(ESCALATION_CSV_PATH):
        try:
            df_escalation = pd.read_csv(ESCALATION_CSV_PATH, quoting=csv.QUOTE_ALL)
            st.dataframe(df_escalation.style.highlight_max(axis=0, color='#2a255e'))
            st.download_button("Export Escalation Log", df_escalation.to_csv(index=False, quoting=csv.QUOTE_ALL), "escalation_log.csv")
        except Exception as e:
            st.error(f"Error loading escalation log: {e}")
    else:
        st.info("No escalation records found")

with tab3:
    st.header("📝 Customer Feedback")
    if os.path.exists(FEEDBACK_LOG_PATH):
        try:
            df_feedback = pd.read_csv(FEEDBACK_LOG_PATH)
            st.dataframe(df_feedback.style.background_gradient(cmap='Purples'))
            st.download_button("Export Feedback Log", df_feedback.to_csv(index=False), "feedback_log.csv")
        except Exception as e:
            st.error(f"Error loading feedback log: {e}")
    else:
        st.info("No feedback records available")

with tab4:
    st.header("📊 Customer Insights")
    
    # Top 10 Common Queries visualisation
    if os.path.exists(FEEDBACK_LOG_PATH):
        try:
            df_feedback = pd.read_csv(FEEDBACK_LOG_PATH)
            common_queries = df_feedback['User Input'].value_counts().reset_index()
            common_queries.columns = ['User Input', 'Count']
            fig = px.bar(
                common_queries.head(10), 
                x='User Input', 
                y='Count', 
                title="Top 10 Customer Queries",
                color_discrete_sequence=['#6c5ce7']
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white'
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error generating insights: {e}")
    else:
        st.info("No data available for visualization")