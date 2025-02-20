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

# Store a negative complaint (for escalation follow-up)
if "last_complaint" not in st.session_state:
    st.session_state.last_complaint = None

st.set_page_config(page_title="AI Sentiment Analysis Customer Service Agent", layout="wide")
st.markdown("""
    <style>
    html, body, .css-18e3th9, .css-1d391kg, .stApp, .main {
         background-color: #000000 !important;
         color: #FFFFFF !important;
    }
    * {
         color: #FFFFFF !important;
    }
    .block-container {
         background-color: #000000;
         color: #FFFFFF;
         padding: 2rem;
    }
    .stButton > button {
         background-color: #007BFF !important;  /* Blue button */
         color: #FFFFFF !important;  /* White text */
         border: none;
         padding: 0.5rem 1rem;
         border-radius: 5px;
         font-size: 16px;
         font-weight: bold;
         cursor: pointer;
    }
    .stButton > button:hover {
         background-color: #0056b3 !important;
    }
    .stTextInput > div > div > input {
         background-color: #333333;
         color: #FFFFFF;
         border: 1px solid #007BFF;
    }
    </style>
    """, unsafe_allow_html=True)


st.title("AI Sentiment Analysis Customer Service Agent Demo")
st.markdown("""
This demo uses semantic search, sentiment analysis, escalation logging for email follow up, fined tuned GPT model and GPT-4 fallback.
Please enter a perfume-related query.
""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Chat Interface", "Escalation Log Dashboard", "Feedback Log Dashboard", "Data Visualisations Dashboard"])

with tab1:
    with st.form(key="query_form"):
        user_input = st.text_input("Enter your query here:")
        submit_button = st.form_submit_button(label="Submit")  
        
        if submit_button and user_input:
            # Check if input contains an email address
            email_match = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", user_input)
            if email_match:
                user_email = email_match.group(0)
                if st.session_state.last_complaint:
                    update_escalation_log(st.session_state.last_complaint, f"Follow-up with Email: {user_email}")
                    st.success("Thank you. Someone will be in touch within the next 24 hours to resolve your issue.")
                    st.session_state.last_complaint = None
                else:
                    log_escalation(user_input, "NEGATIVE", 0.95, escalation_reason=f"Email provided: {user_email} but no previous complaint stored")
                    st.info("Thank you. We'll reach out if needed.")
            else:
                # Process query using the AI agent
                response = handle_user_query(user_input)
                sentiment, confidence = analyze_sentiment(user_input)
                st.markdown(f"**AI Agent Response:** {response}")
                st.markdown(f"**Sentiment:** {sentiment} (Confidence: {round(confidence, 2)})")
                
                # Store complaint for human escalation
                if response.strip().startswith("I'm really sorry"):
                    st.session_state.last_complaint = user_input

with tab2:
    st.header("Escalation Log Dashboard")
    if os.path.exists(ESCALATION_CSV_PATH):
        try:
            df_escalation = pd.read_csv(ESCALATION_CSV_PATH, quoting=csv.QUOTE_ALL)
            st.dataframe(df_escalation)
            st.download_button("Download Escalation Log", df_escalation.to_csv(index=False, quoting=csv.QUOTE_ALL), "escalation_log.csv", "text/csv")
        except Exception as e:
            st.error(f"Error loading escalation log: {e}")
    else:
        st.info("No escalation records available.")

with tab3:
    st.header("Feedback Log Dashboard")
    if os.path.exists(FEEDBACK_LOG_PATH):
        try:
            df_feedback = pd.read_csv(FEEDBACK_LOG_PATH)
            st.dataframe(df_feedback)
            st.download_button("Download Feedback Log", df_feedback.to_csv(index=False), "feedback_log.csv", "text/csv")
        except Exception as e:
            st.error(f"Error loading feedback log: {e}")
    else:
        st.info("No feedback records available.")

with tab4:
    st.header("Data Visualisations Dashboard")
    
    # Visualisation 1: Sentiment Trends (Feedback Log)
    if os.path.exists(FEEDBACK_LOG_PATH):
        try:
            df_feedback = pd.read_csv(FEEDBACK_LOG_PATH)
            if 'Timestamp' not in df_feedback.columns:
                df_feedback['Timestamp'] = pd.to_datetime('now')
            df_feedback['Date'] = pd.to_datetime(df_feedback['Timestamp'], errors='coerce', infer_datetime_format=True).dt.date
            df_feedback = df_feedback.dropna(subset=['Date'])
            sentiment_trends = df_feedback.groupby(['Date', 'Sentiment']).size().reset_index(name='Count')
            fig1 = px.line(sentiment_trends, x='Date', y='Count', color='Sentiment', title="Sentiment Trends (Feedback Log)")
            st.plotly_chart(fig1, use_container_width=True)
        except Exception as e:
            st.error(f"Error generating sentiment trends: {e}")
    else:
        st.info("No feedback records available for sentiment trends.")

    # Visualisation 2: Escalation Rates (Escalation Log)
    if os.path.exists(ESCALATION_CSV_PATH):
        try:
            df_escalation = pd.read_csv(ESCALATION_CSV_PATH, quoting=csv.QUOTE_ALL)
            df_escalation['Date'] = pd.to_datetime(df_escalation['Timestamp'], errors='coerce', infer_datetime_format=True)
            df_escalation = df_escalation.dropna(subset=['Date'])
            df_escalation['Date'] = df_escalation['Date'].dt.date
            start_date = pd.to_datetime("2025-02-02").date()
            df_escalation = df_escalation[df_escalation['Date'] >= start_date]
            escalation_rates = df_escalation.groupby('Date').size().reset_index(name='Count')
            fig2 = px.bar(escalation_rates, x='Date', y='Count', title="Daily Escalation Rates")
            st.plotly_chart(fig2, use_container_width=True)
        except Exception as e:
            st.error(f"Error generating escalation rates: {e}")
    else:
        st.info("No escalation records available for escalation rates.")

    # Visualisation 3: Common Queries (Feedback Log)
    if os.path.exists(FEEDBACK_LOG_PATH):
        try:
            df_feedback = pd.read_csv(FEEDBACK_LOG_PATH)
            common_queries = df_feedback['User Input'].value_counts().reset_index()
            common_queries.columns = ['User Input', 'Count']
            fig3 = px.bar(common_queries.head(10), x='User Input', y='Count', title="Top 10 Common Queries")
            st.plotly_chart(fig3, use_container_width=True)
        except Exception as e:
            st.error(f"Error generating common queries chart: {e}")
    else:
        st.info("No feedback records available for common queries.")
