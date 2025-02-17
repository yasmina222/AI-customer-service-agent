AI Sentiment Analysis Customer Service Agent

This project is a AI-powered customer service agent that uses sentiment analysis to handle user queries. It checks the tone of each query and responds with an answer from a knowledge base or a generated response using a fine-tuned GPT model. If a query is too negative, the issue is logged for human follow-up within 24 hours.

It essentially streamlines customer service, by automating routine queries and ensuring that urgent issues are quickly escalated for resolution, significantly reducing staffing costs and response times. As a result, it improves customer satisfaction, strengthens the company’s reputation, and delivers measurable returns on investment through enhanced operational efficiency.

# How It Works

- **Web Interface:** Built with Streamlit, the app lets users enter queries and see responses.
- **Sentiment Analysis:** The agent evaluates each query as positive, neutral, or negative.
- **Knowledge Base & GPT:** It first looks for answers in a knowledge base. If none are found, it uses a fine-tuned GPT model as a fallback.
- **Logging:** Escalation for high priority cases are logged with a contact email address for follow up, and overall feedback logs are maintained for further review.
- **Plot Graphs** Are availble to monitor trends of user queries (and potential business problems) and effectiveness of AI agent.

## Getting Started

1. **Clone the repository** from GitHub.
2. **Install the dependencies** by running:
   ```bash
   pip install -r requirements.txt
3. **Run in the terminal** Streamlit run streamlit_app.py

