# Customer Service AI Agent
An intelligent customer service agent that combines semantic search, sentiment analysis, and generative AI to handle customer queries autonomously while escalating high-priority cases to human agents.
Built for a perfume e-commerce context, but the architecture is domain-agnostic.

## What It Does

Answers customer queries using a knowledge base with semantic search
Detects negative sentiment and escalates frustrated customers automatically
Falls back to a fine-tuned GPT model when knowledge base doesn't have an answer
Captures email addresses for follow-up on escalated cases
Logs all interactions for analysis and continuous improvement


## Technical Implementation
Semantic Search (RAG)
Knowledge base questions are embedded using OpenAI's text-embedding-3-small and stored as NumPy arrays. At query time:

User query is embedded with the same model
Cosine similarity is calculated against all stored embeddings
If max similarity ≥ 0.7, the corresponding answer is returned
Below threshold triggers the fine-tuned model fallback

Sentiment-Based Routing
Every query passes through sentiment classification before any response logic:

NEGATIVE + confidence > 0.9: Immediate escalation, email capture
NEGATIVE + confidence ≤ 0.9: Normal flow, but logged
NEUTRAL/POSITIVE: Standard query handling

The sentiment model (cardiffnlp/twitter-roberta-base-sentiment) outputs three labels mapped to business logic.

Fine-Tuned Model
A GPT-3.5-turbo model fine-tuned on domain-specific Q&A pairs handles edge cases outside the knowledge base. System prompts adapt based on detected sentiment:

Positive → Enthusiastic tone
Negative → Empathetic tone
Neutral → Informative tone

Logging
Three log files track system behaviour:
FilePurposefeedback_log.csvAll queries with sentiment scores and resolution methodescalation_log.txtHigh-priority cases with contact detailssentiment_log.csvRaw sentiment classification results


## Limitations

Embedding storage uses flat NumPy arrays (works for ~1000 entries, not production scale)
No incremental index updates; full re-embedding required for KB changes
Single-turn only; no conversation memory
Sentiment model optimised for English Twitter data

## Future Improvements

Migrate to vector database (Pinecone, Azure AI Search) for scale
Add conversation context window
Implement feedback loop for automatic KB expansion
A/B test different similarity thresholds

Dependencies

openai - Embeddings and chat completions
transformers - Sentiment analysis model
scikit-learn - Cosine similarity
numpy / pandas - Data handling
streamlit - Web interface
plotly - Analytics visualisation

