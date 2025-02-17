import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from transformers import pipeline
import csv
import re  



# Paths to data files
KNOWLEDGE_BASE_PATH = "/Users/yasminalarouci/Chatbot/knowledge_base.csv"
KB_EMBEDDINGS_PATH = "/Users/yasminalarouci/Chatbot/knowledge_embeddings.npy"
SYNTHETIC_DATASET_PATH = "/Users/yasminalarouci/Chatbot/synthetic_dataset.csv"
SYNTHETIC_DATASET_EMBEDDINGS_PATH = "/Users/yasminalarouci/Chatbot/synthetic_embeddings.npy"
FEEDBACK_LOG_PATH = "/Users/yasminalarouci/Chatbot/feedback_log.csv"
ESCALATION_LOG_PATH = "/Users/yasminalarouci/Chatbot/escalation_log.txt"


# No need to pass api_key, Render will handle it
client = OpenAI()  


FINE_TUNED_MODEL = "ft:gpt-3.5-turbo-0125:personal::AzsqUdO7"

EMBEDDING_MODEL = "text-embedding-3-small"  

# Sentiment analysis model
SENTIMENT_MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"
sentiment_model = pipeline("text-classification", model=SENTIMENT_MODEL_NAME)

# Create and save OpenAI embeddings as .npy file.
def embed_data(input_path, output_path, column_name, force_regenerate=False):
    
    if os.path.exists(output_path) and not force_regenerate:
        print(f"Embeddings already exist at {output_path}. Skipping embedding generation.")
        return

    try:
        print(f"Generating new embeddings for {input_path}...")
        data = pd.read_csv(input_path)
        texts = data[column_name].tolist()
        
        # Process in batches to avoid API timeouts
        batch_size = 100
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            response = client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=batch
            )
            embeddings.extend([e.embedding for e in response.data])
            print(f"Processed {min(i+batch_size, len(texts))}/{len(texts)} records")

        np.save(output_path, np.array(embeddings))
        print(f"Embeddings saved to {output_path}")
        
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        raise

def validate_embeddings(data_path, embeddings_path):
    """Ensure embeddings match knowledge base records."""
    data = pd.read_csv(data_path)
    embeddings = np.load(embeddings_path)
    
    if len(data) != len(embeddings):
        raise ValueError(f"Mismatch: {len(data)} records vs {len(embeddings)} embeddings")
    
    print(f"Validation passed: {len(data)} records match embeddings")
    return True

def semantic_search(user_input, embeddings_path, data_path, threshold=0.7):
   # Find the best matching answer from the knowledge base
    try:
        data = pd.read_csv(data_path)
        
        # Check for an exact match 
        exact_match = data[data['question'].str.lower() == user_input.lower()]
        if not exact_match.empty:
            return exact_match.iloc[0]['answer']
        
        # Embeddings for semantic search & user query
        embeddings = np.load(embeddings_path)
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=user_input)
        user_embedding = np.array(response.data[0].embedding)
        
        # Calculate cosine similarities 
        similarities = cosine_similarity([user_embedding], embeddings)[0]
        max_similarity = np.max(similarities)
        best_index = np.argmax(similarities)
        
        print(f"DEBUG: Max similarity for '{user_input}' is {max_similarity:.2f}")
        
       
        if max_similarity >= threshold:
            return data.iloc[best_index]['answer']
        
        return None  
    except Exception as e:
        print(f"Error in semantic search: {e}")
        return None
 
# Classify text sentiment and log the results
def analyze_sentiment(text, threshold=0.7):
    label_mapping = {"LABEL_0": "NEGATIVE", "LABEL_1": "NEUTRAL", "LABEL_2": "POSITIVE"}
    
    result = sentiment_model(text)[0]
    sentiment_label = label_mapping[result['label']]
    confidence_score = result['score']

    
    if confidence_score < threshold:
        sentiment_label = "NEUTRAL"  


    with open("sentiment_log.csv", mode="a", newline="") as file:
        writer = csv.writer(file)
        file_exists = os.path.isfile("sentiment_log.csv")
        if not file_exists:
            writer.writerow(["User Input", "Sentiment", "Confidence"])
        writer.writerow([text, sentiment_label, round(confidence_score, 2)])

    return sentiment_label, confidence_score

#Log escalated cases into a CSV file
def log_escalation(user_input, sentiment, confidence, escalation_reason="High-confidence negative sentiment"):
    
    ESCALATION_CSV_PATH = "/Users/yasminalarouci/Chatbot/escalation_log.csv"
    escalation_data = {
        "User Input": user_input,
        "Sentiment": sentiment,
        "Confidence": round(confidence, 2),
        "Escalation Reason": escalation_reason,
        "Timestamp": pd.Timestamp.now()
    }
    file_exists = os.path.isfile(ESCALATION_CSV_PATH)
    with open(ESCALATION_CSV_PATH, mode="a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=escalation_data.keys(), quoting=csv.QUOTE_ALL)
        if not file_exists:
            writer.writeheader()
        writer.writerow(escalation_data)


    file_exists = os.path.isfile(ESCALATION_CSV_PATH)
    with open(ESCALATION_CSV_PATH, mode="a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=escalation_data.keys())

        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(escalation_data)

# Save to feedback log to improve future responses.
def append_to_feedback_log(user_input, sentiment, confidence, resolution):
    file_exists = os.path.isfile(FEEDBACK_LOG_PATH)

    with open(FEEDBACK_LOG_PATH, mode="a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["User Input", "Sentiment", "Confidence", "Resolution"])
        writer.writerow([user_input, sentiment, round(confidence, 2), resolution])

# Generate a response with fine-tuned model & set up fall back
def generate_finetuned_response(user_input, sentiment):  
    system_message = {
        "POSITIVE": "You are an enthusiastic and cheerful assistant.",
        "NEGATIVE": "You are a calm and empathetic assistant.",
        "NEUTRAL": "You are a neutral and informative assistant."
    }.get(sentiment, "You are a neutral and informative assistant.")

    messages = [{"role": "system", "content": system_message},
                {"role": "user", "content": user_input}]

    try:
        response = client.chat.completions.create(
            model=FINE_TUNED_MODEL,  
            messages=messages
        )
        return response.choices[0].message.content
    except Exception as e:
        try:
            # Fallback to GPT-4 if fine-tuned model fails
            response = client.chat.completions.create(
                model="gpt-4",
                messages=messages
            )
            return f"{response.choices[0].message.content} [Fallback]"
        except:
            return "Error generating response. Please try again later."

#Process user queries and generate responses        
def handle_user_query(user_input, use_knowledge_base=True):
    sentiment, confidence = analyze_sentiment(user_input)

    if sentiment == "NEGATIVE" and confidence > 0.9:
        log_escalation(user_input, sentiment, confidence)
        return "I'm really sorry for your negative experience. I will escalate your issue. Please provide your contact email details."
    
    if use_knowledge_base:
        knowledge_base_answer = semantic_search(
            user_input=user_input,
            embeddings_path=KB_EMBEDDINGS_PATH,
            data_path=KNOWLEDGE_BASE_PATH,
            threshold=0.7
        )
        if knowledge_base_answer:
            append_to_feedback_log(user_input, sentiment, confidence, resolution="KB Answer")
            return knowledge_base_answer
    
    # Fallback to fine-tuned model
    ft_response = generate_finetuned_response(user_input, sentiment)
    append_to_feedback_log(user_input, sentiment, confidence, resolution="Fine-Tuned Model")
    return ft_response


if __name__ == "__main__":
    print("AI Agent activated! Type 'exit' to end the conversation.")   
    
   # Initialise embeddings and start the chat loop
    embed_data(KNOWLEDGE_BASE_PATH, KB_EMBEDDINGS_PATH, "question")
    embed_data(SYNTHETIC_DATASET_PATH, SYNTHETIC_DATASET_EMBEDDINGS_PATH, "query")

    last_complaint = None  
    while True:
        # User input prompt
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("AI Agent: Goodbye!")
            break

        # Handle email inputs separately
        email_match = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", user_input)
        
        if email_match:
            user_email = email_match.group(0)

            if last_complaint:  
                log_escalation(last_complaint, "NEGATIVE", 0.95, escalation_reason=f"Follow-up with Email: {user_email}")
                print("AI Agent: Thank you. Someone will be in touch within the next 24 hours to resolve your issue.")
                last_complaint = None  
            else:
                print("AI Agent: Thank you. We'll reach out within the next 24 hours.")

            continue  
        sentiment, confidence = analyze_sentiment(user_input)

        # Check for escalation 
        if sentiment == "NEGATIVE" and confidence > 0.9:
            last_complaint = user_input  # Store the complaint, capture the email for follow-up by human agent
            print("AI Agent: I'm really sorry for your negative experience experience. I will escalate your issue. Please provide your email address.")
            continue  

        knowledge_base_answer = semantic_search(
            user_input=user_input,
            embeddings_path=KB_EMBEDDINGS_PATH,
            data_path=KNOWLEDGE_BASE_PATH,
            threshold=0.7  
        )

        # Respond with KB answer
        if knowledge_base_answer:
            print(f"AI Agent: {knowledge_base_answer}")
        else:
            # If no match found, use fine-tuned model
            ft_response = generate_finetuned_response(user_input, sentiment)
            print(f"AI Agent: {ft_response}")

            # Improvement tracking
            append_to_feedback_log(user_input, sentiment, confidence, resolution="Fine-Tuned Model")

