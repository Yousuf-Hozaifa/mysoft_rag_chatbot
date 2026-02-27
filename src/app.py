import os

# This forces the heavy AI models to download to your E drive instead of C!
os.environ["HF_HOME"] = "E:/huggingface_cache"

import time
import numpy as np
import faiss
import gradio as gr
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types
from dotenv import load_dotenv

# 1. Securely load API Key from the .env file
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found! Please add it to your .env file.")

client = genai.Client(api_key=GOOGLE_API_KEY)

# 2. Configuration (Matching your successful Colab notebook)
MODEL_NAME = "gemini-2.5-flash-lite" 
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
INDEX_DIR = "faiss_index"
RELEVANCE_THRESHOLD = 1.5
TOP_K_CHUNKS = 3

SYSTEM_PROMPT = """You are a highly accurate, company-specific AI assistant for Mysoft Heaven (BD) Ltd.

STRICT RULES — YOU MUST FOLLOW THESE AT ALL TIMES:
1. Answer ONLY using the information provided in the CONTEXT section below.
2. Do NOT use any external knowledge, training data, or general information.
3. If the answer is NOT found in the provided context, respond EXACTLY with:
   "I'm sorry, I can only answer questions related to Mysoft Heaven (BD) Ltd. based on the provided company documents. This question falls outside the available information."
4. Do NOT make up, infer, or guess any facts not explicitly stated in the context.
5. Be concise, professional, and helpful within these constraints."""

# 3. Load the AI Models and Local Database
print("Loading embedding model (this takes a few seconds)...")
embedder = SentenceTransformer(EMBEDDING_MODEL)

print("Loading local FAISS database...")
try:
    index = faiss.read_index(f"{INDEX_DIR}/index.faiss")
    chunks = np.load(f"{INDEX_DIR}/chunks.npy", allow_pickle=True)
except Exception as e:
    raise FileNotFoundError(f"Could not find the database! Did you run 'python src/ingest.py' first? Error: {e}")

# 4. Helper Functions
def retrieve_context(query: str):
    """Searches the FAISS database for text matching the user's question."""
    query_embedding = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding.astype(np.float32), TOP_K_CHUNKS)
    
    best_distance = float(distances[0][0])
    retrieved_chunks = [chunks[i] for i in indices[0] if i < len(chunks)]
    context = "\n\n---\n\n".join(retrieved_chunks)
    
    return context, best_distance

def format_history(history: list) -> str:
    """Safely converts Gradio's modern dictionary chat history into a text format."""
    if not history:
        return ""
    lines = ["\n\nPREVIOUS CONVERSATION (for context only):"]
    for msg in history:
        if isinstance(msg, dict):
            role = "User" if msg.get("role") == "user" else "Assistant"
            content = msg.get("content", "")
            lines.append(f"{role}: {content}")
        elif isinstance(msg, (list, tuple)) and len(msg) == 2:
            lines.append(f"User: {msg[0]}\nAssistant: {msg[1]}")
    return "\n".join(lines)

def rag_chat(message: str, history: list):
    """The main pipeline: Retrieve data -> Build Prompt -> Generate Reply."""
    start_time = time.time()
    
    # Step A: Get relevant info from the database
    context, best_distance = retrieve_context(message)
    
    # Step B: Fallback if question is unrelated (Prevents Hallucinations)
    if best_distance > RELEVANCE_THRESHOLD:
        confidence_note = f"\n\n_(Confidence: LOW — query distance {best_distance:.2f} exceeds threshold {RELEVANCE_THRESHOLD})_"
        return (
            "I'm sorry, I can only answer questions related to Mysoft Heaven (BD) Ltd. "
            "based on the provided company documents. This question falls outside the available information."
            + confidence_note
        )
        
    # Step C: Format history for memory
    conversation_history = format_history(history)
    
    # Step D: Build the strict prompt
    full_prompt = f"{SYSTEM_PROMPT}\n\nCONTEXT (from Mysoft Heaven company documents):\n{context}\n{conversation_history}\n\nCURRENT QUESTION: {message}\n\nANSWER:"
    
    # Step E: Get reply from Google Gemini
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=full_prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=512,
            )
        )
        answer = response.text
    except Exception as e:
        answer = f"Error communicating with AI: {str(e)}"
        
    # Step F: Append confidence indicator
    elapsed = round(time.time() - start_time, 2)
    if best_distance < 0.8:
        confidence_label = "HIGH"
    elif best_distance < 1.2:
        confidence_label = "MEDIUM"
    else:
        confidence_label = "LOW"
        
    confidence_note = f"\n\n_(Confidence: {confidence_label} | Distance: {best_distance:.2f} | Response time: {elapsed}s)_"
    
    return answer + confidence_note

# 5. Launch the Web UI
example_questions = [
    ["What is Mysoft Heaven (BD) Ltd.?"],
    ["What products or services does Mysoft Heaven offer?"],
    ["Who are the key clients or partners?"],
    ["What is the capital of France?"] 
]

with gr.Blocks(title="Mysoft Heaven AI Assistant") as demo:
    gr.Markdown("# Mysoft Heaven (BD) Ltd. — AI Assistant\n**Powered by RAG**\n\nThis chatbot answers questions **strictly** based on company documents.")
    
    chatbot_component = gr.Chatbot(height=450, placeholder="Ask me anything about Mysoft Heaven...", show_label=False, type="messages")
    
    gr.ChatInterface(
        fn=rag_chat,
        type="messages",
        chatbot=chatbot_component,
        textbox=gr.Textbox(placeholder="Type your question here...", container=False, scale=7),
        examples=example_questions,
    )

if __name__ == "__main__":
    print("Starting the Chatbot Server...")
    demo.launch(server_name="127.0.0.1", server_port=7860)