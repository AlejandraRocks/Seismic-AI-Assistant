import os
import time
import subprocess
import pickle
import gradio as gr
import requests
from huggingface_hub import InferenceClient
from langchain_community.document_loaders import TextLoader, PyPDFLoader, CSVLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools import Tool

# Verify Hugging Face Pro account
API_KEY = "YOUR API-KEY"

# Initialize Hugging Face Client
client = InferenceClient(model="mistralai/Mistral-7B-Instruct-v0.3", token=API_KEY)

embeddings = HuggingFaceInferenceAPIEmbeddings(
    model_name="thenlper/gte-small",
    api_key=API_KEY
)

# Checkpoint configuration to prevent data loss
CHECKPOINT_INTERVAL = 500  # Save embeddings every 500 documents
EMBEDDINGS_FILE = "embeddings.pkl"

# Generate embeddings in batches

def generate_embeddings(texts):
    """Generates embeddings for multiple texts in a single API call."""
    try:
        print(f"Processing {len(texts)} fragments in a single request...")
        embeddings_list = embeddings.embed_documents(texts)
        print(f"Embeddings generated for {len(texts)} fragments.")
        return embeddings_list
    except Exception as e:
        print(f"Error in API: {e}")
        return None

# Split long texts into smaller fragments

def split_text(text, max_length=300):
    """Splits a long text into smaller fragments to optimize embedding time."""
    return [text[i:i+max_length] for i in range(0, len(text), max_length)]

# Load documents and generate embeddings

def load_documents():
    documents = []
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATASET_DIR = os.path.join(BASE_DIR, "dataset")

    if not os.path.exists(DATASET_DIR):
        raise ValueError("The 'dataset' directory does not exist. Add documents before proceeding.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=30)

    for file in os.listdir(DATASET_DIR):
        path = os.path.join(DATASET_DIR, file)
        
        if file.endswith(".txt"):
            loader = TextLoader(path, encoding="utf-8")
        elif file.endswith(".pdf"):
            loader = PyPDFLoader(path)
        elif file.endswith(".csv"):
            loader = CSVLoader(path)
        else:
            continue  # Ignore unsupported file types
        
        docs = loader.load()
        documents.extend(text_splitter.split_documents(docs))

    if not documents:
        raise ValueError("The 'dataset' directory is empty. Add documents before proceeding.")

    return documents

# Load documents and index them in ChromaDB
print("Loading documents from the dataset folder...")
documents = load_documents()
print(f"{len(documents)} documents loaded.")

# Load previous embeddings if available
if os.path.exists(EMBEDDINGS_FILE):
    with open(EMBEDDINGS_FILE, "rb") as f:
        texts_with_embeddings, embeddings_list = pickle.load(f)
    print(f"Loaded {len(texts_with_embeddings)} previous embeddings.")
else:
    texts_with_embeddings, embeddings_list = [], []

# Generate embeddings in batches
BATCH_SIZE = 5

for i in range(0, len(documents), BATCH_SIZE):
    batch_docs = [doc.page_content for doc in documents[i:i+BATCH_SIZE]]
    embeddings_batch = generate_embeddings(batch_docs)
    
    if embeddings_batch:
        texts_with_embeddings.extend(batch_docs)
        embeddings_list.extend(embeddings_batch)
    
    if len(texts_with_embeddings) % CHECKPOINT_INTERVAL == 0:
        with open(EMBEDDINGS_FILE, "wb") as f:
            pickle.dump((texts_with_embeddings, embeddings_list), f)
        print(f"Saved {len(texts_with_embeddings)} embeddings in {EMBEDDINGS_FILE}.")

    time.sleep(1)

if texts_with_embeddings:
    db = Chroma.from_texts(texts=texts_with_embeddings, embedding=embeddings)
    print("Database successfully created.")
else:
    print("No valid embeddings were generated. The API did not respond.")

# Function to search information in the database

def search_info(query):
    if not texts_with_embeddings:
        return "No indexed documents found. Add documents and restart the chatbot."
    
    results = db.similarity_search(query)
    return results[0].page_content if results else "No relevant information found."

# Function to execute commands in Seismic Unix

def execute_su(command: str):
    """Executes a Seismic Unix command and returns the output."""
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return result.stdout if result.returncode == 0 else result.stderr

# Define tools in LangChain
tools = [
    Tool(name="SeismicUnix", func=execute_su, description="Executes commands in Seismic Unix."),
    Tool(name="SearchDocs", func=search_info, description="Search for information in seismic documents.")
]

# 🔹 Function to suggest Seismic Unix commands without executing them
def suggest_su_command(prompt: str):
    if "high-pass filter" in prompt.lower() or "paso alto" in prompt.lower():
        return "To apply a high-pass filter at 10Hz in Seismic Unix, use: `sufilter f=10`"
    elif "low-pass filter" in prompt.lower() or "paso bajo" in prompt.lower():
        return "To apply a low-pass filter at 5Hz in Seismic Unix, use: `sufilter f=5`"
    else:
        return "❌ Unable to determine the appropriate Seismic Unix command."


# Main chatbot function

def chat(prompt):
    """Determines whether to respond using LLM, search in documents, or suggest a Seismic Unix command."""
    prompt_lower = prompt.lower()
    
    # Search in seismic documents
    if "search" in prompt_lower or "document" in prompt_lower:
        return search_info(prompt)
    
    # Suggest Seismic Unix commands without executing them
    elif "seismic unix" in prompt_lower or "filter" in prompt_lower or "filtra" in prompt_lower:
        if "high-pass" in prompt_lower or "paso alto" in prompt_lower:
            return "To apply a high-pass filter at 10Hz in Seismic Unix, use: sufilter f=10"
        elif "low-pass" in prompt_lower or "paso bajo" in prompt_lower:
            return "To apply a low-pass filter at 5Hz in Seismic Unix, use: sufilter f=5"
        else:
            return "Unable to determine the correct Seismic Unix command."
    
    # Use the LLM for general responses
    else:
        response = client.chat_completion(messages=[{"role": "user", "content": prompt}])
        return response["choices"][0]["message"]["content"]

# Create Gradio interface
interface = gr.Interface(
    fn=chat,
    inputs=gr.Textbox(label="Enter your message"),
    outputs=gr.Textbox(label="Chatbot Response"),
    title="Seismic AI Assistant: Automating Seismic Data Processing and Interpretation",
    description="Ask questions about seismic processing ."
)

# Run the application
if __name__ == "__main__":
    interface.launch(share=True)
