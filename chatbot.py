import os
import time
import subprocess
import pickle
import gradio as gr
from huggingface_hub import InferenceClient
from langchain_community.document_loaders import TextLoader, PyPDFLoader, CSVLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools import Tool
from langchain_community.document_loaders import UnstructuredWordDocumentLoader


API_KEY = "Your API-KEY"
client = InferenceClient(model="mistralai/Mistral-7B-Instruct-v0.3", token=API_KEY)
embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    huggingfacehub_api_token=API_KEY
)

CHECKPOINT_INTERVAL = 500
EMBEDDINGS_FILE = "embeddings.pkl"


def generate_embeddings(texts):
    try:
        print(f"Processing {len(texts)} fragments in a single request...")
        return embeddings.embed_documents(texts)
    except Exception as e:
        print(f"Error in API: {e}")
        return None


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
        elif file.endswith(".docx"):
            loader = UnstructuredWordDocumentLoader(path)
        else:
            continue

        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = file
        documents.extend(text_splitter.split_documents(docs))

    if not documents:
        raise ValueError("The 'dataset' directory is empty. Add documents before proceeding.")

    return documents


print("Loading documents from the dataset folder...")
documents = load_documents()
print(f"{len(documents)} documents loaded.")

if os.path.exists(EMBEDDINGS_FILE):
    with open(EMBEDDINGS_FILE, "rb") as f:
        texts_with_embeddings, embeddings_list = pickle.load(f)
    print(f"Loaded {len(texts_with_embeddings)} previous embeddings.")
else:
    texts_with_embeddings, embeddings_list = [], []

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
    db = Chroma.from_documents(documents=documents, embedding=embeddings)
    print("Database successfully created.")
else:
    print("No valid embeddings were generated. The API did not respond.")


def execute_su(command: str):
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return result.stdout if result.returncode == 0 else result.stderr


def chat(prompt):
    prompt_lower = prompt.lower()

    # if "seismic unix" in prompt_lower or "filter" in prompt_lower or "filtra" in prompt_lower:
    #     if "high-pass" in prompt_lower or "paso alto" in prompt_lower:
    #         return "To apply a high-pass filter at 10Hz in Seismic Unix, use: `sufilter f=10`"
    #     elif "low-pass" in prompt_lower or "paso bajo" in prompt_lower:
    #         return "To apply a low-pass filter at 5Hz in Seismic Unix, use: `sufilter f=5`"
    #     else:
    #         return "❌ Unable to determine the appropriate Seismic Unix command."

    if texts_with_embeddings:
        docs = db.similarity_search(prompt, k=3)
        if docs:
            context = "\n\n".join([doc.page_content for doc in docs])
            source = docs[0].metadata.get("source", "Unknown document")

            # Generamos SIEMPRE con contexto, aunque no sea perfecto
            system_prompt = (
                "You are a helpful assistant for seismic data interpretation. "
                "Use the CONTEXT below to answer the USER's question. "
                "If the context is not sufficient, you can enrich the answer with your own knowledge.\n\n"
                f"CONTEXT:\n{context}"
            )

            response = client.chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
            )
            answer = response["choices"][0]["message"]["content"]
            return f"{answer}\n\n📄 This answer was enriched using: `{source}`"

        else:
            fallback = client.chat_completion(
                messages=[
                    {"role": "system", "content": "You are a seismic assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            return f"{fallback['choices'][0]['message']['content']}\n\n🤖 This answer was generated by the AI model because no relevant documents were found."

    return "⚠️ No indexed documents found. Add documents and restart the chatbot."



interface = gr.Interface(
    fn=chat,
    inputs=gr.Textbox(label="Enter your message"),
    outputs=gr.Textbox(label="Chatbot Response"),
    title="Seismic AI Assistant: Automating Seismic Data Processing and Interpretation",
    description="Ask questions about seismic processing ."
)

if __name__ == "__main__":
    interface.launch(share=True)
