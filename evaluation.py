import os
import time
import pickle
import re
from langchain_community.document_loaders import TextLoader, PyPDFLoader, CSVLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from huggingface_hub import InferenceClient
import evaluate

API_KEY = "API-KEY"

client = InferenceClient(model="mistralai/Mistral-7B-Instruct-v0.3", token=API_KEY)
embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    huggingfacehub_api_token=API_KEY
)

bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")

def clean(text):
    return re.sub(r"[^\w\s]", "", text.lower()).strip()

def load_documents():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))# For .py 
    #BASE_DIR = os.getcwd()

    DATASET_DIR = os.path.join(BASE_DIR, "dataset") 
    documents = []

    if not os.path.exists(DATASET_DIR):
        raise FileNotFoundError("Missing 'dataset' directory")

    splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=30)

    for file in os.listdir(DATASET_DIR):
        path = os.path.join(DATASET_DIR, file)

        if file.endswith(".txt"):
            loader = TextLoader(path, encoding="utf-8")
        elif file.endswith(".pdf"):
            loader = PyPDFLoader(path)
        elif file.endswith(".csv"):
            loader = CSVLoader(path)
        else:
            continue

        docs = loader.load()
        for doc in splitter.split_documents(docs):
            doc.metadata["source_file"] = file
            documents.append(doc)

    return documents

def build_chroma(documents):
    texts = [doc.page_content for doc in documents]
    embs = embeddings.embed_documents(texts)
    return Chroma.from_texts(texts, embedding=embeddings, metadatas=[doc.metadata for doc in documents])

def answer_question(query, db):
    results = db.similarity_search(query, k=1)
    if results:
        return results[0].page_content.strip(), results[0].metadata.get("source_file", "Unknown")
    else:
        return "No relevant information found.", "N/A"

def evaluate_response(question, expected_answer, bot_answer):
    pred = clean(bot_answer)
    ref = clean(expected_answer)

    contains_expected = expected_answer.lower() in bot_answer.lower()
    bleu_score = bleu.compute(predictions=[pred], references=[[ref]])["bleu"]
    rouge_score = rouge.compute(predictions=[pred], references=[ref])["rougeL"]

    return contains_expected, bleu_score, rouge_score

# Load and embed documents
print("🔄 Loading documents...")
documents = load_documents()
db = build_chroma(documents)
print("✅ Documents loaded and embedded.")

# Define test questions and answers
test_cases = [
    {
        "question": "What is a stacked trace?",
        "expected": "A stacked trace is a single trace formed by summing or stacking together traces of each CMP gather. It is often used to approximate a zero-offset trace."
    },
    {
        "question": "How do I apply a high-pass filter in Seismic Unix?",
        "expected": "To apply a high-pass filter at 10Hz in Seismic Unix, use: sufilter f=10"
    }
]

print("\n🔍 Evaluation Results\n" + "-" * 40)

total = len(test_cases)
match_count = 0
total_bleu = 0
total_rouge = 0

for i, case in enumerate(test_cases, 1):
    print(f"\n📌 Question {i}: {case['question']}")
    expected = case["expected"]
    answer, source = answer_question(case["question"], db)
    print(f"Expected: {expected}")
    print(f"Bot replied: {answer}")
    print(f"Source file: {source}")

    contains, bleu_score, rouge_score = evaluate_response(case["question"], expected, answer)

    print(f"Contains expected phrase? {'✅' if contains else '❌'}")
    print(f"BLEU score: {bleu_score:.2f}")
    print(f"ROUGE-L score: {rouge_score:.2f}")
    print(f"📎 This answer was found in the document: {source}")

    match_count += contains
    total_bleu += bleu_score
    total_rouge += rouge_score

# Summary
print("\n✅ Summary")
print(f"Matched answers: {match_count}/{total} ({(match_count/total)*100:.2f}%)")
print(f"Average BLEU: {total_bleu/total:.2f}")
print(f"Average ROUGE-L: {total_rouge/total:.2f}")
