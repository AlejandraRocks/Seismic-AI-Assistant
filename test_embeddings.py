from langchain_huggingface import HuggingFaceEndpointEmbeddings

embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    huggingfacehub_api_token="API-KEY"  # ← tu clave real
)

texts = ["Seismic data processing", "Velocity analysis in SU", "Migration techniques"]

response = embeddings.embed_documents(texts)

print("✅ Embeddings generated:")
print(response)
