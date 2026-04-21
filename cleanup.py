# cleanup.py
from app.retrieval.vector_store import get_client
from app.config import settings

client = get_client()
client.delete_collection(settings.qdrant_collection)
print("✅ Collection deleted")