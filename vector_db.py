from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct


class QdrantStorage:
    def __init__(self, url="http://localhost:6333", collection="docs", dim=3072):
        self.client = QdrantClient(url=url, timeout=30)
        self.collection = collection
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

    def upsert(self, ids, vectors, payloads):
        points = [PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i]) for i in range(len(ids))]
        self.client.upsert(self.collection, points=points)

    def search(self, query_vector, top_k: int = 5):
        results = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            with_payload=True,
            limit=top_k
        ).points
        contexts = []
        sources = set()

        for r in results:
            payload = getattr(r, "payload", None) or {}
            text = payload.get("text", "")
            source = payload.get("source", "")
            if text:
                contexts.append(text)
                sources.add(source)

        return {"contexts": contexts, "sources": list(sources)}

    def reset_collection(self):
        """Delete and recreate the collection to clear all data."""
        try:
            # Delete the existing collection
            if self.client.collection_exists(self.collection):
                self.client.delete_collection(self.collection)

            # Recreate the collection
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=3072, distance=Distance.COSINE),
            )
            return True
        except Exception as e:
            print(f"Error resetting collection: {e}")
            return False

    def get_collection_count(self):
        """Get the number of vectors in the collection."""
        try:
            if not self.client.collection_exists(self.collection):
                return 0
            count = self.client.count(collection_name=self.collection)
            return count.count
        except Exception as e:
            print(f"Error getting collection count: {e}")
            return 0