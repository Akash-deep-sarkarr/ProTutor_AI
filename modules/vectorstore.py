# modules/vectorstore.py
import os
import pickle
import numpy as np
try:
    from sentence_transformers import SentenceTransformer
    HAVE_SENT_TRANSFORMERS = True
except Exception:
    SentenceTransformer = None
    HAVE_SENT_TRANSFORMERS = False

# Prefer faiss when available, but provide a numpy fallback so the
# app can run on systems without faiss installed.
try:
    import faiss
    USE_FAISS = True
except Exception:
    faiss = None
    USE_FAISS = False

if HAVE_SENT_TRANSFORMERS:
    embed_model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
else:
    # Lightweight fallback embedder (hashing trick). This is a deterministic,
    # tiny embedding used to allow the app to run without `sentence-transformers`.
    _FALLBACK_DIM = 768

    def _basic_embed_texts(texts):
        arr = np.zeros((len(texts), _FALLBACK_DIM), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in str(t).split():
                idx = abs(hash(w)) % _FALLBACK_DIM
                arr[i, idx] += 1.0
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-9, norms)
        return arr / norms

    class _BasicEmbedder:
        def encode(self, texts, convert_to_numpy=False):
            is_list = isinstance(texts, (list, tuple))
            items = list(texts) if is_list else [texts]
            arr = _basic_embed_texts(items)
            if not is_list and convert_to_numpy is False:
                # keep previous API shape for single string when convert_to_numpy=False
                return arr[0].tolist()
            return arr if convert_to_numpy else arr.tolist()

    embed_model = _BasicEmbedder()


def _ensure_embeddings_dir(index_path: str):
    d = os.path.dirname(index_path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def build_vectorstore(texts: list, index_path: str = "embeddings/vectordb"):
    """Build and persist embeddings. Use faiss index if available,
    otherwise save embeddings to disk for numpy-based search.
    """
    _ensure_embeddings_dir(index_path)

    # produce numpy embeddings
    embeddings = embed_model.encode(texts, convert_to_numpy=True)

    if USE_FAISS:
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        faiss.write_index(index, f"{index_path}.index")
    else:
        # save embeddings array for fallback search
        np.save(f"{index_path}_embeddings.npy", embeddings)

    with open(f"{index_path}_texts.pkl", "wb") as f:
        pickle.dump(texts, f)


def retrieve_relevant_chunks(query: str, k=3, index_path: str = "embeddings/vectordb", score_threshold: float = 0.2):
    """Retrieve relevant text chunks for a query.

    If faiss is present, use it; otherwise compute cosine similarity
    against stored embeddings and return top-k results.
    """
    texts_path = f"{index_path}_texts.pkl"
    if not os.path.exists(texts_path):
        return []

    with open(texts_path, "rb") as f:
        texts = pickle.load(f)

    query_embedding = embed_model.encode([query], convert_to_numpy=True)

    if USE_FAISS:
        try:
            index = faiss.read_index(f"{index_path}.index")
            distances, indices = index.search(np.array(query_embedding), k)

            relevant_chunks = []
            for distance, idx in zip(distances[0], indices[0]):
                if idx < 0 or idx >= len(texts):
                    continue
                if distance <= score_threshold:
                    relevant_chunks.append(texts[idx])

            return relevant_chunks
        except Exception:
            # if index not present or read fails, fall back to numpy
            pass

    # Numpy fallback: cosine similarity
    emb_path = f"{index_path}_embeddings.npy"
    if not os.path.exists(emb_path):
        return []

    embeddings = np.load(emb_path)

    # normalize
    emb_norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    safe_emb_norms = np.where(emb_norms == 0, 1e-9, emb_norms)
    emb_unit = embeddings / safe_emb_norms

    q = query_embedding[0]
    q_norm = np.linalg.norm(q)
    if q_norm == 0:
        return []
    q_unit = q / q_norm

    sims = np.dot(emb_unit, q_unit)
    topk_idx = np.argsort(sims)[-k:][::-1]

    relevant_chunks = []
    for idx in topk_idx:
        if sims[idx] >= score_threshold:
            relevant_chunks.append(texts[int(idx)])

    return relevant_chunks