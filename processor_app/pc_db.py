import os
import asyncio
import json
import itertools
import re
import numpy as np
import chromadb
import uuid
import logging
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
class PC_Mistral:
    def __init__(self, embed_model):
        self.embedder = embed_model
        # Persistent local ChromaDB storage
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")

    def validate_chunk_for_sparse_embedding(self, text: str) -> bool:
        if not text or len(text.strip()) < 10: return False
        if not re.search(r"[a-zA-Z0-9]", text): return False
        if len(text.split()) < 2: return False
        return True

    def namespace_exists(self, namespace: str) -> bool:
        try:
            self.chroma_client.get_collection(name=self._sanitize_collection_name(namespace))
            return True
        except Exception:
            return False

    def _sanitize_collection_name(self, name: str) -> str:
        name = re.sub(r'[^a-zA-Z0-9._-]', '', name)
        name = re.sub(r'^[^a-zA-Z0-9]+', '', name)
        name = re.sub(r'[^a-zA-Z0-9]+$', '', name)
        if len(name) < 3: name = f"col_{name}"
        return name

    def chunks(self, iterable, batch_size=50):
        it = iter(iterable)
        while True:
            chunk = tuple(itertools.islice(it, batch_size))
            if not chunk: break
            yield chunk

    def _get_collection(self, namespace: str):
        safe_namespace = self._sanitize_collection_name(namespace)
        return self.chroma_client.get_or_create_collection(
            name=safe_namespace,
            metadata={"hnsw:space": "cosine"}
        )

    async def parallel_upsert_batch(self, index, namespace, batch, *args, **kwargs):
        collection = self._get_collection(namespace)
        docs = [b["chunk_text"] for b in batch]
        ids = [b["_id"] for b in batch]
        metas = [{k: v for k, v in b.items() if k not in ["_id", "chunk_text"]} for b in batch]
        
        embeddings = await asyncio.to_thread(
            self.embedder.embed_documents, [f"passage: {d}" for d in docs]
        )
        
        collection.add(documents=docs, embeddings=embeddings, metadatas=metas, ids=ids)
        return len(batch)

    async def chunk_upload_hybrid(self, filenames, dense_index, sparse_index, namespace):
        if isinstance(filenames, str): filenames = [filenames]

        async def process_file(filename):
            if not os.path.exists(filename):
                logging.error(f"File not found locally: {filename}")
                return {"filename": filename, "chunk_count": 0, "status": "not_found"}

            with open(filename, 'r', encoding='utf-8') as f:
                raw = f.read()
            
            data = json.loads(raw)

            documents = []
            if isinstance(data, dict) and "text" in data:
                documents = [Document(page_content=data["text"], metadata={"file": filename, "page": 1})]
            elif isinstance(data, list):
                documents = [
                    Document(page_content=str(d.get("page_content", "")), metadata=d.get("metadata", {}))
                    for d in data
                ]

            splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=350)
            records = []
            
            for doc_idx, doc in enumerate(documents):
                chunks = splitter.split_documents([doc])
                for chunk_idx, chunk in enumerate(chunks):
                    if self.validate_chunk_for_sparse_embedding(chunk.page_content):
                        chunk_id = f"{os.path.basename(filename)}_p{chunk.metadata.get('page','unknown')}_d{doc_idx}_c{chunk_idx}_{uuid.uuid4().hex[:8]}"
                        records.append({
                            "_id": chunk_id,
                            "chunk_text": chunk.page_content,
                            "page": chunk.metadata.get("page", "unknown"),
                            "file": os.path.basename(filename),
                            "type": "text"
                        })

            for batch in self.chunks(records):
                await self.parallel_upsert_batch(None, namespace, batch)

            return {"filename": filename, "chunk_count": len(records), "status": "success"}

        results = await asyncio.gather(*[process_file(f) for f in filenames])
        total_chunks = sum(r["chunk_count"] for r in results)
        
        return total_chunks

    def hybrid_search(self, *args, **kwargs):
        try:
            query = args[0] if len(args) > 0 else kwargs.get("query", kwargs.get("question", ""))
            namespace = args[3] if len(args) > 3 else kwargs.get("namespace", "")
            top_n = kwargs.get("top_n", 5)

            if not namespace:
                return []

            collection = self._get_collection(namespace)
            q_emb = self.embedder.embed_documents([f"query: {query}"])[0]

            results = collection.query(query_embeddings=[q_emb], n_results=top_n * 2)

            if not results or not results.get("documents") or len(results["documents"][0]) == 0:
                return []

            docs = results["documents"][0]
            metas = results["metadatas"][0]
            distances = results["distances"][0]

            try:
                dense_scores = 1 - np.array(distances)
                bm25 = BM25Okapi([d.split() for d in docs])
                sparse_scores = bm25.get_scores(query.split())
                scores = 0.8 * dense_scores + 0.2 * sparse_scores
                ranked = sorted(zip(docs, metas, scores), key=lambda x: x[2], reverse=True)[:top_n]
            except Exception:
                ranked = list(zip(docs, metas, [1.0]*len(docs)))[:top_n]

            return [
                {"page_content": d, "page": m.get("page", "unknown"), "file": m.get("file", "unknown")}
                for d, m, _ in ranked
            ]
        except Exception as e:
            logging.error(f"Error in hybrid_search: {str(e)}")
            return []

    async def hybrid_search_async(self, *args, **kwargs):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.hybrid_search(*args, **kwargs))

    def create_hybrid_indexes(self, dense_index_name, sparse_index_name, metric="cosine"):
        return dense_index_name, sparse_index_name

    def get_hybrid_retriever(self, namespace: str, k: int):
        return self.init_hybrid_retriever(dense_index=None, sparse_index=None, namespace=namespace, k=k)

    def init_hybrid_retriever(self, dense_index, sparse_index, namespace, k=20):
        class HybridRetriever:
            def __init__(self, pc, namespace, k):
                self.pc = pc
                self.namespace = namespace
                self.k = k

            def invoke(self, query, n=None):
                res = self.pc.hybrid_search(
                    query, None, None, self.namespace,
                    k_dense=self.k, k_sparse=self.k, top_n=n or self.k
                )
                return [Document(page_content=r["page_content"], metadata={"page": r["page"], "file": r["file"], "type": "text"}) for r in res]

            async def ainvoke(self, query, n=None):
                res = await self.pc.hybrid_search_async(
                    query, None, None, self.namespace,
                    k_dense=self.k, k_sparse=self.k, top_n=n or self.k
                )
                return [Document(page_content=r["page_content"], metadata={"page": r["page"], "file": r["file"], "type": "text"}) for r in res]

        return HybridRetriever(self, namespace, k)

    async def get_context_hybrid_async(self, question, retriever, top_n=25):
        docs = await retriever.ainvoke(question, top_n)
        if not docs:
            return []
        return [{"page_content": d.page_content, "page": d.metadata.get("page", "unknown"), "file": d.metadata.get("file", "unknown")} for d in docs]