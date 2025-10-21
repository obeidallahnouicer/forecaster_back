"""
Chroma Vector Store Client - Encapsulates Chroma operations for multi-agent system.

This module provides a clean interface to ChromaDB operations, abstracting away
the complexity of vector retrieval and enabling seamless integration with the
multi-agent reasoning framework.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import asyncio
from dataclasses import dataclass

try:
    from langchain_community.vectorstores import Chroma
    from langchain.schema import Document
    HAS_CHROMA = True
except ImportError:
    try:
        from langchain.vectorstores import Chroma
        from langchain.schema import Document
        HAS_CHROMA = True
    except ImportError:
        Chroma = None
        Document = None
        HAS_CHROMA = False

from rag_chatbot import config, retriever

logger = logging.getLogger("core.chroma_client")


@dataclass
class RetrievalResult:
    """Structured result from vector retrieval."""
    documents: List[Dict[str, Any]]
    scores: List[float]
    metadata: Dict[str, Any]
    query: str
    retrieval_time: float


class ChromaClient:
    """
    Encapsulates Chroma vector store operations for multi-agent system.
    
    Provides:
    - Semantic search over forecast data
    - Structured retrieval results
    - Error handling and fallbacks
    - Performance monitoring
    """
    
    def __init__(self, collection_name: str = "forecast_analysis"):
        """
        Initialize Chroma client.
        
        Args:
            collection_name: Name of the Chroma collection to use
        """
        self.collection_name = collection_name
        self._vectorstore = None
        self._retriever = None
        self.logger = logging.getLogger(f"core.chroma_client.{collection_name}")
    
    async def initialize(self) -> bool:
        """
        Initialize the vector store connection.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if not HAS_CHROMA:
                self.logger.error("Chroma not available")
                return False
            
            # Get or build vectorstore
            self._vectorstore = retriever.get_vectorstore()
            if self._vectorstore is None:
                self.logger.error("Failed to initialize vectorstore")
                return False
            
            # Create retriever
            self._retriever = self._vectorstore.as_retriever(
                search_kwargs={"k": config.RETRIEVAL_K}
            )
            
            self.logger.info(f"Chroma client initialized with collection: {self.collection_name}")
            return True
            
        except Exception as e:
            self.logger.exception(f"Failed to initialize Chroma client: {e}")
            return False
    
    async def search(
        self, 
        query: str, 
        top_k: Optional[int] = None,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> RetrievalResult:
        """
        Perform semantic search over the vector store.
        
        Args:
            query: Search query
            top_k: Number of documents to retrieve (default: config.RETRIEVAL_K)
            filter_metadata: Optional metadata filters
            
        Returns:
            RetrievalResult with documents, scores, and metadata
        """
        if not self._retriever:
            await self.initialize()
        
        if not self._retriever:
            return RetrievalResult(
                documents=[],
                scores=[],
                metadata={"error": "Vectorstore not initialized"},
                query=query,
                retrieval_time=0.0
            )
        
        import time
        start_time = time.time()
        
        try:
            # Perform retrieval
            if top_k:
                # Temporarily update retriever k value
                original_k = self._retriever.search_kwargs.get("k", config.RETRIEVAL_K)
                self._retriever.search_kwargs["k"] = top_k
            
            # Get documents
            docs = self._retriever.get_relevant_documents(query)
            
            # Restore original k value
            if top_k:
                self._retriever.search_kwargs["k"] = original_k
            
            # Convert to structured format
            documents = []
            scores = []
            
            for doc in docs:
                doc_dict = {
                    "id": doc.metadata.get("source_id", f"doc_{len(documents)}"),
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                documents.append(doc_dict)
                
                # Extract similarity score if available
                score = getattr(doc, 'score', 1.0)  # Default score if not available
                scores.append(score)
            
            retrieval_time = time.time() - start_time
            
            self.logger.info(
                f"Retrieved {len(documents)} documents in {retrieval_time:.3f}s"
            )
            
            return RetrievalResult(
                documents=documents,
                scores=scores,
                metadata={
                    "collection": self.collection_name,
                    "query_length": len(query),
                    "retrieval_k": top_k or config.RETRIEVAL_K,
                    "success": True
                },
                query=query,
                retrieval_time=retrieval_time
            )
            
        except Exception as e:
            self.logger.exception(f"Search failed: {e}")
            return RetrievalResult(
                documents=[],
                scores=[],
                metadata={"error": str(e), "success": False},
                query=query,
                retrieval_time=time.time() - start_time
            )
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store collection.
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            if not self._vectorstore:
                await self.initialize()
            
            if not self._vectorstore:
                return {"error": "Vectorstore not initialized"}
            
            # Get collection count
            count = self._vectorstore._collection.count()
            
            return {
                "collection_name": self.collection_name,
                "document_count": count,
                "status": "active"
            }
            
        except Exception as e:
            self.logger.exception(f"Failed to get collection stats: {e}")
            return {"error": str(e), "status": "error"}
    
    def is_initialized(self) -> bool:
        """Check if the client is properly initialized."""
        return self._vectorstore is not None and self._retriever is not None


# Global client instance
_chroma_client: Optional[ChromaClient] = None


async def get_chroma_client() -> ChromaClient:
    """
    Get or create the global Chroma client instance.
    
    Returns:
        ChromaClient instance
    """
    global _chroma_client
    
    if _chroma_client is None:
        _chroma_client = ChromaClient()
        await _chroma_client.initialize()
    
    return _chroma_client


async def search_forecasts(
    query: str, 
    top_k: Optional[int] = None
) -> RetrievalResult:
    """
    Convenience function to search forecast data.
    
    Args:
        query: Search query
        top_k: Number of documents to retrieve
        
    Returns:
        RetrievalResult with forecast documents
    """
    client = await get_chroma_client()
    return await client.search(query, top_k)
