"""
Retriever Agent - Queries Chroma for relevant forecast vectors.

This agent specializes in semantic search and retrieval of forecast data,
providing high-quality, relevant documents for downstream analysis.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from .base_agent import BaseAgent, AgentInput, AgentOutput
from core.chroma_client import get_chroma_client, RetrievalResult
import json

try:
    from rag_chatbot.llm_reasoner import LLMReasoner
    HAS_LLM = True
except Exception:
    LLMReasoner = None
    HAS_LLM = False
from core.logger import LogLevel


class RetrieverAgent(BaseAgent):
    """
    Specialized agent for data retrieval from Chroma vector store.
    
    Responsibilities:
    - Perform semantic search over forecast data
    - Rank and filter retrieved documents
    - Extract relevant metadata and metrics
    - Validate data quality and relevance
    - Optimize retrieval parameters based on query
    """
    
    def __init__(self):
        """Initialize retriever agent."""
        super().__init__(
            name="RetrieverAgent",
            description="Queries Chroma for relevant forecast vectors"
        )
        self.chroma_client = None
    
    async def reason(self, agent_input: AgentInput) -> AgentOutput:
        """
        Retrieve relevant forecast data from vector store.
        
        Args:
            agent_input: Input containing query and context
            
        Returns:
            AgentOutput with retrieved documents and metadata
        """
        reasoning_steps = []
        session_id = agent_input.session_id
        query = agent_input.query
        
        try:
            # Step 1: Optionally use LLM to refine query for better retrieval
            await self._log_reasoning_step(session_id, "Refining query for retrieval (LLM)")
            refined_query = query
            if HAS_LLM:
                try:
                    llm = LLMReasoner()
                    if llm.is_available():
                        system = "You are a query expansion assistant for business forecasting retrieval."
                        human = f"User query: '{query}'\n\nProvide 3 short refined or synonym queries (comma-separated) that preserve intent and improve retrieval recall for forecasting/stability/business analysis." 
                        resp = llm.invoke_with_prompts(system, human)
                        # Try to parse simple comma separated suggestions
                        suggestions = []
                        for part in resp.replace('\n', ',').split(','):
                            s = part.strip()
                            if s:
                                suggestions.append(s.strip('"'))
                        if suggestions:
                            # use the first suggestion appended to original query to expand
                            refined_query = query + " " + suggestions[0]
                            await self._log_reasoning_step(session_id, f"Using refined query: {refined_query}")
                except Exception:
                    # If LLM fails, keep original query
                    refined_query = query

            # Step 2: Initialize Chroma client
            await self._log_reasoning_step(session_id, "Initializing Chroma client")
            if not await self._initialize_chroma_client():
                raise Exception("Failed to initialize Chroma client")
            reasoning_steps.append("Chroma client initialized")
            
            # Step 2: Analyze query for retrieval optimization
            await self._log_reasoning_step(session_id, "Analyzing query for retrieval optimization")
            retrieval_params = await self._analyze_query_for_retrieval(query)
            reasoning_steps.append(f"Retrieval parameters: top_k={retrieval_params['top_k']}")
            
            # Step 3: Perform semantic search (using refined_query if available)
            await self._log_reasoning_step(session_id, "Performing semantic search")
            retrieval_result = await self._perform_semantic_search(refined_query, retrieval_params)
            reasoning_steps.append(f"Retrieved {len(retrieval_result.documents)} documents")
            
            # Step 4: Process and rank documents
            await self._log_reasoning_step(session_id, "Processing and ranking documents")
            processed_docs = await self._process_retrieved_documents(retrieval_result)
            reasoning_steps.append(f"Processed {len(processed_docs)} documents")
            
            # Step 5: Extract product information
            await self._log_reasoning_step(session_id, "Extracting product information")
            product_info = await self._extract_product_information(processed_docs)
            reasoning_steps.append(f"Extracted info for {len(product_info['products'])} products")
            
            # Step 6: Validate data quality
            await self._log_reasoning_step(session_id, "Validating data quality")
            quality_assessment = await self._assess_data_quality(processed_docs, product_info)
            reasoning_steps.append(f"Data quality: {quality_assessment['overall_quality']}")
            
            # Calculate confidence based on retrieval quality and data completeness
            confidence = self._calculate_retrieval_confidence(retrieval_result, quality_assessment)
            
            return await self._create_output(
                success=True,
                data={
                    "retrieved_documents": processed_docs,
                    "product_information": product_info,
                    "quality_assessment": quality_assessment,
                    "retrieval_metadata": {
                        "query": query,
                        "total_documents": len(processed_docs),
                        "retrieval_time": retrieval_result.retrieval_time,
                        "search_parameters": retrieval_params
                    },
                    "search_quality": {
                        "relevance_score": quality_assessment["relevance_score"],
                        "coverage_score": quality_assessment["coverage_score"],
                        "diversity_score": quality_assessment["diversity_score"]
                    }
                },
                reasoning_steps=reasoning_steps,
                confidence=confidence,
                execution_time_ms=0.0
            )
            
        except Exception as e:
            await self._log_reasoning_step(
                session_id,
                f"Error during retrieval: {str(e)}",
                success=False,
                error=str(e)
            )
            
            return await self._create_output(
                success=False,
                data={"retrieved_documents": [], "error": str(e)},
                reasoning_steps=[f"Error: {str(e)}"],
                confidence=0.0,
                execution_time_ms=0.0,
                error=str(e)
            )
    
    async def _initialize_chroma_client(self) -> bool:
        """Initialize Chroma client if not already done."""
        if not self.chroma_client:
            self.chroma_client = await get_chroma_client()
            return self.chroma_client.is_initialized()
        return True
    
    async def _analyze_query_for_retrieval(self, query: str) -> Dict[str, Any]:
        """
        Analyze query to determine optimal retrieval parameters.
        
        Args:
            query: User query
            
        Returns:
            Retrieval parameters dictionary
        """
        query_lower = query.lower()
        
        # Determine optimal top_k based on query complexity
        if any(word in query_lower for word in ["all", "complete", "comprehensive", "everything"]):
            top_k = 20
        elif any(word in query_lower for word in ["top", "best", "worst", "few"]):
            top_k = 5
        elif len(query.split()) > 10:
            top_k = 15
        else:
            top_k = 10
        
        # Determine if we need specific product filtering
        product_filters = {}
        if any(word in query_lower for word in ["stable", "stability"]):
            product_filters["trend_type"] = "stable"
        elif any(word in query_lower for word in ["uptrend", "growing"]):
            product_filters["trend_type"] = "uptrend"
        elif any(word in query_lower for word in ["downtrend", "declining"]):
            product_filters["trend_type"] = "downtrend"
        
        return {
            "top_k": top_k,
            "filters": product_filters,
            "search_type": "semantic",
            "similarity_threshold": 0.7
        }
    
    async def _perform_semantic_search(
        self, 
        query: str, 
        retrieval_params: Dict[str, Any]
    ) -> RetrievalResult:
        """
        Perform semantic search using Chroma.
        
        Args:
            query: Search query
            retrieval_params: Retrieval parameters
            
        Returns:
            RetrievalResult with documents and scores
        """
        return await self.chroma_client.search(
            query=query,
            top_k=retrieval_params["top_k"],
            filter_metadata=retrieval_params.get("filters")
        )
    
    async def _process_retrieved_documents(self, retrieval_result: RetrievalResult) -> List[Dict[str, Any]]:
        """
        Process and enhance retrieved documents.
        
        Args:
            retrieval_result: Raw retrieval result
            
        Returns:
            List of processed documents
        """
        processed_docs = []
        
        for i, doc in enumerate(retrieval_result.documents):
            # Extract and structure document data
            processed_doc = {
                "id": doc.get("id", f"doc_{i}"),
                "content": doc.get("content", ""),
                "metadata": doc.get("metadata", {}),
                "relevance_score": retrieval_result.scores[i] if i < len(retrieval_result.scores) else 1.0,
                "retrieval_rank": i + 1
            }
            
            # Extract key metrics from metadata
            metadata = doc.get("metadata", {})
            if "ref_article" in metadata:
                processed_doc["product_code"] = metadata["ref_article"]
            
            if "avg_forecast" in metadata:
                processed_doc["forecast_value"] = float(metadata["avg_forecast"])
            
            if "trend_pct" in metadata:
                processed_doc["trend_percentage"] = float(metadata["trend_pct"])
            
            if "trend_label" in metadata:
                processed_doc["trend_direction"] = metadata["trend_label"]
            
            if "data_points" in metadata:
                processed_doc["data_reliability"] = int(metadata["data_points"])
            
            processed_docs.append(processed_doc)
        
        return processed_docs

    async def _llm_rerank_documents(self, session_id: str, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Ask the LLM to score and prioritize documents for business relevance.
        Expected LLM output format (lines): id|score(0-1)|priority(high|medium|low)|short justification
        """
        if not HAS_LLM:
            return docs

        try:
            llm = LLMReasoner()
            if not llm.is_available():
                return docs

            # Build human prompt with short doc summaries
            items = []
            for d in docs:
                mid = d.get("id")
                meta = d.get("metadata", {})
                product = meta.get("ref_article", "")
                avg = meta.get("avg_forecast", "N/A")
                trend = meta.get("trend_pct", "N/A")
                items.append(f"{mid}: product={product}; avg_forecast={avg}; trend_pct={trend}")

            human = (
                "You are a business analyst. For each document below, assign a relevance score (0.0-1.0) "
                "for the user's business intent and a business priority (high/medium/low).\n\n"
                "Documents:\n" + "\n".join(items) + "\n\n"
                "Output one line per document in the format: id|score|priority|justification"
            )
            system = "You are an expert business analyst who scores retrieved forecast documents for relevance/prioritization."
            resp = llm.invoke_with_prompts(system, human)

            # Parse lines
            rankings = {}
            for line in resp.splitlines():
                parts = [p.strip() for p in line.split('|')]
                if len(parts) >= 3:
                    doc_id = parts[0]
                    try:
                        score = float(parts[1])
                    except Exception:
                        score = 0.0
                    priority = parts[2] if len(parts) > 2 else 'medium'
                    justification = parts[3] if len(parts) > 3 else ''
                    rankings[doc_id] = {'score': score, 'priority': priority, 'justification': justification}

            # Attach ranking info and sort
            for d in docs:
                r = rankings.get(d.get('id'))
                if r:
                    d['llm_relevance'] = r['score']
                    d['business_priority'] = r['priority']
                    d['llm_justification'] = r['justification']
                else:
                    d['llm_relevance'] = d.get('relevance_score', 0.0)
                    d['business_priority'] = 'medium'

            docs_sorted = sorted(docs, key=lambda x: x.get('llm_relevance', 0.0), reverse=True)
            return docs_sorted

        except Exception:
            return docs
    
    async def _extract_product_information(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract and aggregate product information from documents.
        
        Args:
            documents: Processed documents
            
        Returns:
            Product information dictionary
        """
        products = {}
        product_metrics = []
        
        for doc in documents:
            if "product_code" in doc:
                product_code = doc["product_code"]
                
                if product_code not in products:
                    products[product_code] = {
                        "product_code": product_code,
                        "documents": [],
                        "metrics": {},
                        "relevance_scores": []
                    }
                
                products[product_code]["documents"].append(doc)
                products[product_code]["relevance_scores"].append(doc.get("relevance_score", 1.0))
                
                # Aggregate metrics
                if "forecast_value" in doc:
                    products[product_code]["metrics"]["avg_forecast"] = doc["forecast_value"]
                
                if "trend_percentage" in doc:
                    products[product_code]["metrics"]["trend_pct"] = doc["trend_percentage"]
                
                if "trend_direction" in doc:
                    products[product_code]["metrics"]["trend_label"] = doc["trend_direction"]
                
                if "data_reliability" in doc:
                    products[product_code]["metrics"]["data_points"] = doc["data_reliability"]
        
        # Calculate aggregated metrics
        for product_code, product_data in products.items():
            if product_data["relevance_scores"]:
                product_data["avg_relevance"] = sum(product_data["relevance_scores"]) / len(product_data["relevance_scores"])
            else:
                product_data["avg_relevance"] = 0.0
        
        return {
            "products": products,
            "total_products": len(products),
            "product_codes": list(products.keys()),
            "metrics_summary": self._calculate_metrics_summary(products)
        }
    
    async def _assess_data_quality(
        self, 
        documents: List[Dict[str, Any]], 
        product_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Assess the quality of retrieved data.
        
        Args:
            documents: Retrieved documents
            product_info: Product information
            
        Returns:
            Quality assessment dictionary
        """
        if not documents:
            return {
                "overall_quality": "poor",
                "relevance_score": 0.0,
                "coverage_score": 0.0,
                "diversity_score": 0.0,
                "issues": ["No documents retrieved"]
            }
        
        # Calculate relevance score
        relevance_scores = [doc.get("relevance_score", 0.0) for doc in documents]
        avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0
        
        # Calculate coverage score (how many products are covered)
        total_products = product_info.get("total_products", 0)
        coverage_score = min(1.0, total_products / 10)  # Normalize to 10 products
        
        # Calculate diversity score (how diverse are the products)
        product_codes = product_info.get("product_codes", [])
        diversity_score = min(1.0, len(set(product_codes)) / 5)  # Normalize to 5 unique products
        
        # Determine overall quality
        overall_score = (avg_relevance * 0.4 + coverage_score * 0.3 + diversity_score * 0.3)
        
        if overall_score >= 0.8:
            overall_quality = "excellent"
        elif overall_score >= 0.6:
            overall_quality = "good"
        elif overall_score >= 0.4:
            overall_quality = "fair"
        else:
            overall_quality = "poor"
        
        # Identify issues
        issues = []
        if avg_relevance < 0.5:
            issues.append("Low relevance scores")
        if total_products < 3:
            issues.append("Insufficient product coverage")
        if len(set(product_codes)) < 2:
            issues.append("Low product diversity")
        
        return {
            "overall_quality": overall_quality,
            "relevance_score": avg_relevance,
            "coverage_score": coverage_score,
            "diversity_score": diversity_score,
            "overall_score": overall_score,
            "issues": issues,
            "document_count": len(documents),
            "product_count": total_products
        }
    
    def _calculate_metrics_summary(self, products: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate summary metrics for all products."""
        if not products:
            return {}
        
        forecasts = []
        trends = []
        data_points = []
        
        for product_data in products.values():
            metrics = product_data.get("metrics", {})
            
            if "avg_forecast" in metrics:
                forecasts.append(metrics["avg_forecast"])
            
            if "trend_pct" in metrics:
                trends.append(metrics["trend_pct"])
            
            if "data_points" in metrics:
                data_points.append(metrics["data_points"])
        
        summary = {}
        
        if forecasts:
            summary["avg_forecast_mean"] = sum(forecasts) / len(forecasts)
            summary["avg_forecast_min"] = min(forecasts)
            summary["avg_forecast_max"] = max(forecasts)
        
        if trends:
            summary["trend_pct_mean"] = sum(trends) / len(trends)
            summary["trend_pct_min"] = min(trends)
            summary["trend_pct_max"] = max(trends)
        
        if data_points:
            summary["data_points_mean"] = sum(data_points) / len(data_points)
            summary["data_points_min"] = min(data_points)
            summary["data_points_max"] = max(data_points)
        
        return summary
    
    def _calculate_retrieval_confidence(
        self, 
        retrieval_result: RetrievalResult, 
        quality_assessment: Dict[str, Any]
    ) -> float:
        """Calculate confidence in retrieval results."""
        if not retrieval_result.documents:
            return 0.0
        
        # Base confidence from quality assessment
        quality_score = quality_assessment.get("overall_score", 0.0)
        
        # Adjust for retrieval time (faster is better)
        time_factor = 1.0 if retrieval_result.retrieval_time < 2.0 else 0.8
        
        # Adjust for document count (more documents can be better, up to a point)
        doc_count = len(retrieval_result.documents)
        count_factor = min(1.0, doc_count / 10) if doc_count <= 20 else 0.9
        
        return min(1.0, quality_score * time_factor * count_factor)
    
    def get_capabilities(self) -> List[str]:
        """Get list of retriever agent capabilities."""
        return [
            "Semantic search over forecast data",
            "Document ranking and filtering",
            "Product information extraction",
            "Data quality assessment",
            "Retrieval optimization",
            "Metadata analysis"
        ]
