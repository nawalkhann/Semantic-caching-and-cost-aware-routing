"""Semantic caching and cost-aware model routing for RAG services."""

from .cache import ExactMatchCache, KeyTermExtractor, QuerySignature, SemanticCache
from .embeddings import HashingEmbedder
from .llm import ModelSpec, OpenAICompatibleLLM, SimulatedLLM
from .metrics import MetricsStore, RequestRecord, summarize
from .router import ComplexityRouter
from .service import Answer, RAGService

__all__ = ["Answer", "ComplexityRouter", "ExactMatchCache", "HashingEmbedder",
           "KeyTermExtractor", "MetricsStore", "ModelSpec", "OpenAICompatibleLLM",
           "QuerySignature", "RAGService", "RequestRecord", "SemanticCache", "SimulatedLLM",
           "summarize"]

__version__ = "0.1.0"
