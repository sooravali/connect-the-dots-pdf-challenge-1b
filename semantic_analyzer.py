"""
Semantic Analysis Module for Challenge 1B
Handles persona-driven document intelligence using sentence embeddings.
"""

import logging
import time
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Configure logging
logger = logging.getLogger(__name__)

# Global flag to track if sentence-transformers is available
SENTENCE_TRANSFORMERS_AVAILABLE = False
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
    logger.info("Sentence-transformers library loaded successfully")
except ImportError:
    logger.warning("Sentence-transformers not available, falling back to TF-IDF")


class SemanticAnalyzer:
    """
    Semantic analysis engine for persona-driven document intelligence.
    Uses sentence embeddings (preferred) or TF-IDF (fallback) for relevance scoring.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.vectorizer = None
        self.use_embeddings = SENTENCE_TRANSFORMERS_AVAILABLE
        
        if self.use_embeddings:
            self._load_embedding_model()
        else:
            self._setup_tfidf_fallback()
    
    def _load_embedding_model(self):
        """Load the sentence transformer model."""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            self.use_embeddings = False
            self._setup_tfidf_fallback()
    
    def _setup_tfidf_fallback(self):
        """Setup TF-IDF vectorizer as fallback."""
        logger.info("Setting up TF-IDF fallback for semantic analysis")
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95
        )
    
    def rank_sections(self, query: str, sections: List[Dict], 
                     top_k: int = 10) -> List[Tuple[Dict, float]]:
        """
        Rank document sections by relevance to the query.
        
        Args:
            query: Combined persona and job description
            sections: List of section dictionaries
            top_k: Number of top sections to return
            
        Returns:
            List of (section, score) tuples sorted by relevance
        """
        if not sections:
            return []
        
        if self.use_embeddings and self.model:
            return self._rank_with_embeddings(query, sections, top_k)
        else:
            return self._rank_with_tfidf(query, sections, top_k)
    
    def _rank_with_embeddings(self, query: str, sections: List[Dict], 
                            top_k: int) -> List[Tuple[Dict, float]]:
        """Rank sections using sentence embeddings."""
        try:
            # Encode query
            query_embedding = self.model.encode([query])
            
            # Encode section texts
            section_texts = [self._prepare_section_text(section) for section in sections]
            section_embeddings = self.model.encode(section_texts)
            
            # Calculate similarities
            similarities = cosine_similarity(query_embedding, section_embeddings)[0]
            
            # Create scored sections
            scored_sections = list(zip(sections, similarities))
            scored_sections.sort(key=lambda x: x[1], reverse=True)
            
            return scored_sections[:top_k]
            
        except Exception as e:
            logger.error(f"Embedding-based ranking failed: {e}")
            return self._rank_with_tfidf(query, sections, top_k)
    
    def _rank_with_tfidf(self, query: str, sections: List[Dict], 
                        top_k: int) -> List[Tuple[Dict, float]]:
        """Fallback ranking using TF-IDF."""
        try:
            # Prepare texts
            section_texts = [self._prepare_section_text(section) for section in sections]
            all_texts = [query] + section_texts
            
            # Fit and transform
            tfidf_matrix = self.vectorizer.fit_transform(all_texts)
            
            # Calculate similarities (query is at index 0)
            query_vector = tfidf_matrix[0]
            section_vectors = tfidf_matrix[1:]
            
            similarities = cosine_similarity(query_vector, section_vectors)[0]
            
            # Create scored sections
            scored_sections = list(zip(sections, similarities))
            scored_sections.sort(key=lambda x: x[1], reverse=True)
            
            return scored_sections[:top_k]
            
        except Exception as e:
            logger.error(f"TF-IDF ranking failed: {e}")
            return [(section, 0.0) for section in sections[:top_k]]
    
    def extract_relevant_subsections(self, section: Dict, query: str, 
                                   max_subsections: int = 3) -> List[Dict]:
        """
        Extract the most relevant subsections from a section.
        
        Args:
            section: Section dictionary with content
            query: Search query
            max_subsections: Maximum number of subsections to return
            
        Returns:
            List of subsection dictionaries
        """
        content = section.get("content", "")
        if not content or len(content) < 100:
            return [{
                "text": content,
                "page": section.get("start_page", 1)
            }]
        
        # Split content into paragraphs/sentences
        chunks = self._split_into_chunks(content)
        
        if len(chunks) <= max_subsections:
            return [{"text": chunk, "page": section.get("start_page", 1)} 
                   for chunk in chunks]
        
        # Rank chunks by relevance
        chunk_sections = [{"content": chunk} for chunk in chunks]
        ranked_chunks = self.rank_sections(query, chunk_sections, max_subsections)
        
        return [
            {
                "text": chunk_data[0]["content"],
                "page": section.get("start_page", 1)
            }
            for chunk_data in ranked_chunks
        ]
    
    def _prepare_section_text(self, section: Dict) -> str:
        """Prepare section text for analysis."""
        title = section.get("title", "")
        content = section.get("content", "")
        
        # Combine title and content with title given more weight
        if title and content:
            return f"{title}. {title}. {content}"  # Title repeated for emphasis
        elif title:
            return title
        elif content:
            return content
        else:
            return "No content"
    
    def _split_into_chunks(self, text: str, max_chunk_size: int = 300) -> List[str]:
        """Split text into meaningful chunks for subsection analysis."""
        # First try to split by paragraphs
        paragraphs = [p.strip() for p in text.split('\\n\\n') if p.strip()]
        
        if paragraphs and all(len(p) <= max_chunk_size for p in paragraphs):
            return paragraphs
        
        # Fall back to sentence splitting
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk + sentence) <= max_chunk_size:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks if chunks else [text[:max_chunk_size]]


class PersonaQueryBuilder:
    """
    Builds effective queries from persona and job descriptions.
    """
    
    @staticmethod
    def build_query(persona: str, job_to_be_done: str) -> str:
        """
        Combine persona and job into an effective search query.
        
        Args:
            persona: User persona/role description
            job_to_be_done: Task description
            
        Returns:
            Combined query string
        """
        # Clean inputs
        persona = persona.strip()
        job_to_be_done = job_to_be_done.strip()
        
        # Extract key terms from persona
        persona_terms = PersonaQueryBuilder._extract_key_terms(persona)
        
        # Extract key terms from job
        job_terms = PersonaQueryBuilder._extract_key_terms(job_to_be_done)
        
        # Build comprehensive query
        query_parts = []
        
        if persona_terms:
            query_parts.append(f"For {persona}:")
        
        if job_terms:
            query_parts.append(job_to_be_done)
        
        # Add emphasis on key terms
        all_terms = persona_terms + job_terms
        if all_terms:
            query_parts.append(f"Key topics: {', '.join(all_terms[:10])}")
        
        return " ".join(query_parts)
    
    @staticmethod
    def _extract_key_terms(text: str) -> List[str]:
        """Extract key terms from text."""
        # Remove common stop words and extract meaningful terms
        stop_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has',
            'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may',
            'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you',
            'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'
        }
        
        # Extract words (including compound terms)
        words = re.findall(r'\\b[a-zA-Z][a-zA-Z\\s]*[a-zA-Z]\\b', text.lower())
        
        # Filter and clean
        key_terms = []
        for word in words:
            word = word.strip()
            if len(word) > 2 and word not in stop_words:
                key_terms.append(word)
        
        return key_terms[:10]  # Limit to top 10 terms
