"""
Semantic Analysis Module for Challenge 1B
Handles persona-driven document intelligence using sentence embeddings.
"""

import logging
import time
import os
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
    Implements a tiered extraction strategy for optimal results.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.vectorizer = None
        self.use_embeddings = SENTENCE_TRANSFORMERS_AVAILABLE
        
        # Set system environment variable for transformers to avoid warnings
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        
        # Always set up TF-IDF as a backup
        self._setup_tfidf_fallback()
        
        # Try to load embedding model if available
        if self.use_embeddings:
            try:
                start_time = time.time()
                logger.info("Starting embedding model loading...")
                self._load_embedding_model()
                logger.info(f"Embedding model loaded in {time.time() - start_time:.2f}s")
            except Exception as e:
                logger.warning(f"Failed to load embedding model, will use TF-IDF fallback: {e}")
                self.use_embeddings = False
                
        # Tiered extraction configuration
        self.extraction_tier = 1  # Start with Tier 1 (fastest)
        self.max_tier = 2  # Currently implementing up to Tier 2 (Tier 3 would be LLM-based)
    
    def _load_embedding_model(self):
        """Load the sentence transformer model."""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            # Check if we're using the cached model path from environment variable
            cache_dir = os.environ.get('SENTENCE_TRANSFORMERS_HOME')
            if cache_dir:
                logger.info(f"Using model cache directory: {cache_dir}")
                
            self.model = SentenceTransformer(self.model_name)
            
            # Verify model is loaded
            if hasattr(self.model, '_first_module') and hasattr(self.model, '_modules'):
                logger.info("Embedding model loaded successfully with all components")
            else:
                logger.warning("Model loaded but may not have all components")
                
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
        """
        Fallback ranking using TF-IDF with enhanced ranking features.
        This provides a robust alternative when embeddings are unavailable.
        """
        try:
            # Prepare texts
            section_texts = [self._prepare_section_text(section) for section in sections]
            
            # Extract key terms from query for explicit matching
            query_terms = set(re.findall(r'\b\w+\b', query.lower()))
            
            # Remove common stop words from query terms
            stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
                        'by', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been'}
            query_terms = {term for term in query_terms if len(term) > 2 and term not in stop_words}
            
            # Combine TF-IDF with exact term matching for better relevance
            all_texts = [query] + section_texts
            
            # Calculate TF-IDF similarity
            tfidf_matrix = self.vectorizer.fit_transform(all_texts)
            query_vector = tfidf_matrix[0]
            section_vectors = tfidf_matrix[1:]
            tfidf_similarities = cosine_similarity(query_vector, section_vectors)[0]
            
            # Calculate additional term frequency match scores
            term_match_scores = []
            for i, section in enumerate(sections):
                text = section_texts[i].lower()
                
                # Count matches for each query term
                term_matches = sum(1 for term in query_terms if term in text)
                
                # Weighted by section properties
                score = term_matches / max(1, len(query_terms))
                
                # Boost scores for sections with good metadata
                if 'title' in section and any(term in section['title'].lower() for term in query_terms):
                    score *= 1.5  # Title match bonus
                
                if 'section_type' in section:
                    if section['section_type'] in ['introduction', 'conclusion', 'summary']:
                        score *= 1.2  # Important section types bonus
                
                term_match_scores.append(score)
            
            # Normalize term match scores
            if term_match_scores:
                max_score = max(term_match_scores) if max(term_match_scores) > 0 else 1
                term_match_scores = [score/max_score for score in term_match_scores]
                
            # Combine scores (70% TF-IDF, 30% term matching)
            combined_scores = [0.7 * tfidf + 0.3 * term for tfidf, term 
                              in zip(tfidf_similarities, term_match_scores)]
            
            # Create scored sections
            scored_sections = list(zip(sections, combined_scores))
            scored_sections.sort(key=lambda x: x[1], reverse=True)
            
            return scored_sections[:top_k]
            
        except Exception as e:
            logger.error(f"Enhanced TF-IDF ranking failed: {e}")
            # Fallback to basic ordering if all else fails
            return [(section, 1.0 - i/len(sections)) for i, section in enumerate(sections[:top_k])]
    
    def extract_relevant_subsections(self, section: Dict, query: str, 
                                   max_subsections: int = 3) -> List[Dict]:
        """
        Extract the most relevant subsections from a section with context preservation.
        Implements a tiered extraction strategy:
        
        Tier 1 (Fast): Uses keyword matching and TF-IDF for quick extraction
        Tier 2 (Precise): Uses embeddings-based semantic similarity with context preservation
        
        Args:
            section: Section dictionary with content
            query: Search query
            max_subsections: Maximum number of subsections to return
            
        Returns:
            List of subsection dictionaries with optimal context preservation
        """
        content = section.get("content", "")
        if not content or len(content) < 100:
            return [{
                "text": content,
                "page": section.get("start_page", 1)
            }]
        
        # Include section title in subsection context if available
        section_title = section.get("title", "").strip()
        start_page = section.get("start_page", 1)
        end_page = section.get("end_page", start_page)
        
        # Implement tiered extraction strategy
        if self.extraction_tier == 1:
            # Tier 1: Fast extraction using keyword matching
            subsections = self._extract_subsections_tier1(content, query, section_title, start_page, end_page, max_subsections)
            
            # Check if extraction was successful and sufficient
            if subsections and len(subsections) >= max_subsections // 2:
                return subsections
            
            # If tier 1 failed to produce good results, escalate to tier 2
            logger.info("Escalating to Tier 2 extraction for better results")
            self.extraction_tier = 2
        
        # Tier 2: More precise extraction with context preservation
        # Split content into paragraphs/sentences
        chunks = self._split_into_chunks(content)
        
        # If limited content, return all chunks
        if len(chunks) <= max_subsections:
            return [{"text": self._prepare_subsection_text(chunk, section_title), 
                   "page": self._estimate_chunk_page(i, len(chunks), start_page, end_page)} 
                   for i, chunk in enumerate(chunks)]
        
        # Rank chunks by relevance
        chunk_sections = [{"content": chunk} for chunk in chunks]
        ranked_chunks = self.rank_sections(query, chunk_sections, max_subsections * 2)  # Get more for context
        
        # Check if the ranked chunks are contextually related or scattered
        chunk_indices = []
        for chunk_data in ranked_chunks:
            chunk_text = chunk_data[0]["content"]
            chunk_score = chunk_data[1]
            chunk_idx = chunks.index(chunk_text) if chunk_text in chunks else -1
            if chunk_idx >= 0:
                chunk_indices.append((chunk_idx, chunk_score))
        
        # Sort indices by position in document, not by score
        chunk_indices.sort(key=lambda x: x[0])
        
        # If chunks are close to each other (likely part of same topic),
        # consider including context between them for better readability
        subsections = []
        
        # First try to identify contiguous blocks of text
        if len(chunk_indices) >= 2:
            # Group adjacent chunks together (chunks with small gaps)
            chunk_groups = []
            current_group = [chunk_indices[0]]
            
            for i in range(1, len(chunk_indices)):
                prev_idx = current_group[-1][0]
                current_idx = chunk_indices[i][0]
                
                # If chunks are close (gap <= 2 chunks), keep them in the same group
                if current_idx - prev_idx <= 3:
                    current_group.append(chunk_indices[i])
                else:
                    # Start a new group
                    chunk_groups.append(current_group)
                    current_group = [chunk_indices[i]]
            
            # Add the last group
            if current_group:
                chunk_groups.append(current_group)
            
            # Process each group to create a subsection with context
            for group in chunk_groups:
                if not group:
                    continue
                
                # Get start and end indices for this group
                min_idx = group[0][0]
                max_idx = group[-1][0]
                avg_score = sum(score for _, score in group) / len(group)
                
                # Include 1 chunk before and after for context if available
                context_min = max(0, min_idx - 1)
                context_max = min(len(chunks) - 1, max_idx + 1)
                
                # Extract continuous context
                combined_text = " ".join(chunks[context_min:context_max + 1])
                
                # Estimate page number based on chunk position in section
                relative_pos = (min_idx + max_idx) / (2 * len(chunks))
                page = start_page
                
                if end_page > start_page:
                    # Interpolate page number based on position in section
                    page = int(start_page + relative_pos * (end_page - start_page))
                
                subsections.append({
                    "text": self._prepare_subsection_text(combined_text, section_title),
                    "page": page,
                    "_score": avg_score  # Internal score for debugging
                })
                
                # Stop if we've reached our max subsections limit
                if len(subsections) >= max_subsections:
                    break
        
        # If we couldn't find good contextual groups, extract individual chunks
        if not subsections:
            # Take top scoring chunks
            top_indices = sorted(chunk_indices, key=lambda x: x[1], reverse=True)[:max_subsections]
            
            for chunk_idx, score in top_indices:
                chunk_text = chunks[chunk_idx]
                
                # Try to add some context (1 chunk before and after)
                context_start = max(0, chunk_idx - 1)
                context_end = min(len(chunks) - 1, chunk_idx + 1)
                context_text = " ".join(chunks[context_start:context_end + 1])
                
                # Estimate page number
                relative_pos = chunk_idx / len(chunks)
                page = start_page
                if end_page > start_page:
                    page = int(start_page + relative_pos * (end_page - start_page))
                
                subsections.append({
                    "text": self._prepare_subsection_text(context_text, section_title),
                    "page": page
                })
        
        return subsections
    
    def _estimate_chunk_page(self, chunk_idx: int, total_chunks: int, start_page: int, end_page: int) -> int:
        """Estimate the page number for a chunk based on its position within the section."""
        if start_page == end_page:
            return start_page
            
        # Calculate relative position in the section
        relative_pos = chunk_idx / total_chunks
        
        # Interpolate between start and end page
        return int(start_page + relative_pos * (end_page - start_page))
        
    def _prepare_subsection_text(self, text: str, section_title: str = "") -> str:
        """Format subsection text for optimal readability and context preservation."""
        if not text:
            return ""
            
        # Remove excessive whitespace while preserving paragraph structure
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\\n\\s*\\n', '\n\n', text)
        text = text.replace("\\n", "\n").replace("\\t", " ")
        
        # Clean up common PDF extraction artifacts
        text = text.replace("• ", "- ")  # Replace bullets with dashes
        text = re.sub(r'([.:;,!?])\s*\n', r'\1 ', text)  # Fix sentence breaks
        
        # Add section title as context if available and not already in the text
        if section_title and section_title.strip() and not text.startswith(section_title):
            # Check if the title content appears in the first 50 chars of text
            if not section_title.lower() in text[:50].lower():
                text = f"{section_title}:\n{text}"
        
        # Ensure text ends with proper sentence termination
        if text and not text.rstrip().endswith(('.', '!', '?', ':', ';', '"', "'", ')')):
            text = text.rstrip() + "."
        
        # Add ellipsis for truncated content
        if len(text) > 2000:
            # Try to find a sentence boundary for cleaner truncation
            truncate_pos = min(1950, len(text))
            last_period = text.rfind('.', 0, truncate_pos)
            if last_period > truncate_pos - 200:  # Only use if reasonably close to target
                truncate_pos = last_period + 1
                
            text = text[:truncate_pos] + " [...]"
            
        return text.strip()
    
    def _extract_subsections_tier1(self, content: str, query: str, 
                                 section_title: str, start_page: int, end_page: int, 
                                 max_subsections: int) -> List[Dict]:
        """
        Tier 1 extraction using fast keyword matching and basic relevance scoring.
        
        This is a faster alternative to the full semantic matching, useful for:
        1. Simple documents with clear structure
        2. When quick results are needed with lower precision requirements
        3. As a first pass before potentially escalating to more complex methods
        """
        # Extract key terms from query
        query_terms = self._extract_query_terms(query)
        if not query_terms:
            return []
            
        # Split content into paragraphs
        paragraphs = re.split(r'\n\s*\n|\r\n\s*\r\n', content)
        if not paragraphs:
            return []
            
        # Score paragraphs based on query term frequency
        scored_paragraphs = []
        for i, para in enumerate(paragraphs):
            if len(para) < 50:  # Skip very short paragraphs
                continue
                
            # Count query term occurrences
            para_lower = para.lower()
            term_matches = sum(1 for term in query_terms if term in para_lower)
            
            # Calculate position-based score (prioritize earlier paragraphs slightly)
            position_score = 1.0 - (i / max(1, len(paragraphs)))
            
            # Combine scores
            relevance_score = (0.8 * term_matches / max(1, len(query_terms))) + (0.2 * position_score)
            
            scored_paragraphs.append((para, relevance_score, i))
            
        # Sort by score and take top paragraphs
        scored_paragraphs.sort(key=lambda x: x[1], reverse=True)
        top_paragraphs = scored_paragraphs[:max_subsections]
        
        # Sort back by original order for context coherence
        top_paragraphs.sort(key=lambda x: x[2])
        
        # Convert to subsections
        subsections = []
        for para, score, idx in top_paragraphs:
            # Estimate page number based on paragraph position
            relative_pos = idx / max(1, len(paragraphs))
            page = start_page
            if end_page > start_page:
                page = int(start_page + relative_pos * (end_page - start_page))
                
            text = self._prepare_subsection_text(para, section_title)
            subsections.append({
                "text": text,
                "page": page
            })
            
        return subsections
    
    def _extract_query_terms(self, query: str) -> List[str]:
        """Extract key terms from query for matching."""
        # Simple stopword list for filtering
        stopwords = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
                   'by', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has'}
                   
        # Extract terms with basic cleaning
        terms = []
        words = re.findall(r'\b[a-zA-Z][a-zA-Z\-]+\b', query.lower())
        
        for word in words:
            if len(word) > 2 and word not in stopwords:
                terms.append(word)
                
        return terms
        
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
    
    def _split_into_chunks(self, text: str, max_chunk_size: int = 400, min_chunk_size: int = 100) -> List[str]:
        """
        Split text into meaningful semantic chunks for subsection analysis.
        Uses multiple strategies to ensure optimal chunking.
        
        Args:
            text: Input text to split into chunks
            max_chunk_size: Maximum size of each chunk
            min_chunk_size: Minimum size for a chunk to be considered valid
            
        Returns:
            List of text chunks with semantic coherence preserved
        """
        if not text or len(text) <= min_chunk_size:
            return [text] if text else []
            
        # Clean up text - normalize newlines and fix common issues
        text = text.replace('\\n', '\n').replace('\\t', ' ')
        text = re.sub(r'\s+', ' ', text)
        
        # Strategy 1: Try to split by semantic boundaries (headings, list markers)
        semantic_chunks = self._split_by_semantic_boundaries(text, max_chunk_size)
        if semantic_chunks and all(len(chunk) >= min_chunk_size for chunk in semantic_chunks):
            return semantic_chunks
            
        # Strategy 2: Try to split by paragraphs
        paragraph_pattern = re.compile(r'\n\s*\n|\r\n\s*\r\n')
        paragraphs = [p.strip() for p in paragraph_pattern.split(text) if p.strip()]
        
        if paragraphs and all(len(p) <= max_chunk_size for p in paragraphs):
            # Combine very short paragraphs
            combined_paragraphs = []
            current_p = ""
            
            for p in paragraphs:
                if len(current_p) + len(p) <= max_chunk_size:
                    current_p = current_p + " " + p if current_p else p
                else:
                    if current_p and len(current_p) >= min_chunk_size:
                        combined_paragraphs.append(current_p.strip())
                    current_p = p
                    
            if current_p and len(current_p) >= min_chunk_size:
                combined_paragraphs.append(current_p.strip())
                
            if combined_paragraphs:
                return combined_paragraphs
        
        # Strategy 3: Split by sentences with context preservation
        # This ensures we don't break in the middle of a sentence
        sentence_pattern = re.compile(r'(?<=[.!?])\s+')
        sentences = [s.strip() for s in sentence_pattern.split(text) if s.strip()]
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # If adding this sentence exceeds max size and we already have content
            if current_chunk and (len(current_chunk) + len(sentence) > max_chunk_size):
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                # Either the chunk is empty or we can add this sentence
                current_chunk = current_chunk + " " + sentence if current_chunk else sentence
        
        # Add the last chunk if it exists and meets minimum size
        if current_chunk and len(current_chunk) >= min_chunk_size:
            chunks.append(current_chunk.strip())
        
        # Strategy 4: Fallback - if all else fails, do simple character-based chunking
        # but try to break at word boundaries
        if not chunks:
            chunks = []
            for i in range(0, len(text), max_chunk_size - 50):
                # Get chunk that's a bit smaller than max to find a good break point
                chunk = text[i:i + max_chunk_size - 50]
                
                # If not at the end, try to find a sentence or word break
                if i + max_chunk_size - 50 < len(text):
                    # Try to find end of sentence
                    last_period = max(chunk.rfind('.'), chunk.rfind('!'), chunk.rfind('?'))
                    if last_period > len(chunk) - 20 and last_period > 0:
                        chunk = chunk[:last_period+1]
                    else:
                        # Try to find word break
                        last_space = chunk.rfind(' ')
                        if last_space > 0:
                            chunk = chunk[:last_space]
                
                if len(chunk) >= min_chunk_size:
                    chunks.append(chunk.strip())
        
        return chunks
        
    def _split_by_semantic_boundaries(self, text: str, max_size: int) -> List[str]:
        """Split text using semantic boundaries like headings and list markers."""
        # Common patterns that indicate semantic boundaries in text
        patterns = [
            # Headers - variations of numbering and formatting
            r'\n\s*\d+\.\d+\s+[A-Z]',  # Numbered headers like "1.2 TITLE"
            r'\n\s*[A-Z][A-Z\s]{5,}\n',  # ALL CAPS headers
            r'\n\s*(?:Section|Chapter|Part)\s+\d+',  # Section/Chapter headers
            
            # List markers
            r'\n\s*(?:•|\*|-|–)\s+',  # Bullet points
            r'\n\s*\d+\.\s+',  # Numbered lists
            r'\n\s*[a-z]\)\s+'  # Letter lists: a), b), etc.
        ]
        
        # Combine patterns
        combined_pattern = '|'.join(patterns)
        regex = re.compile(combined_pattern)
        
        # Find all matches
        matches = list(regex.finditer(text))
        if not matches:
            return []
            
        # Extract chunks based on these boundaries
        chunks = []
        start_pos = 0
        
        for match in matches:
            # Get the text up to this boundary
            if match.start() - start_pos > max_size:
                # Too big, need to split further
                mid_point = start_pos + max_size
                # Try to find a good break point
                last_period = text.rfind('.', start_pos, mid_point)
                if last_period > start_pos + (max_size // 2):
                    chunks.append(text[start_pos:last_period+1].strip())
                    start_pos = last_period + 1
                    
                # Continue splitting if needed
                while start_pos < match.start():
                    end_pos = min(start_pos + max_size, match.start())
                    chunks.append(text[start_pos:end_pos].strip())
                    start_pos = end_pos
            else:
                # This section is a reasonable size
                chunk = text[start_pos:match.start()].strip()
                if chunk:
                    chunks.append(chunk)
            
            start_pos = match.start()
                
        # Add the last chunk
        if start_pos < len(text):
            remaining = text[start_pos:].strip()
            if remaining:
                if len(remaining) <= max_size:
                    chunks.append(remaining)
                else:
                    # Split the remaining text
                    for i in range(0, len(remaining), max_size):
                        chunks.append(remaining[i:i+max_size].strip())
        
        return [c for c in chunks if c.strip()]


class PersonaQueryBuilder:
    """
    Builds effective queries from persona and job descriptions.
    """
    
    @staticmethod
    def build_query(persona: str, job_to_be_done: str) -> str:
        """
        Build an optimized semantic query from persona and job descriptions.
        
        This enhanced method creates a well-structured query that:
        1. Captures the expertise level and specialized knowledge of the persona
        2. Identifies actionable elements of the job-to-be-done
        3. Includes domain-specific terminology to improve matching
        4. Prioritizes information types most valuable to the persona
        
        Args:
            persona: User persona/role description
            job_to_be_done: Task description
            
        Returns:
            Combined query string optimized for semantic matching
        """
        # Clean inputs
        persona = persona.strip()
        job_to_be_done = job_to_be_done.strip()
        
        # Extract key terms with importance weighting
        persona_terms = PersonaQueryBuilder._extract_key_terms(persona)
        job_terms = PersonaQueryBuilder._extract_key_terms(job_to_be_done)
        
        # Analyze persona type and expertise level
        expertise_level = "beginner"  # Default assumption
        if any(term in persona.lower() for term in ["professional", "expert", "specialist", 
                                                  "senior", "experienced", "advanced"]):
            expertise_level = "expert"
        elif any(term in persona.lower() for term in ["student", "junior", "beginner", 
                                                    "learning", "novice"]):
            expertise_level = "beginner"
        else:
            expertise_level = "intermediate"
            
        # Identify persona domain
        domain = "general"
        domain_keywords = {
            "technical": ["developer", "engineer", "programmer", "technical", "scientist", "researcher"],
            "business": ["manager", "executive", "analyst", "consultant", "professional", "planner"],
            "creative": ["designer", "artist", "writer", "creator", "producer"],
            "educational": ["teacher", "professor", "educator", "student", "academic"],
            "medical": ["doctor", "nurse", "therapist", "health", "medical", "clinician"]
        }
        
        for dom, keywords in domain_keywords.items():
            if any(kw in persona.lower() for kw in keywords):
                domain = dom
                break
        
        # Identify task action type
        action_type = "research"  # Default
        action_keywords = {
            "create": ["create", "build", "develop", "design", "write", "produce", "implement"],
            "analyze": ["analyze", "review", "evaluate", "assess", "compare", "examine", "study"],
            "plan": ["plan", "prepare", "organize", "schedule", "arrange", "strategize", "outline"],
            "present": ["present", "communicate", "explain", "demonstrate", "showcase", "report"],
        }
        
        for act, keywords in action_keywords.items():
            if any(kw in job_to_be_done.lower() for kw in keywords):
                action_type = act
                break
        
        # Build query parts with strategic structure
        query_parts = []
        
        # 1. Core task statement with persona context
        query_parts.append(f"Information needed for a {persona} to {job_to_be_done.lower()}")
        
        # 2. Expertise-adjusted content focus
        if expertise_level == "expert":
            query_parts.append(f"Technical details and advanced information for {persona}")
            query_parts.append(f"Specialized knowledge about {' '.join(job_terms[:5])}")
        elif expertise_level == "beginner":
            query_parts.append(f"Fundamental concepts and step-by-step guides for {persona}")
            query_parts.append(f"Clear explanations of {' '.join(job_terms[:5])}")
        else:  # intermediate
            query_parts.append(f"Practical information and implementation guidelines for {persona}")
            query_parts.append(f"Applied knowledge about {' '.join(job_terms[:5])}")
        
        # 3. Action-specific information needs
        if action_type == "create":
            query_parts.append(f"Templates, examples, and methods to create {' '.join(job_terms[:3])}")
        elif action_type == "analyze":
            query_parts.append(f"Data points, frameworks, and criteria to evaluate {' '.join(job_terms[:3])}")
        elif action_type == "plan":
            query_parts.append(f"Strategies, timelines, and best practices for planning {' '.join(job_terms[:3])}")
        elif action_type == "present":
            query_parts.append(f"Key insights, visualizations, and communication techniques for {' '.join(job_terms[:3])}")
        else:  # research
            query_parts.append(f"Background information, sources, and findings about {' '.join(job_terms[:3])}")
        
        # 4. Domain-specific priorities
        if domain == "technical":
            query_parts.append(f"Technical specifications, implementation details, algorithms for {' '.join(job_terms[:3])}")
        elif domain == "business":
            query_parts.append(f"Business impact, metrics, cost-benefit analysis of {' '.join(job_terms[:3])}")
        elif domain == "creative":
            query_parts.append(f"Design principles, inspiration, creative approaches to {' '.join(job_terms[:3])}")
        elif domain == "educational":
            query_parts.append(f"Learning objectives, teaching materials, curriculum design for {' '.join(job_terms[:3])}")
        elif domain == "medical":
            query_parts.append(f"Clinical guidelines, patient outcomes, treatment protocols for {' '.join(job_terms[:3])}")
            
        # 5. Direct keyword matching for critical terms (high precision)
        high_priority_terms = persona_terms[:3] + job_terms[:5]
        if high_priority_terms:
            query_parts.append(f"Important keywords: {' '.join(high_priority_terms)}")
            
        # 6. Synonyms and related concepts for better recall
        if job_terms:
            alternatives = []
            for term in job_terms[:3]:
                # Add simple word variations
                if term.endswith('ing'):
                    alternatives.append(term[:-3])
                elif term.endswith('s') and len(term) > 3:
                    alternatives.append(term[:-1])
                elif not term.endswith('s') and len(term) > 3:
                    alternatives.append(term + 's')
            
            if alternatives:
                query_parts.append(f"Related terms: {' '.join(alternatives)}")
        
        # Combine all parts and ensure proper length
        final_query = " ".join(query_parts)
        
        # Avoid overly long queries
        if len(final_query) > 1200:
            # Truncate smartly by removing least important parts first
            parts = final_query.split('. ')
            final_query = '. '.join(parts[:5])
            
        # Add persona and job as top-level priority terms at the end
        final_query = f"{final_query} {persona} {job_to_be_done}"
            
        return final_query
    
    @staticmethod
    def _extract_key_terms(text: str) -> List[str]:
        """
        Extract key terms from text with advanced filtering and n-gram extraction.
        
        Args:
            text: Input text to extract terms from
            
        Returns:
            List of key terms ordered by importance
        """
        if not text:
            return []
            
        # Expanded stop words list for better filtering
        stop_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has',
            'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may',
            'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you',
            'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
            'my', 'your', 'his', 'our', 'their', 'its', 'am', 'being', 'as', 'if',
            'then', 'than', 'so', 'too', 'very', 'just', 'more', 'most', 'some',
            'any', 'all', 'also', 'about', 'need', 'want', 'using'
        }
        
        # Normalize text
        text = text.lower()
        
        # Extract single terms
        single_terms = []
        words = re.findall(r'\\b[a-zA-Z][a-zA-Z]*\\b', text)
        for word in words:
            if len(word) > 2 and word not in stop_words:
                single_terms.append(word)
        
        # Extract bigrams (important compound terms)
        bigrams = []
        words = text.split()
        for i in range(len(words) - 1):
            w1, w2 = words[i].strip(",.;:!?"), words[i + 1].strip(",.;:!?")
            if (len(w1) > 2 and len(w2) > 2 and 
                w1 not in stop_words and w2 not in stop_words):
                bigram = f"{w1} {w2}"
                # Check if bigram appears multiple times (more important)
                if text.count(bigram) > 1 or len(bigram) > 12:  # Longer or repeated bigrams likely important
                    bigrams.append(bigram)
        
        # Extract noun phrases (simplified approach)
        noun_phrases = re.findall(r'\\b[a-zA-Z]+\\s+(?:of|for|in|on)\\s+[a-zA-Z]+\\b', text)
        filtered_phrases = [phrase for phrase in noun_phrases 
                           if len(phrase) > 7 and not any(word in phrase.split() for word in stop_words)]
        
        # Prioritize terms (weighted ranking)
        key_terms = []
        
        # First add bigrams (most specific)
        key_terms.extend(bigrams[:5])
        
        # Then noun phrases
        key_terms.extend(filtered_phrases[:3])
        
        # Then single terms to round out
        key_terms.extend(single_terms)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_terms = []
        for term in key_terms:
            if term not in seen:
                seen.add(term)
                unique_terms.append(term)
        
        return unique_terms[:15]  # Return top 15 terms for better coverage
