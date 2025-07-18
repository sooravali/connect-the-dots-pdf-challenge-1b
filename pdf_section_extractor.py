"""
PDF Text Extraction Utilities for Challenge 1B
Reuses and extends Challenge 1A components for document analysis.
"""

import fitz  # PyMuPDF
from pdfminer.high_level import extract_text as pdfminer_extract_text
from pathlib import Path
import logging
from typing import List, Dict, Tuple, Optional, Union
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFSectionExtractor:
    """
    Extracts sections from PDFs for semantic analysis.
    Builds on Challenge 1A's text extraction with section-aware processing.
    """
    
    def __init__(self):
        self.min_section_length = 50  # Minimum characters for a valid section
        self.max_section_length = 5000  # Maximum characters per section
        
    def extract_document_sections(self, pdf_path: Union[str, Path]) -> Dict:
        """
        Extract document sections with text content.
        
        Returns:
            Dictionary with document metadata and sections
        """
        try:
            return self._extract_with_pymupdf(pdf_path)
        except Exception as e:
            logger.warning(f"PyMuPDF extraction failed for {pdf_path}: {e}")
            return self._extract_with_pdfminer_fallback(pdf_path)
    
    def _extract_with_pymupdf(self, pdf_path: Union[str, Path]) -> Dict:
        """Extract sections using PyMuPDF with heading detection."""
        doc = fitz.open(str(pdf_path))
        
        # Extract all text with structure
        all_text = ""
        page_texts = {}
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_text = page.get_text()
            page_texts[page_num + 1] = page_text
            all_text += f"\\n[PAGE {page_num + 1}]\\n{page_text}\\n"
        
        doc.close()
        
        # Identify sections using heading patterns
        sections = self._identify_sections(all_text, page_texts)
        
        return {
            "filename": Path(pdf_path).name,
            "total_pages": len(page_texts),
            "sections": sections,
            "full_text": all_text
        }
    
    def _extract_with_pdfminer_fallback(self, pdf_path: Union[str, Path]) -> Dict:
        """Fallback extraction using pdfminer.six."""
        logger.info(f"Using pdfminer fallback for {pdf_path}")
        
        try:
            text = pdfminer_extract_text(str(pdf_path))
            
            # Create a single section if no structure can be detected
            sections = [{
                "title": f"Content from {Path(pdf_path).name}",
                "content": text,
                "start_page": 1,
                "end_page": 1,
                "section_type": "document"
            }]
            
            return {
                "filename": Path(pdf_path).name,
                "total_pages": 1,
                "sections": sections,
                "full_text": text
            }
            
        except Exception as e:
            logger.error(f"pdfminer extraction also failed for {pdf_path}: {e}")
            return {
                "filename": Path(pdf_path).name,
                "total_pages": 0,
                "sections": [],
                "full_text": ""
            }
    
    def _identify_sections(self, full_text: str, page_texts: Dict) -> List[Dict]:
        """Identify document sections using heading patterns and page breaks."""
        sections = []
        
        # Split text into potential sections using various patterns
        section_patterns = [
            r'\\n\\s*(?:Chapter|Section|Part)\\s+\\d+[^\\n]*\\n',
            r'\\n\\s*\\d+\\.\\s+[A-Z][^\\n]{10,100}\\n',
            r'\\n\\s*[A-Z][A-Z\\s]{10,}\\n',  # ALL CAPS headings
            r'\\n\\[PAGE\\s+\\d+\\]\\n',  # Page breaks
        ]
        
        # Find all potential section boundaries
        boundaries = [0]  # Start of document
        
        for pattern in section_patterns:
            matches = re.finditer(pattern, full_text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                boundaries.append(match.start())
        
        boundaries.append(len(full_text))  # End of document
        boundaries = sorted(set(boundaries))
        
        # Create sections from boundaries
        for i in range(len(boundaries) - 1):
            start_pos = boundaries[i]
            end_pos = boundaries[i + 1]
            
            section_text = full_text[start_pos:end_pos].strip()
            
            if len(section_text) < self.min_section_length:
                continue
            
            # Extract section title (first significant line)
            lines = section_text.split('\\n')
            title = self._extract_section_title(lines)
            
            # Determine page range
            start_page, end_page = self._determine_page_range(section_text, page_texts)
            
            # Clean section content
            content = self._clean_section_content(section_text)
            
            if content:
                sections.append({
                    "title": title,
                    "content": content[:self.max_section_length],
                    "start_page": start_page,
                    "end_page": end_page,
                    "section_type": self._classify_section_type(title, content)
                })
        
        # If no clear sections found, create page-based sections
        if not sections:
            sections = self._create_page_based_sections(page_texts)
        
        return sections
    
    def _extract_section_title(self, lines: List[str]) -> str:
        """Extract a meaningful title from section lines."""
        for line in lines[:10]:  # Check first 10 lines
            line = line.strip()
            if not line or line.startswith('[PAGE'):
                continue
            
            # Clean and validate as title
            if 10 <= len(line) <= 100 and not line.endswith('.'):
                # Remove leading numbers/markers
                title = re.sub(r'^\\d+\\.?\\s*', '', line)
                title = re.sub(r'^(Chapter|Section|Part)\\s+\\d+[:\\.]?\\s*', '', title, flags=re.IGNORECASE)
                return title.strip()
        
        return "Untitled Section"
    
    def _determine_page_range(self, section_text: str, page_texts: Dict) -> Tuple[int, int]:
        """Determine the page range for a section."""
        page_markers = re.findall(r'\\[PAGE\\s+(\\d+)\\]', section_text)
        
        if page_markers:
            pages = [int(p) for p in page_markers]
            return min(pages), max(pages)
        
        return 1, 1  # Default
    
    def _clean_section_content(self, content: str) -> str:
        """Clean section content for analysis."""
        # Remove page markers
        content = re.sub(r'\\[PAGE\\s+\\d+\\]', '', content)
        
        # Normalize whitespace
        content = re.sub(r'\\s+', ' ', content)
        
        # Remove very short lines (likely artifacts)
        lines = content.split('\\n')
        cleaned_lines = [line.strip() for line in lines if len(line.strip()) > 3]
        
        return '\\n'.join(cleaned_lines).strip()
    
    def _classify_section_type(self, title: str, content: str) -> str:
        """Classify the type of section based on title and content."""
        title_lower = title.lower()
        content_lower = content.lower()[:500]  # First 500 chars
        
        # Classification patterns
        if any(word in title_lower for word in ['introduction', 'overview', 'abstract']):
            return 'introduction'
        elif any(word in title_lower for word in ['conclusion', 'summary', 'final']):
            return 'conclusion'
        elif any(word in title_lower for word in ['method', 'approach', 'technique']):
            return 'methodology'
        elif any(word in title_lower for word in ['result', 'finding', 'outcome']):
            return 'results'
        elif any(word in title_lower for word in ['reference', 'citation', 'bibliography']):
            return 'references'
        else:
            return 'content'
    
    def _create_page_based_sections(self, page_texts: Dict) -> List[Dict]:
        """Create sections based on pages when no clear structure is found."""
        sections = []
        
        for page_num, page_text in page_texts.items():
            if len(page_text.strip()) < self.min_section_length:
                continue
            
            sections.append({
                "title": f"Page {page_num} Content",
                "content": page_text.strip()[:self.max_section_length],
                "start_page": page_num,
                "end_page": page_num,
                "section_type": "page"
            })
        
        return sections


def split_text_into_chunks(text: str, max_chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Split text into overlapping chunks for detailed analysis.
    
    Args:
        text: Input text to split
        max_chunk_size: Maximum characters per chunk
        overlap: Characters to overlap between chunks
        
    Returns:
        List of text chunks
    """
    if len(text) <= max_chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + max_chunk_size
        
        # Try to break at sentence boundary
        if end < len(text):
            # Look for sentence endings within the last 100 characters
            search_start = max(start + max_chunk_size - 100, start)
            sentence_end = text.rfind('.', search_start, end)
            if sentence_end > start:
                end = sentence_end + 1
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = end - overlap
        if start >= len(text):
            break
    
    return chunks


def clean_text(text: str) -> str:
    """Clean and normalize text content."""
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\\s+', ' ', text)
    
    # Remove non-printable characters except newlines and tabs
    text = re.sub(r'[^\\x20-\\x7E\\n\\t]', '', text)
    
    return text.strip()
