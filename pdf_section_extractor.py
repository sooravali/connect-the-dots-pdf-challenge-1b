"""
PDF Text Extraction Utilities for Challenge 1B
Reuses and extends Challenge 1A components for document analysis.
"""

# Try PyMuPDF import, fall back to pdfminer only if available
PYMUPDF_AVAILABLE = False
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    print("PyMuPDF not available - using pdfminer.six only")
    pass

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
        if PYMUPDF_AVAILABLE:
            try:
                return self._extract_with_pymupdf(pdf_path)
            except Exception as e:
                logger.warning(f"PyMuPDF extraction failed for {pdf_path}: {e}")
                return self._extract_with_pdfminer_fallback(pdf_path)
        else:
            return self._extract_with_pdfminer_fallback(pdf_path)
    
    def _extract_with_pymupdf(self, pdf_path: Union[str, Path]) -> Dict:
        """Extract sections using PyMuPDF with heading detection based on font analysis."""
        if not PYMUPDF_AVAILABLE:
            raise ImportError("PyMuPDF not available")
            
        doc = fitz.open(str(pdf_path))
        sections = []
        current_section = None
        total_pages = len(doc)
        
        try:
            for page_num in range(total_pages):
                page = doc[page_num]
                page_number = page_num + 1
                
                # Get structured text with font information
                text_dict = page.get_text("dict")
                
                for block in text_dict["blocks"]:
                    if "lines" not in block: 
                        continue
                    
                    # Extract text and check for heading characteristics
                    block_text = ""
                    is_heading = False
                    
                    for line in block["lines"]:
                        for span in line["spans"]:
                            span_text = span.get("text", "").strip()
                            if span_text:
                                block_text += span_text + " "
                                
                                # Check if this span looks like a heading
                                # Bold font (flags & 16) or font name contains 'Bold'
                                font_flags = span.get("flags", 0)
                                font_name = span.get("font", "")
                                
                                if (font_flags & 16) or ("Bold" in font_name):
                                    is_heading = True
                    
                    block_text = block_text.strip()
                    if not block_text:
                        continue
                    
                    # Skip bullet points as headings
                    if block_text.startswith("•") or block_text.startswith("o "):
                        is_heading = False
                    
                    if is_heading and len(block_text) < 200:  # Reasonable heading length
                        # Start a new section
                        if current_section and current_section["content"].strip():
                            sections.append(current_section)
                        
                        current_section = {
                            "title": block_text,
                            "content": "",
                            "start_page": page_number,
                            "end_page": page_number,
                            "section_type": "heading"
                        }
                    else:
                        # Add content to current section
                        if current_section is None:
                            # Create initial section if no heading found yet
                            current_section = {
                                "title": "Document Content",
                                "content": "",
                                "start_page": page_number,
                                "end_page": page_number,
                                "section_type": "content"
                            }
                        
                        current_section["content"] += " " + block_text
                        current_section["end_page"] = page_number
            
            # Add the final section
            if current_section and current_section["content"].strip():
                sections.append(current_section)
            
        finally:
            doc.close()
        
        # Clean up sections
        cleaned_sections = []
        for section in sections:
            # Clean content
            content = re.sub(r'\s+', ' ', section["content"]).strip()
            if len(content) >= self.min_section_length:
                section["content"] = content[:self.max_section_length]
                cleaned_sections.append(section)
        
        # If no sections found, create a fallback
        if not cleaned_sections:
            doc_reopen = fitz.open(str(pdf_path))
            try:
                full_text = ""
                for page in doc_reopen:
                    full_text += page.get_text() + " "
                
                cleaned_sections = [{
                    "title": "Complete Document",
                    "content": full_text[:self.max_section_length],
                    "start_page": 1,
                    "end_page": total_pages,
                    "section_type": "document"
                }]
            finally:
                doc_reopen.close()
        
        return {
            "filename": Path(pdf_path).name,
            "total_pages": total_pages,
            "sections": cleaned_sections,
            "full_text": ""  # Not needed for new approach
        }
    
    def _extract_with_pdfminer_fallback(self, pdf_path: Union[str, Path]) -> Dict:
        """Enhanced fallback extraction using pdfminer.six with section detection."""
        logger.info(f"Using pdfminer fallback for {pdf_path}")
        
        try:
            text = pdfminer_extract_text(str(pdf_path))
            
            # Try to identify sections using patterns
            sections = self._identify_sections_from_text(text)
            
            return {
                "filename": Path(pdf_path).name,
                "total_pages": self._estimate_pages(text),
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
    
    def _estimate_pages(self, text: str) -> int:
        """Estimate number of pages from text length."""
        # Rough estimate: ~3000 chars per page
        return max(1, len(text) // 3000)
    
    def _identify_sections_from_text(self, text: str) -> List[Dict]:
        """Identify sections from plain text using patterns."""
        sections = []
        
        # Split text into potential sections
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        current_section = {
            "title": "Document Content",
            "content": "",
            "start_page": 1,
            "end_page": 1,
            "section_type": "content"
        }
        
        for i, paragraph in enumerate(paragraphs):
            # Check if paragraph looks like a heading
            if self._looks_like_heading(paragraph):
                # Save previous section if it has content
                if current_section["content"].strip():
                    current_section["content"] = current_section["content"][:self.max_section_length]
                    sections.append(current_section.copy())
                
                # Start new section
                current_section = {
                    "title": paragraph[:100],  # Use first part as title
                    "content": paragraph,
                    "start_page": max(1, i // 10),  # Rough page estimate
                    "end_page": max(1, i // 10),
                    "section_type": "heading"
                }
            else:
                # Add to current section
                current_section["content"] += "\n" + paragraph
                current_section["end_page"] = max(1, i // 10)
        
        # Add final section
        if current_section["content"].strip():
            current_section["content"] = current_section["content"][:self.max_section_length]
            sections.append(current_section)
        
        # If no sections identified, create one big section
        if not sections:
            sections.append({
                "title": "Complete Document",
                "content": text[:self.max_section_length],
                "start_page": 1,
                "end_page": self._estimate_pages(text),
                "section_type": "document"
            })
        
        return sections
    
    def _looks_like_heading(self, text: str) -> bool:
        """Check if text looks like a section heading."""
        text = text.strip()
        return (
            len(text) < 200 and  # Short text
            (text.isupper() or  # All caps
             text.endswith(':') or  # Ends with colon
             bool(re.match(r'^\\d+\\.', text)) or  # Starts with number
             bool(re.match(r'^[A-Z][^.!?]*$', text)))  # Title case, no punctuation
        )
    
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
