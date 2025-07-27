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
        # First perform triage to determine document type
        doc_type = self._perform_document_triage(pdf_path)
        logger.info(f"Document type detected: {doc_type} for {pdf_path}")
        
        if doc_type == "image_based":
            # For image-based PDFs, OCR would be ideal but is not implemented
            # Fall back to pdfminer which sometimes can extract text from image PDFs
            logger.warning(f"Image-based PDF detected: {pdf_path}. OCR would be ideal but using fallback.")
            return self._extract_with_pdfminer_fallback(pdf_path)
        else:
            # For text-based or hybrid PDFs, use PyMuPDF
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
        
        # Perform document triage during extraction
        document_type = self._analyze_document_content(doc)
        
        doc.close()
        
        # Identify sections using heading patterns
        sections = self._identify_sections(all_text, page_texts)
        
        result = {
            "filename": Path(pdf_path).name,
            "total_pages": len(page_texts),
            "sections": sections,
            "full_text": all_text,
            "document_type": document_type
        }
        
        return result
        
    def _analyze_document_content(self, doc) -> str:
        """Analyze document content to determine its characteristics."""
        # Check a sample of pages (first, middle, last)
        total_pages = len(doc)
        
        if total_pages == 0:
            return "empty"
            
        # Sample pages to analyze
        pages_to_check = [0]  # Always check first page
        
        if total_pages > 2:
            pages_to_check.append(total_pages // 2)  # Middle page
            
        if total_pages > 1:
            pages_to_check.append(total_pages - 1)  # Last page
            
        # Check for forms, tables, images
        has_forms = False
        has_tables = False
        has_images = False
        has_text = False
        
        for page_num in pages_to_check:
            page = doc[page_num]
            
            # Check for forms
            if len(page.widgets()) > 0:
                has_forms = True
                
            # Check for images
            if len(page.get_images()) > 0:
                has_images = True
                
            # Check for text
            if page.get_text().strip():
                has_text = True
                
            # Rough table detection (looking for grid-like structures)
            # This is simplified and not fully reliable
            lines = page.get_drawings()
            horizontal_lines = 0
            vertical_lines = 0
            
            for line in lines:
                if "rect" in line:
                    rect = line["rect"]
                    width = rect[2] - rect[0]
                    height = rect[3] - rect[1]
                    
                    if width > 3 * height:  # Likely horizontal line
                        horizontal_lines += 1
                    elif height > 3 * width:  # Likely vertical line
                        vertical_lines += 1
                        
            if horizontal_lines >= 3 and vertical_lines >= 3:
                has_tables = True
                
        # Determine document type based on content
        if has_forms:
            return "form"
        elif has_tables and has_text:
            return "table_document"
        elif has_images and not has_text:
            return "image_only"
        elif has_images and has_text:
            return "mixed_content"
        else:
            return "text_document"
    
    def _extract_with_pdfminer_fallback(self, pdf_path: Union[str, Path]) -> Dict:
        """Fallback extraction using pdfminer.six."""
        logger.info(f"Using pdfminer fallback for {pdf_path}")
        
        try:
            text = pdfminer_extract_text(str(pdf_path))
            
            # Try to extract a meaningful title from the first few lines
            lines = text.split('\n')
            title = "Untitled Section"
            
            for line in lines[:20]:  # Check first 20 lines
                line = line.strip()
                if line and len(line) > 5 and len(line) < 150:
                    # Skip obvious non-titles (single words, page numbers, etc.)
                    if len(line.split()) > 1 and not re.match(r'^\d+$', line):
                        title = line
                        break
            
            # If no good title found, use the filename but try to make it more descriptive
            if title == "Untitled Section":
                filename = Path(pdf_path).name
                # Remove file extension and replace underscores/hyphens with spaces
                title = re.sub(r'\.\w+$', '', filename)
                title = re.sub(r'[_\-]', ' ', title)
            
            # Create a single section with better title
            sections = [{
                "title": title,
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
        
        # Enhanced section detection patterns - expanded for better coverage
        section_patterns = [
            # Standard section identifiers
            r'\\n\\s*(?:Chapter|Section|Part|Appendix)\\s+\\d+[^\\n]*\\n',
            # Numbered headings (1.2, 3.4.5, etc.)
            r'\\n\\s*\\d+(?:\\.\\d+)*\\s+[A-Z][^\\n]*\\n',
            # ALL CAPS headings (common for titles)
            r'\\n\\s*[A-Z][A-Z\\s]{5,}\\n', 
            # Common section headers
            r'\\n\\s*(?:INTRODUCTION|ABSTRACT|CONCLUSION|METHODOLOGY|RESULTS|DISCUSSION|REFERENCES|BIBLIOGRAPHY|APPENDIX)[^\\n]*\\n',
            # Headers with common prefixes
            r'\\n\\s*(?:Overview|Introduction|Background|Executive Summary|Key Findings|Analysis|Recommendations)[^\\n]*\\n',
            # Headers with formatting indicators (bullets, etc.)
            r'\\n\\s*[•\\*\\-–—]\\s+[A-Z][^\\n]*\\n',
            # Headers with numbering patterns
            r'\\n\\s*(?:[A-Z]\\.|[ivxlcdm]+\\.|[IVXLCDM]+\\.)\\s+[A-Z][^\\n]*\\n',
            # FAQ or Q&A patterns
            r'\\n\\s*(?:Q|Question)\\s*(?:\\d+|\\.)?\\s*[:\\.]?\\s*[A-Z][^\\n]*\\n',
            # Form field headers and labels
            r'\\n\\s*[A-Z][a-zA-Z\\s]+\\s*:\\s*\\_*\\s*\\n',
            # Page breaks - important for fallback segmentation
            r'\\n\\[PAGE\\s+\\d+\\]\\n',
        ]
        
        # Find all potential section boundaries
        boundaries = [0]  # Start of document
        
        for pattern in section_patterns:
            matches = re.finditer(pattern, full_text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                boundaries.append(match.start())
        
        boundaries.append(len(full_text))  # End of document
        boundaries = sorted(set(boundaries))
        
        # Smart section merging - avoid creating too small sections
        merged_boundaries = [boundaries[0]]
        for i in range(1, len(boundaries) - 1):
            current_boundary = boundaries[i]
            next_boundary = boundaries[i + 1]
            
            # If section would be too small, skip this boundary
            if next_boundary - current_boundary < self.min_section_length:
                continue
            
            # Don't merge if this looks like a real section header
            section_start = full_text[current_boundary:current_boundary + 200]
            potential_header = section_start.split('\n')[0] if '\n' in section_start else section_start[:100]
            
            # Check if this looks like a real header (not just a page break)
            if not potential_header.startswith('[PAGE') and len(potential_header.strip()) > 5:
                merged_boundaries.append(current_boundary)
        
        merged_boundaries.append(boundaries[-1])  # Always include the end
        boundaries = merged_boundaries
        
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
            
        # Post-process: ensure sections have unique, meaningful titles
        processed_sections = []
        seen_titles = set()
        
        for i, section in enumerate(sections):
            title = section["title"]
            
            # If we have a duplicate or generic "Untitled Section"
            if title in seen_titles or title == "Untitled Section":
                # Try to extract a better title from the first paragraph
                content = section["content"]
                first_para = content.split("\n\n")[0] if "\n\n" in content else content[:300]
                
                # Create a more descriptive title from content
                if len(first_para) > 10:
                    content_title = first_para[:50].strip()
                    if content_title.endswith('.'):
                        content_title = content_title[:-1]
                    
                    # Make sure this derived title is unique
                    if content_title not in seen_titles:
                        section["title"] = content_title
                    else:
                        # Add a numeric suffix if still duplicate
                        section["title"] = f"{title} ({i+1})"
                else:
                    section["title"] = f"Content Section {i+1}"
            
            seen_titles.add(section["title"])
            processed_sections.append(section)
        
        return processed_sections
    
    def _extract_section_title(self, lines: List[str]) -> str:
        """Extract a meaningful title from section lines."""
        # First, look for strong title indicators in the first several lines
        for line in lines[:15]:  # Examine more lines (15 instead of 10)
            line = line.strip()
            if not line or line.startswith('[PAGE'):
                continue
            
            # Check for common title patterns with high confidence
            # ALL CAPS lines are often titles
            if line.isupper() and 4 <= len(line) <= 150:
                return line.strip()
                
            # Lines with specific prefixes are likely titles
            if any(line.lower().startswith(prefix) for prefix in 
                  ['chapter ', 'section ', 'part ', 'introduction', 'overview', 
                   'appendix', 'executive summary', 'abstract']):
                # Remove leading numbers/markers
                title = re.sub(r'^\\d+\\.?\\s*', '', line)
                title = re.sub(r'^(Chapter|Section|Part|Appendix)\\s+\\d+[:\\.]?\\s*', '', title, flags=re.IGNORECASE)
                return title.strip()
        
        # More flexible title extraction for regular cases
        for line in lines[:15]:
            line = line.strip()
            if not line or line.startswith('[PAGE'):
                continue
            
            # More flexible length requirements
            if 4 <= len(line) <= 150:  # Allow shorter and longer titles
                # Skip lines that look like paragraphs (multiple sentences ending with periods)
                if len(line.split('. ')) > 2 and line.endswith('.'):
                    continue
                    
                # Skip likely dates, single words, and footer/header information
                if re.match(r'^\d{1,2}/\d{1,2}/\d{2,4}$', line) or len(line.split()) <= 1:
                    continue
                    
                # Remove leading numbers, bullets, etc.
                title = re.sub(r'^[0-9]+\\.?\\s*', '', line)
                title = re.sub(r'^[•\\*\\-–—]\\s*', '', title)
                title = re.sub(r'^(Chapter|Section|Part)\\s+\\d+[:\\.]?\\s*', '', title, flags=re.IGNORECASE)
                
                # Truncate the title if it's too long
                if len(title) > 100:
                    title = title[:97] + '...'
                
                return title.strip()
        
        # Look for the first non-empty line as a last resort
        for line in lines:
            if line.strip() and not line.startswith('[PAGE'):
                title = line.strip()
                if len(title) > 100:
                    title = title[:97] + '...'
                return title
        
        return "Untitled Section"
    
    def _determine_page_range(self, section_text: str, page_texts: Dict) -> Tuple[int, int]:
        """Determine the page range for a section."""
        page_markers = re.findall(r'\\[PAGE\\s+(\\d+)\\]', section_text)
        
        if page_markers:
            pages = [int(p) for p in page_markers]
            if pages:
                return min(pages), max(pages)
        
        # Fallback: try to match section content against page text
        # This helps when page markers aren't properly captured
        start_page = 1
        end_page = 1
        
        # Get a sample of text from the beginning and end of the section (excluding page markers)
        clean_text = re.sub(r'\\[PAGE\\s+\\d+\\]', '', section_text)
        
        if len(clean_text) > 50:  # Only try matching if we have enough content
            # Get samples from beginning and end of section
            start_sample = clean_text[:100].strip()
            end_sample = clean_text[-100:].strip()
            
            # Find pages containing these samples
            start_page_candidates = []
            end_page_candidates = []
            
            for page_num, page_text in page_texts.items():
                if start_sample and start_sample in page_text:
                    start_page_candidates.append(page_num)
                if end_sample and end_sample in page_text:
                    end_page_candidates.append(page_num)
            
            # Use the best matches if found
            if start_page_candidates:
                start_page = min(start_page_candidates)
            if end_page_candidates:
                end_page = max(end_page_candidates)
                
            # Ensure end_page is not less than start_page
            if end_page < start_page:
                end_page = start_page
        
        return start_page, end_page
    
    def _clean_section_content(self, content: str) -> str:
        """Clean section content for analysis while preserving meaningful structure."""
        if not content:
            return ""
            
        # Remove page markers
        content = re.sub(r'\\[PAGE\\s+\\d+\\]', '', content)
        
        # Split into lines for better processing
        lines = content.split('\\n')
        cleaned_lines = []
        
        # Process line by line
        for line in lines:
            line = line.strip()
            
            # Skip empty lines or very short lines that are likely artifacts
            if not line or len(line) <= 2:
                continue
                
            # Skip lines that are just page numbers or irrelevant markers
            if re.match(r'^-?\d+$', line) or line in ['-', '•', '*', '|']:
                continue
                
            # Handle common formatting issues
            # Fix line breaks in the middle of sentences
            if cleaned_lines and cleaned_lines[-1] and not cleaned_lines[-1].endswith('.') and \
               not cleaned_lines[-1].endswith('?') and not cleaned_lines[-1].endswith('!') and \
               not cleaned_lines[-1].endswith(':') and not cleaned_lines[-1].endswith(';'):
                if not line[0].isupper() or line[0].isdigit():
                    # Likely a continuation of the previous line
                    cleaned_lines[-1] = cleaned_lines[-1] + ' ' + line
                    continue
            
            # Add the cleaned line
            cleaned_lines.append(line)
        
        # Join lines with proper paragraph handling
        # Group consecutive lines into paragraphs
        paragraphs = []
        current_para = []
        
        for line in cleaned_lines:
            if not line.strip():  # Empty line indicates paragraph break
                if current_para:
                    paragraphs.append(' '.join(current_para))
                    current_para = []
            else:
                current_para.append(line)
                
        # Add the last paragraph if it exists
        if current_para:
            paragraphs.append(' '.join(current_para))
        
        # Join paragraphs with double newline for readability
        result = '\\n\\n'.join(paragraphs)
        
        # Final cleanup - normalize whitespace within paragraphs
        result = re.sub(r' {2,}', ' ', result)
        
        return result.strip()
    
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
    
    def _perform_document_triage(self, pdf_path: Union[str, Path]) -> str:
        """
        Perform triage to determine if a PDF is text-based, image-based, or hybrid.
        Uses PyMuPDF to analyze content blocks and their types.
        
        Returns:
            str: "text_based", "image_based", or "hybrid"
        """
        try:
            doc = fitz.open(str(pdf_path))
            text_area_total = 0
            image_area_total = 0
            total_pages = len(doc)
            
            # Check at least the first 3 pages or all pages if fewer
            pages_to_check = min(3, total_pages)
            
            for page_num in range(pages_to_check):
                page = doc[page_num]
                
                # Get all content blocks on the page
                blocks = page.get_text("blocks")
                
                page_text_area = 0
                page_image_area = 0
                
                for block in blocks:
                    # Get block coordinates
                    x0, y0, x1, y1, block_text, block_type, _ = block
                    block_area = (x1 - x0) * (y1 - y0)
                    
                    # Check if this is an image block
                    if '<image:' in block_text:
                        page_image_area += block_area
                    else:
                        page_text_area += block_area
                
                text_area_total += page_text_area
                image_area_total += page_image_area
            
            doc.close()
            
            # Determine document type based on content areas
            # Using thresholds to categorize document
            if text_area_total == 0 and image_area_total > 0:
                return "image_based"
            elif image_area_total > 0 and image_area_total / (text_area_total + image_area_total) > 0.8:
                # If images cover more than 80% of content area
                return "image_based"
            elif image_area_total > 0 and image_area_total / (text_area_total + image_area_total) > 0.3:
                # If images cover between 30% and 80% of content area
                return "hybrid"
            else:
                return "text_based"
                
        except Exception as e:
            logger.warning(f"Error during document triage: {e}")
            # Default to text_based if triage fails
            return "text_based"
    
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
    
    # Convert literal escape sequences to actual characters
    text = text.replace('\\n', '\n').replace('\\t', '\t').replace('\\r', '\r')
    
    # Replace common encoding artifacts from PDF extraction
    replacements = {
        '•': '- ',  # Replace bullets with dashes
        '–': '-',   # Standardize dashes
        '—': '-',   # Standardize dashes
        ''': "'",   # Standardize quotes
        ''': "'",   # Standardize quotes
        '"': '"',   # Standardize quotes
        '"': '"',   # Standardize quotes
        '…': '...',  # Standardize ellipses
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Remove excessive whitespace while preserving newlines
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n+', '\n', text)
    
    # Remove control characters except newlines and tabs
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    
    # Use UTF-8 encoding with fallback to preserve as much text as possible
    try:
        # Try to normalize text to closest ASCII representation
        import unicodedata
        text = unicodedata.normalize('NFKD', text)
        # Only remove characters that can't be represented in UTF-8
        text = text.encode('utf-8', 'ignore').decode('utf-8')
    except Exception:
        # Fallback to ASCII if Unicode processing fails
        text = text.encode('ascii', 'ignore').decode('ascii')
    
    return text.strip()
