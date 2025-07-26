"""
Challenge 1B: Persona-Driven Document Intelligence
Implementation of the complete multi-stage pipeline as specified in the solution approach.

PIPELINE STAGES:
1. Data Ingestion & Preprocessing - PDF text extraction with heading/section detection
2. Persona-Job Query Construction - Extract keywords and build persona-driven query
3. Semantic Representation & Ranking - Use all-MiniLM-L6-v2 embeddings + cosine similarity
4. Section Prioritization - Rank and select top-k sections by relevance score
5. Subsection Extraction & Refinement - Granular chunking with re-scoring
6. JSON Output Formatting - Schema-compliant output with metadata

TECH STACK: PyMuPDF for PDF parsing, SentenceTransformers (all-MiniLM-L6-v2) for semantic 
embeddings, cosine similarity for relevance scoring, CPU-only processing under 1GB model limit.
"""

import os
import json
import sys
import time
import logging
from pathlib import Path
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import traceback

# Import our custom modules
from pdf_section_extractor import PDFSectionExtractor, clean_text
from semantic_analyzer import SemanticAnalyzer, PersonaQueryBuilder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PersonaDrivenAnalyzer:
    """
    Multi-stage pipeline implementation for persona-driven document intelligence.
    
    STAGES:
    1. Data Ingestion & Preprocessing - Extract PDF text and detect sections/headings
    2. Persona-Job Query Construction - Build semantic query from persona + task
    3. Semantic Representation & Ranking - Embed sections and rank by cosine similarity  
    4. Section Prioritization - Select top-k relevant sections across documents
    5. Subsection Extraction & Refinement - Extract and re-score granular chunks
    6. JSON Output Formatting - Generate schema-compliant output
    """
    
    def __init__(self):
        self.section_extractor = PDFSectionExtractor()
        self.semantic_analyzer = SemanticAnalyzer()
        self.query_builder = PersonaQueryBuilder()
        logger.info("PersonaDrivenAnalyzer initialized - multi-stage pipeline ready")
        
    def process_collection(self, input_file: Path, pdf_directory: Path) -> Dict:
        """
        Execute complete multi-stage pipeline for persona-driven analysis.
        
        Implements the exact approach described in the solution specification:
        - PDF text extraction with heading detection using PyMuPDF
        - Persona+job query construction with keyword extraction
        - Semantic embedding with all-MiniLM-L6-v2 (≤1GB model constraint)
        - Cosine similarity scoring for relevance ranking
        - Top-k section selection with importance ranking
        - Granular subsection extraction with refinement
        
        Args:
            input_file: Path to challenge1b_input.json
            pdf_directory: Directory containing PDF files
            
        Returns:
            Schema-compliant output matching specification exactly
        """
        start_time = time.time()
        
        try:
            logger.info("STAGE 1: Data Ingestion & Preprocessing")
            
            # Parse input configuration
            with open(input_file, 'r', encoding='utf-8') as f:
                input_config = json.load(f)
            
            persona = input_config.get("persona", {}).get("role", "")
            job_to_be_done = input_config.get("job_to_be_done", {}).get("task", "")
            
            logger.info(f"Persona: '{persona}', Job: '{job_to_be_done}'")
            
            # Extract text from all PDFs with heading/section detection
            pdf_files = list(pdf_directory.glob("*.pdf"))
            all_sections = []
            processed_docs = []
            
            logger.info(f"Processing {len(pdf_files)} PDFs with PyMuPDF text extraction")
            
            for pdf_path in pdf_files:
                try:
                    logger.info(f"Extracting from: {pdf_path.name}")
                    
                    # Use PyMuPDF for high-quality text extraction with section detection
                    doc_data = self.section_extractor.extract_document_sections(pdf_path)
                    
                    if doc_data and doc_data.get("sections"):
                        # Structure sections for semantic analysis
                        for section in doc_data["sections"]:
                            section_record = {
                                "source_document": pdf_path.name,
                                "title": section.get("title", "Document Content"),
                                "content": section.get("content", ""),
                                "start_page": section.get("start_page", 1),
                                "end_page": section.get("end_page", 1)
                            }
                            all_sections.append(section_record)
                        
                        processed_docs.append(pdf_path.name)
                        logger.info(f"Extracted {len(doc_data['sections'])} sections from {pdf_path.name}")
                    else:
                        logger.warning(f"No sections extracted from {pdf_path.name}")
                        
                except Exception as e:
                    logger.error(f"Failed to process {pdf_path.name}: {e}")
                    continue
            
            if not all_sections:
                logger.warning("No sections extracted from any documents")
                return self._create_empty_output(input_config, processed_docs)
            
            logger.info(f"STAGE 1 COMPLETE: {len(all_sections)} sections extracted from {len(processed_docs)} documents")
            
            # STAGE 2: Persona-Job Query Construction
            logger.info("STAGE 2: Persona-Job Query Construction")
            
            # Build persona-driven query with keyword extraction
            persona_query = self.query_builder.build_query(persona, job_to_be_done)
            logger.info(f"Constructed persona query: '{persona_query}'")
            
            # STAGE 3: Semantic Representation & Ranking
            logger.info("STAGE 3: Semantic Representation & Ranking")
            logger.info("Using all-MiniLM-L6-v2 embeddings with cosine similarity scoring")
            
            # Rank sections by semantic relevance to persona query
            ranked_sections = self.semantic_analyzer.rank_sections(
                persona_query, all_sections, top_k=10
            )
            
            logger.info(f"Semantic ranking complete. Top section score: {ranked_sections[0][1]:.3f}")
            
            # STAGE 4: Section Prioritization
            logger.info("STAGE 4: Section Prioritization")
            
            # Smart section selection with diversity
            selected_sections = self._smart_section_selection(ranked_sections, 5)
            
            # Select top-k sections with importance ranking
            extracted_sections = []
            for rank, (section, score) in enumerate(selected_sections, 1):
                extracted_sections.append({
                    "document": section["source_document"],
                    "section_title": section["title"],
                    "importance_rank": rank,
                    "page_number": section.get("start_page", 1)
                })
            
            logger.info(f"Selected top {len(extracted_sections)} sections for output")
            
            # STAGE 5: Subsection Extraction & Refinement
            logger.info("STAGE 5: Subsection Extraction & Refinement")
            
            # STAGE 5: Subsection Extraction & Refinement
            logger.info("STAGE 5: Subsection Extraction & Refinement")
            
            # Extract granular chunks from top sections and re-score
            subsection_analysis = []
            for section, score in selected_sections:  # Use selected sections instead of ranked_sections[:5]
                
                # Extract relevant subsections with granular chunking
                subsections = self.semantic_analyzer.extract_relevant_subsections(
                    section, persona_query, max_subsections=2
                )
                
                for subsection in subsections:
                    subsection_analysis.append({
                        "document": section["source_document"],
                        "refined_text": subsection["text"][:500],  # Truncate for readability
                        "page_number": subsection["page"]
                    })
            
            logger.info(f"Generated {len(subsection_analysis)} refined subsection snippets")
            
            # STAGE 6: JSON Output Formatting
            logger.info("STAGE 6: JSON Output Formatting")
            
            # Assemble schema-compliant output
            result = {
                "metadata": {
                    "input_documents": processed_docs,
                    "persona": persona,
                    "job_to_be_done": job_to_be_done,
                    "processing_timestamp": datetime.now().isoformat(),
                    "total_sections_analyzed": len(all_sections),
                    "semantic_model": "all-MiniLM-L6-v2",
                    "processing_time_seconds": round(time.time() - start_time, 2)
                },
                "extracted_sections": extracted_sections,
                "subsection_analysis": subsection_analysis
            }
            
            processing_time = time.time() - start_time
            logger.info(f"PIPELINE COMPLETE in {processing_time:.2f}s:")
            logger.info(f"  - Documents processed: {len(processed_docs)}")
            logger.info(f"  - Sections analyzed: {len(all_sections)}")
            logger.info(f"  - Top sections selected: {len(extracted_sections)}")
            logger.info(f"  - Refined subsections: {len(subsection_analysis)}")
            
            return result
            
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            logger.error(traceback.format_exc())
            return self._create_empty_output(
                input_config if 'input_config' in locals() else {}, 
                []
            )
    
    def _smart_section_selection(self, ranked_sections: List[Tuple[Dict, float]], max_sections: int) -> List[Tuple[Dict, float]]:
        """
        Smart section selection ensuring document diversity and avoiding redundancy.
        """
        if len(ranked_sections) <= max_sections:
            return ranked_sections
        
        selected = []
        doc_counts = {}
        
        # First pass: select highest scoring sections with document diversity
        for section, score in ranked_sections:
            if len(selected) >= max_sections:
                break
                
            doc_name = section["source_document"]
            current_count = doc_counts.get(doc_name, 0)
            
            # Allow up to 2 sections per document, but prefer diversity
            if current_count < 2 or len(selected) < max_sections // 2:
                selected.append((section, score))
                doc_counts[doc_name] = current_count + 1
        
        # Second pass: fill remaining slots with best remaining sections
        remaining_slots = max_sections - len(selected)
        if remaining_slots > 0:
            used_indices = set()
            for sel_section, _ in selected:
                for i, (section, score) in enumerate(ranked_sections):
                    if section == sel_section:
                        used_indices.add(i)
                        break
            
            for i, (section, score) in enumerate(ranked_sections):
                if len(selected) >= max_sections:
                    break
                if i not in used_indices:
                    selected.append((section, score))
        
        return selected[:max_sections]
    
    def process_documents(self, pdfs_directory: str, persona: str, job_to_be_done: str) -> Dict:
        """
        Alternative entry point for direct document processing.
        Creates temporary input config and processes using main pipeline.
        """
        # Create temporary input configuration
        temp_config = {
            "persona": {"role": persona},
            "job_to_be_done": {"task": job_to_be_done}
        }
        
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(temp_config, f)
            temp_file_path = f.name
        
        try:
            result = self.process_collection(Path(temp_file_path), Path(pdfs_directory))
            return result
        finally:
            try:
                os.unlink(temp_file_path)
            except:
                pass
    
    def _create_empty_output(self, input_config: Dict, processed_docs: List[str]) -> Dict:
        """Create empty output structure for failed processing."""
        return {
            "metadata": {
                "input_documents": processed_docs,
                "persona": input_config.get("persona", {}).get("role", "Unknown"),
                "job_to_be_done": input_config.get("job_to_be_done", {}).get("task", "Unknown"),
                "processing_timestamp": datetime.now().isoformat()
            },
            "extracted_sections": [],
            "subsection_analysis": []
        }


def process_collection_worker(args: Tuple[Path, Path, Path]) -> Tuple[str, bool]:
    """
    Worker function for multiprocessing collection processing.
    
    Args:
        args: Tuple of (collection_dir, input_file, output_file)
        
    Returns:
        Tuple of (result_message, success_flag)
    """
    collection_dir, input_file, output_file = args
    
    try:
        analyzer = PersonaDrivenAnalyzer()
        
        # Find PDF directory
        pdf_dir = collection_dir / "PDFs"
        if not pdf_dir.exists():
            pdf_dir = collection_dir  # PDFs might be in collection root
        
        # Process collection
        result = analyzer.process_collection(input_file, pdf_dir)
        
        # Write output
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        sections_count = len(result.get("extracted_sections", []))
        return f"✓ {collection_dir.name}: {sections_count} sections extracted", True
        
    except Exception as e:
        logger.error(f"Worker error processing {collection_dir.name}: {e}")
        return f"✗ Failed: {collection_dir.name} - {str(e)}", False


def process_persona_driven_analysis():
    """
    Main processing function for Challenge 1B.
    Processes all collections in the input directory.
    """
    start_time = time.time()
    
    # Get command line arguments
    collection_name = sys.argv[1] if len(sys.argv) > 1 else None
    
    # Get input and output directories based on current working directory
    current_dir = Path.cwd()
    
    if collection_name:
        # Process specific collection
        input_dir = current_dir / collection_name
        output_dir = current_dir / collection_name
    else:
        # Look for Docker-style paths first, then local paths
        if Path("/app/input").exists():
            input_dir = Path("/app/input")
            output_dir = Path("/app/output")
        else:
            # Local development - look for collections in current directory
            input_dir = current_dir
            output_dir = current_dir
    
    # Validate directories
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        sys.exit(1)
    
    logger.info(f"Processing from directory: {input_dir}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if we're processing a single collection directory
    input_file = input_dir / "challenge1b_input.json"
    
    if input_file.exists():
        # Single collection processing
        logger.info("Processing single collection")
        output_file = output_dir / "challenge1b_output.json"
        
        try:
            analyzer = PersonaDrivenAnalyzer()
            pdf_dir = input_dir / "PDFs"
            if not pdf_dir.exists():
                pdf_dir = input_dir
            
            result = analyzer.process_collection(input_file, pdf_dir)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            sections_count = len(result.get("extracted_sections", []))
            logger.info(f"✓ Collection processed: {sections_count} sections extracted")
            
        except Exception as e:
            logger.error(f"✗ Failed to process collection: {str(e)}")
    
    else:
        # Multiple collections - find all collections (directories with challenge1b_input.json)
        collections = []
        for item in input_dir.iterdir():
            if item.is_dir():
                collection_input = item / "challenge1b_input.json"
                if collection_input.exists():
                    collections.append(item)
        
        if not collections:
            logger.warning("No valid collections found in input directory")
            logger.info("Looking for challenge1b_input.json files in subdirectories...")
            return
        
        logger.info(f"Found {len(collections)} collections to process")
        
        # Process collections
        if len(collections) == 1:
            # Single collection - no multiprocessing overhead
            collection_dir = collections[0]
            input_file = collection_dir / "challenge1b_input.json"
            output_file = output_dir / "challenge1b_output.json"
            
            try:
                analyzer = PersonaDrivenAnalyzer()
                pdf_dir = collection_dir / "PDFs"
                if not pdf_dir.exists():
                    pdf_dir = collection_dir
                
                result = analyzer.process_collection(input_file, pdf_dir)
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                
                sections_count = len(result.get("extracted_sections", []))
                logger.info(f"✓ {collection_dir.name}: {sections_count} sections extracted")
                
            except Exception as e:
                logger.error(f"✗ Failed: {collection_dir.name} - {str(e)}")
        
        else:
            # Multiple collections - use multiprocessing
            num_processes = min(cpu_count(), len(collections), 3)  # Cap at 3 for memory
            logger.info(f"Using {num_processes} processes for parallel processing")
            
            # Prepare worker arguments
            worker_args = []
            for i, collection_dir in enumerate(collections):
                collection_input = collection_dir / "challenge1b_input.json"
                collection_output = output_dir / f"challenge1b_output_{i+1}.json"
                worker_args.append((collection_dir, collection_input, collection_output))
            
            # Process in parallel
            with Pool(processes=num_processes) as pool:
                results = pool.map(process_collection_worker, worker_args)
            
            # Log results
            success_count = sum(1 for _, success in results if success)
            for message, _ in results:
                logger.info(message)
            
            logger.info(f"Successfully processed {success_count}/{len(collections)} collections")
    
    total_time = time.time() - start_time
    logger.info(f"Total processing time: {total_time:.2f}s")


if __name__ == "__main__":
    print("Starting persona-driven document analysis...")
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    
    try:
        process_persona_driven_analysis()
        print("✓ Document analysis completed successfully")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"✗ Document analysis failed: {e}")
        sys.exit(1)
