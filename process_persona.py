"""
Challenge 1B: Persona-Driven Document Intelligence
Professional implementation for Adobe Hackathon 2025

This module processes multiple PDF collections and extracts relevant content
based on specific personas and job-to-be-done scenarios.
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
    Main analyzer for persona-driven document intelligence.
    Processes document collections and extracts relevant sections.
    """
    
    def __init__(self):
        self.section_extractor = PDFSectionExtractor()
        self.semantic_analyzer = SemanticAnalyzer()
        self.query_builder = PersonaQueryBuilder()
        
    def process_collection(self, input_file: Path, pdf_directory: Path) -> Dict:
        """
        Process a document collection based on input configuration.
        
        Args:
            input_file: Path to challenge1b_input.json
            pdf_directory: Directory containing PDF files
            
        Returns:
            Analysis results in required output format
        """
        start_time = time.time()
        
        try:
            # Load input configuration
            with open(input_file, 'r', encoding='utf-8') as f:
                input_config = json.load(f)
            
            logger.info(f"Processing collection: {input_config.get('challenge_info', {}).get('challenge_id', 'unknown')}")
            
            # Extract components
            documents_info = input_config.get("documents", [])
            persona = input_config.get("persona", {}).get("role", "")
            job_to_be_done = input_config.get("job_to_be_done", {}).get("task", "")
            
            # Validate inputs
            if not documents_info:
                raise ValueError("No documents specified in input configuration")
            if not persona:
                raise ValueError("No persona specified in input configuration")
            if not job_to_be_done:
                raise ValueError("No job-to-be-done specified in input configuration")
            
            # Build semantic query
            query = self.query_builder.build_query(persona, job_to_be_done)
            logger.info(f"Built query: {query[:100]}...")
            
            # Process each document
            all_sections = []
            processed_docs = []
            
            for doc_info in documents_info:
                filename = doc_info.get("filename", "")
                pdf_path = pdf_directory / filename
                
                if not pdf_path.exists():
                    logger.warning(f"PDF file not found: {pdf_path}")
                    continue
                
                logger.info(f"Processing document: {filename}")
                doc_data = self.section_extractor.extract_document_sections(pdf_path)
                
                # Add document info to sections
                for section in doc_data.get("sections", []):
                    section["source_document"] = filename
                    all_sections.append(section)
                
                processed_docs.append(filename)
            
            if not all_sections:
                logger.warning("No sections extracted from any documents")
                return self._create_empty_output(input_config, processed_docs)
            
            logger.info(f"Extracted {len(all_sections)} sections from {len(processed_docs)} documents")
            
            # Rank sections by relevance
            ranked_sections = self.semantic_analyzer.rank_sections(
                query, all_sections, top_k=10
            )
            
            # Generate extracted sections output
            extracted_sections = []
            for rank, (section, score) in enumerate(ranked_sections, 1):
                extracted_sections.append({
                    "document": section["source_document"],
                    "section_title": section["title"],
                    "importance_rank": rank,
                    "page_number": section.get("start_page", 1)
                })
            
            # Generate subsection analysis
            subsection_analysis = []
            for section, score in ranked_sections[:5]:  # Top 5 sections for detailed analysis
                subsections = self.semantic_analyzer.extract_relevant_subsections(
                    section, query, max_subsections=2
                )
                
                for subsection in subsections:
                    subsection_analysis.append({
                        "document": section["source_document"],
                        "refined_text": subsection["text"][:500],  # Limit text length
                        "page_number": subsection["page"]
                    })
            
            # Create final output
            result = {
                "metadata": {
                    "input_documents": processed_docs,
                    "persona": persona,
                    "job_to_be_done": job_to_be_done,
                    "processing_timestamp": datetime.now().isoformat()
                },
                "extracted_sections": extracted_sections,
                "subsection_analysis": subsection_analysis
            }
            
            processing_time = time.time() - start_time
            logger.info(f"Collection processed in {processing_time:.2f}s - "
                       f"Found {len(extracted_sections)} relevant sections")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing collection: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return self._create_empty_output(
                input_config if 'input_config' in locals() else {}, 
                []
            )
    
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
    
    # Get input and output directories
    input_dir = Path("/app/input")
    output_dir = Path("/app/output")
    
    # Validate directories
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        sys.exit(1)
    
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
