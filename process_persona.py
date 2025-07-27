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
import re
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
        logger.info("Initializing PersonaDrivenAnalyzer with multi-stage pipeline")
        logger.info("Stage 1: PDF Section Extraction with document triage")
        self.section_extractor = PDFSectionExtractor()
        
        logger.info("Stage 2: Semantic Analysis with three-pillar framework")
        self.semantic_analyzer = SemanticAnalyzer()
        
        # Load semantic analyzer extensions
        try:
            from semantic_analyzer_extensions import integrate_with_semantic_analyzer
            logger.info("Loading semantic analyzer extensions for multi-document intelligence")
            self.semantic_analyzer = integrate_with_semantic_analyzer(self.semantic_analyzer)
            logger.info("Successfully loaded multi-document intelligence extensions")
        except Exception as e:
            logger.warning(f"Failed to load semantic analyzer extensions: {e}")
        
        logger.info("Stage 3: Persona-Driven Query Building")
        self.query_builder = PersonaQueryBuilder()
        
    def process_collection(self, input_file: Path, pdf_directory: Path) -> Dict:
        """
        Process a document collection based on input configuration.
        
        Implements a tiered processing approach:
        1. Document triage - Identify document types (text/image/hybrid)
        2. Adaptive extraction - Use optimal strategy based on document characteristics
        3. Schema validation - Ensure output conforms to expected format
        
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
            document_types = {}
            
            for doc_info in documents_info:
                filename = doc_info.get("filename", "")
                pdf_path = pdf_directory / filename
                
                if not pdf_path.exists():
                    logger.warning(f"PDF file not found: {pdf_path}")
                    continue
                
                logger.info(f"Processing document: {filename}")
                
                # Extract document sections with automatic triage
                doc_data = self.section_extractor.extract_document_sections(pdf_path)
                document_types[filename] = getattr(doc_data, 'document_type', 'unknown')
                
                # Add document info to sections and ensure titles are properly set
                for section in doc_data.get("sections", []):
                    section["source_document"] = filename
                    
                    # Ensure section has a proper title - fix the issue with missing titles
                    if not section.get("title") or len(section.get("title", "").strip()) < 3:
                        # Extract a title from the first 50 chars of content if available
                        if section.get("content"):
                            # Get first line or first 50 chars
                            first_line = section["content"].split('\n')[0][:100]
                            section["title"] = first_line.strip()
                        else:
                            # Use document name as fallback
                            section["title"] = f"Content from {filename}"
                    
                    all_sections.append(section)
                
                processed_docs.append(filename)
            
            logger.info(f"Document types detected: {document_types}")
            
            if not all_sections:
                logger.warning("No sections extracted from any documents")
                return self._create_empty_output(input_config, processed_docs)
            
            logger.info(f"Extracted {len(all_sections)} sections from {len(processed_docs)} documents")
            
            # Organize sections by document for multi-document intelligence
            docs_with_sections = {}
            for section in all_sections:
                doc_name = section["source_document"]
                if doc_name not in docs_with_sections:
                    docs_with_sections[doc_name] = {
                        "title": doc_name,
                        "sections": []
                    }
                docs_with_sections[doc_name]["sections"].append(section)
            
            # Try to use cross-document intelligence if available
            cross_doc_connections = None
            if hasattr(self.semantic_analyzer, 'identify_connections'):
                try:
                    logger.info("Identifying cross-document connections")
                    cross_doc_connections = self.semantic_analyzer.identify_connections(
                        list(docs_with_sections.values())
                    )
                    logger.info(f"Found connections between {len(cross_doc_connections.get('connections', {}))} sections")
                except Exception as e:
                    logger.warning(f"Failed to identify cross-document connections: {e}")
            
            # Extract content from document collection if multi-doc intelligence is available
            multi_doc_content = None
            if hasattr(self.semantic_analyzer, 'extract_from_document_collection'):
                try:
                    logger.info("Using multi-document intelligence for extraction")
                    multi_doc_content = self.semantic_analyzer.extract_from_document_collection(
                        list(docs_with_sections.values()), query, chars_limit=10000
                    )
                    logger.info(f"Extracted {len(multi_doc_content)} chars using multi-document intelligence")
                except Exception as e:
                    logger.warning(f"Failed to extract using multi-document intelligence: {e}")
            
            # Rank sections by relevance (fallback approach)
            ranked_sections = self.semantic_analyzer.rank_sections(
                query, all_sections, top_k=10
            )
            
            # Generate extracted sections output
            extracted_sections = []
            for rank, (section, score) in enumerate(ranked_sections, 1):
                section_entry = {
                    "document": section["source_document"],
                    "section_title": section["title"],
                    "importance_rank": rank,
                    "page_number": section.get("start_page", 1)
                }
                
                # Add cross-document connection information if available
                if cross_doc_connections and "connections" in cross_doc_connections:
                    # Create a unique section ID that matches our connection mapping
                    doc_id = section["source_document"]
                    section_idx = next((i for i, s in enumerate(docs_with_sections[doc_id]["sections"]) 
                                    if s.get("title") == section["title"]), None)
                    
                    if section_idx is not None:
                        section_id = f"{doc_id}_section_{section_idx}"
                        connections = cross_doc_connections["connections"].get(section_id, [])
                        
                        if connections:
                            top_connection = connections[0]
                            section_entry["related_to"] = {
                                "document": top_connection["doc_title"],
                                "section": top_connection["title"],
                                "relevance": round(top_connection["score"], 2)
                            }
                
                extracted_sections.append(section_entry)
            
            # Generate subsection analysis with enhanced context and formatting
            subsection_analysis = []
            
            # If multi-document content is available, use it first
            if multi_doc_content:
                logger.info("Using multi-document intelligence results for subsection analysis")
                
                # Split the multi-document content by separators and create subsections
                parts = multi_doc_content.split("\n\n---\n\n")
                
                for part in parts:
                    if not part.strip():
                        continue
                        
                    # Extract document name from the content if possible
                    doc_name = processed_docs[0]  # Default to first document
                    page_number = 1
                    
                    # Try to extract document and page information
                    doc_match = re.search(r'From document:\s*([^\n]+)', part)
                    if doc_match:
                        potential_doc = doc_match.group(1).strip()
                        # Find closest matching document name
                        for doc in processed_docs:
                            if potential_doc in doc or doc in potential_doc:
                                doc_name = doc
                                break
                    
                    # Try to extract page number
                    page_match = re.search(r'\[Page\s+(\d+)\]', part)
                    if page_match:
                        page_number = int(page_match.group(1))
                    
                    # Clean up the text
                    refined_text = clean_text(part)
                    
                    subsection_analysis.append({
                        "document": doc_name,
                        "refined_text": refined_text[:1000],
                        "page_number": page_number
                    })
                
                # If we have enough subsections, skip the standard processing
                if len(subsection_analysis) >= 5:
                    logger.info(f"Using {len(subsection_analysis)} subsections from multi-document intelligence")
            
            # If we don't have enough subsections from multi-document processing, use standard approach
            if not subsection_analysis or len(subsection_analysis) < 3:
                logger.info("Using standard subsection extraction as fallback or supplement")
                
                # Determine optimal number of subsections based on document count
                doc_count = len(processed_docs)
                sections_per_doc = max(1, 10 // doc_count)  # Distribute subsections evenly
                
                # Process top sections for detailed analysis using tiered approach
                top_sections = ranked_sections[:min(5, len(ranked_sections))]
                
                # Set initial extraction tier based on document complexity
                complex_docs = [doc for doc, doc_type in document_types.items() 
                               if doc_type in ['mixed_content', 'image_based', 'table_document']]
                
                # Start with tier 1 for simple docs, tier 2 for complex docs
                initial_tier = 2 if complex_docs else 1
                self.semantic_analyzer.extraction_tier = initial_tier
                
                logger.info(f"Starting extraction at tier {initial_tier} based on document complexity")
            
            for section, score in top_sections:
                # Extract more subsections from high-scoring sections
                # Adjust the number of subsections based on relevance score
                # Higher scores get proportionally more subsections
                max_subsections = max(2, int(sections_per_doc * score * 2))
                
                # Apply adaptive scaling for important sections
                if score > 0.8:  # Highly relevant sections
                    max_subsections = min(max_subsections + 2, 8)  # Add more but cap at 8
                
                logger.info(f"Extracting up to {max_subsections} subsections from section: {section['title']}")
                
                # For very important sections, ensure we use tier 2
                if score > 0.9 and self.semantic_analyzer.extraction_tier == 1:
                    logger.info("High-importance section detected, using tier 2 extraction")
                    self.semantic_analyzer.extraction_tier = 2
                
                subsections = self.semantic_analyzer.extract_relevant_subsections(
                    section, query, max_subsections=max_subsections
                )
                
                for subsection in subsections:
                    # Ensure text is properly cleaned and formatted
                    refined_text = clean_text(subsection["text"])
                    
                    # Verify text isn't too short or truncated inappropriately
                    if len(refined_text) < 50:
                        logger.debug(f"Skipping short subsection: {len(refined_text)} chars")
                        continue
                    
                    # Enhance context by including section title in ambiguous cases
                    if not any(word in refined_text.lower() for word in section['title'].lower().split()):
                        context_prefix = f"{section['title']}: "
                        refined_text = context_prefix + refined_text
                    
                    subsection_analysis.append({
                        "document": section["source_document"],
                        "refined_text": refined_text[:1000],  # Allow longer excerpts but respect max length
                        "page_number": subsection["page"]
                    })
                    
                # Ensure we get at least some content from each highly relevant section
                if score > 0.7 and not subsections:
                    logger.info(f"Adding fallback subsection for high-scoring section: {section['title']}")
                    # Create a fallback subsection with the beginning of the section content
                    content = section.get("content", "")
                    if content:
                        refined_text = clean_text(content[:1500])  # Take beginning of section
                        subsection_analysis.append({
                            "document": section["source_document"],
                            "refined_text": f"{section['title']}: {refined_text[:1000]}",
                            "page_number": section.get("start_page", 1)
                        })
            
                # Validate and ensure schema compliance
            
            # Ensure we have at least some extracted sections
            if not extracted_sections:
                logger.warning("No extracted sections found - using fallback approach")
                # Create at least one section entry per document as fallback
                for doc_name in processed_docs:
                    extracted_sections.append({
                        "document": doc_name,
                        "section_title": f"Content from {doc_name}",
                        "importance_rank": len(extracted_sections) + 1,
                        "page_number": 1
                    })
            
            # Ensure we have subsections for analysis
            if not subsection_analysis:
                logger.warning("No subsection analysis found - using fallback approach")
                # Create at least one subsection entry per document as fallback
                for doc_name in processed_docs:
                    subsection_analysis.append({
                        "document": doc_name,
                        "refined_text": f"No relevant content extracted from {doc_name}. Please review the document manually.",
                        "page_number": 1
                    })
                
            # Create final output with schema validation
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
            
            # Validate output schema
            self._validate_output_schema(result)
            
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
        
    def _validate_output_schema(self, result: Dict) -> None:
        """Validate output schema structure and types."""
        # Ensure metadata has all required fields
        metadata = result.get("metadata", {})
        if not isinstance(metadata.get("input_documents"), list):
            logger.warning("Fixing metadata.input_documents schema")
            metadata["input_documents"] = []
            
        if not isinstance(metadata.get("persona"), str):
            logger.warning("Fixing metadata.persona schema")
            metadata["persona"] = "Unknown"
            
        if not isinstance(metadata.get("job_to_be_done"), str):
            logger.warning("Fixing metadata.job_to_be_done schema")
            metadata["job_to_be_done"] = "Unknown"
            
        if not isinstance(metadata.get("processing_timestamp"), str):
            logger.warning("Fixing metadata.processing_timestamp schema")
            metadata["processing_timestamp"] = datetime.now().isoformat()
            
        # Validate extracted sections
        sections = result.get("extracted_sections", [])
        for i, section in enumerate(sections):
            if not isinstance(section.get("document"), str):
                logger.warning(f"Fixing section[{i}].document schema")
                section["document"] = f"unknown_doc_{i}.pdf"
                
            if not isinstance(section.get("section_title"), str):
                logger.warning(f"Fixing section[{i}].section_title schema")
                section["section_title"] = f"Untitled Section {i+1}"
                
            if not isinstance(section.get("importance_rank"), int):
                logger.warning(f"Fixing section[{i}].importance_rank schema")
                section["importance_rank"] = i + 1
                
            if not isinstance(section.get("page_number"), int):
                logger.warning(f"Fixing section[{i}].page_number schema")
                section["page_number"] = 1
                
        # Validate subsection analysis
        subsections = result.get("subsection_analysis", [])
        for i, subsection in enumerate(subsections):
            if not isinstance(subsection.get("document"), str):
                logger.warning(f"Fixing subsection[{i}].document schema")
                subsection["document"] = f"unknown_doc_{i}.pdf"
                
            if not isinstance(subsection.get("refined_text"), str):
                logger.warning(f"Fixing subsection[{i}].refined_text schema")
                subsection["refined_text"] = "No text available"
                
            if not isinstance(subsection.get("page_number"), int):
                logger.warning(f"Fixing subsection[{i}].page_number schema")
                subsection["page_number"] = 1


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
            logger.error(f"PDF directory not found in {collection_dir}")
            return (f"Error: PDF directory not found in {collection_dir}", False)
        
        # Process collection
        start_time = time.time()
        result = analyzer.process_collection(input_file, pdf_dir)
        processing_time = time.time() - start_time
        
        # Write output
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)
        
        # Log detailed statistics
        sections_count = len(result.get("extracted_sections", []))
        subsections_count = len(result.get("subsection_analysis", []))
        
        # Get documents and page counts
        documents = result.get("metadata", {}).get("input_documents", [])
        doc_count = len(documents)
        
        logger.info(f"Collection {collection_dir.name} processed successfully:")
        logger.info(f" - Processing time: {processing_time:.2f}s")
        logger.info(f" - Documents processed: {doc_count}")
        logger.info(f" - Sections extracted: {sections_count}")
        logger.info(f" - Subsections analyzed: {subsections_count}")
        
        # Check for potential issues
        if sections_count < 3 and doc_count > 0:
            logger.warning(f"Low section count ({sections_count}) for {doc_count} documents")
        if subsections_count < sections_count:
            logger.warning(f"Low subsection count ({subsections_count}) compared to sections ({sections_count})")
        
        return (f"Successfully processed {collection_dir.name}: {doc_count} docs, {sections_count} sections, {processing_time:.2f}s", True)
        
    except Exception as e:
        logger.error(f"Worker error processing {collection_dir.name}: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return (f"Error processing {collection_dir.name}: {str(e)}", False)


def process_persona_driven_analysis():
    """
    Main processing function for Challenge 1B.
    Processes all collections in the input directory.
    """
    start_time = time.time()
    
    # Get input and output directories
    input_dir = Path("/app/input")
    output_dir = Path("/app/output")
    
    # For development/testing environment
    if not input_dir.exists():
        # Try current directory as fallback
        current_dir = Path.cwd()
        if (current_dir / "Collection 1").exists():
            input_dir = current_dir
            output_dir = current_dir / "test_output"
            logger.info(f"Using current directory as input: {input_dir}")
        else:
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
                logger.warning(f"PDFs directory not found in root, checking for PDFs in input dir")
                # Check if PDFs exist directly in the input directory
                pdf_files = list(input_dir.glob("*.pdf"))
                if pdf_files:
                    pdf_dir = input_dir
                    logger.info(f"Using PDFs from input directory: {len(pdf_files)} files found")
                else:
                    logger.error("No PDFs directory or PDF files found")
                    sys.exit(1)
            
            logger.info(f"Processing input from {input_file}")
            logger.info(f"Using PDFs from {pdf_dir}")
            
            result = analyzer.process_collection(input_file, pdf_dir)
            
            # Write output
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            sections_count = len(result.get("extracted_sections", []))
            subsections_count = len(result.get("subsection_analysis", []))
            logger.info(f"✓ Collection processed: {sections_count} sections, {subsections_count} subsections extracted")
            
        except Exception as e:
            logger.error(f"✗ Failed to process collection: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            sys.exit(1)
    
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
            # Try one level deeper
            for item in input_dir.iterdir():
                if item.is_dir():
                    for subitem in item.iterdir():
                        if subitem.is_dir():
                            collection_input = subitem / "challenge1b_input.json"
                            if collection_input.exists():
                                collections.append(subitem)
            
            if not collections:
                logger.error("No collections found. Each collection should contain challenge1b_input.json file.")
                sys.exit(1)
        
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
                    # Try to find PDFs in the collection directory
                    pdf_files = list(collection_dir.glob("*.pdf"))
                    if pdf_files:
                        pdf_dir = collection_dir
                        logger.info(f"Using PDFs from collection directory: {len(pdf_files)} files found")
                    else:
                        logger.error(f"No PDFs directory or PDF files found in {collection_dir}")
                        sys.exit(1)
                
                logger.info(f"Processing collection: {collection_dir.name}")
                result = analyzer.process_collection(input_file, pdf_dir)
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                
                sections_count = len(result.get("extracted_sections", []))
                subsections_count = len(result.get("subsection_analysis", []))
                logger.info(f"✓ {collection_dir.name}: {sections_count} sections, {subsections_count} subsections extracted")
                
            except Exception as e:
                logger.error(f"✗ Failed: {collection_dir.name} - {str(e)}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                sys.exit(1)
        
        else:
            # Multiple collections - use multiprocessing
            # Balance memory usage vs performance based on collection count
            num_processes = min(cpu_count(), len(collections))
            
            # For many collections or memory-constrained environments, limit concurrency
            if num_processes > 3 or len(collections) > 5:
                num_processes = min(3, num_processes)  # Cap at 3 for memory
                
            logger.info(f"Using {num_processes} processes for parallel processing")
            
            # Prepare worker arguments
            worker_args = []
            for collection_dir in collections:
                collection_input = collection_dir / "challenge1b_input.json"
                collection_output = output_dir / f"{collection_dir.name}_output.json"
                worker_args.append((collection_dir, collection_input, collection_output))
            
            # Process in parallel
            try:
                with Pool(processes=num_processes) as pool:
                    results = pool.map(process_collection_worker, worker_args)
                
                # Log results
                success_count = sum(1 for _, success in results if success)
                for message, success in results:
                    if success:
                        logger.info(f"✓ {message}")
                    else:
                        logger.error(f"✗ {message}")
                
                logger.info(f"Successfully processed {success_count}/{len(collections)} collections")
                
                if success_count == 0:
                    logger.error("All collections failed. Check logs for details.")
                    sys.exit(1)
                    
            except Exception as e:
                logger.error(f"Error in parallel processing: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                sys.exit(1)
    
    total_time = time.time() - start_time
    logger.info(f"✓ Total processing time: {total_time:.2f}s")
    logger.info(f"✓ Results saved to {output_dir}")


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
