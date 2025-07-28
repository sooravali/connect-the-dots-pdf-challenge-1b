# Challenge 1B: Multi-Document Analysis with Persona-Based Intelligence
> Intelligent document analysis that connects what matters for the user who matters

## Table of Contents

| # | Section |
|---|---------|
| 1 | [Project Overview](#project-overview) |
| 2 | [Technical Architecture](#technical-architecture) |
| 3 | [Key Features](#key-features) |
| 4 | [Installation & Usage](#installation--usage) |
| 5 | [Configuration](#configuration) |
| 6 | [Performance Considerations](#performance-considerations) |
| 7 | [Sample Collections](#sample-collections) |
| 8 | [Approach Explanation](approach_explanation.md) |

## Project Overview

The solution addresses the "Connect What Matters — For the User Who Matters" challenge by implementing a persona-driven document intelligence system. 

This solution implements an advanced document intelligence system capable of analyzing multiple PDF collections based on specific personas and task requirements. The system extracts, ranks, and presents the most relevant content across document sets.
It processes collections of PDFs and extracts the most relevant information based on:

- **Persona Definition**: Specific role with expertise and focus areas
- **Job-to-be-Done**: Concrete task the persona needs to accomplish

The system handles diverse document types, formats, and content structures while maintaining high relevance accuracy and performance.

> [Read our detailed approach explanation here](approach_explanation.md)

## Technical Architecture

The implementation follows a modular design with three primary components:

1. **PDF Section Extractor** (`pdf_section_extractor.py`)
   - Document type classification (text-based, image-based, hybrid)
   - Section extraction with hierarchical structure detection
   - Content cleaning and normalization

2. **Semantic Analyzer** (`semantic_analyzer.py`)
   - Sentence embeddings for semantic similarity analysis
   - TF-IDF vectorization as fallback mechanism
   - Multi-tiered extraction strategy based on document complexity

3. **Persona-Driven Analyzer** (`process_persona.py`)
   - Collection processing with multi-document intelligence
   - Relevance ranking and prioritization
   - Output generation in standardized JSON format

## Key Features

- **Adaptive Document Processing**: Automatically adapts to different document types and structures
- **Multi-Document Intelligence**: Analyzes connections between documents in a collection
- **Tiered Extraction Strategy**: Balances speed and accuracy based on document complexity
- **Robust Fallback Mechanisms**: Ensures reliable performance across diverse PDF types
- **Parallel Processing Support**: Handles multiple document collections efficiently

For more details on our implementation approach and methodology, see our **[approach explanation document](approach_explanation.md)**.

## Installation & Usage

### Prerequisites

- Docker with AMD64 support
- At least 8 CPUs and 16GB RAM recommended

### Building the Image

```bash
docker build --platform linux/amd64 -t connect-the-dots-pdf-challenge-1b:latest .
```

### Running the Solution

The solution processes each collection individually. For each collection, run:

```bash
docker run --rm \
  -v "$(pwd)/Collection 1:/app/input" \
  -v "$(pwd)/Collection 1:/app/output" \
  --network none \
  connect-the-dots-pdf-challenge-1b:latest
```

Replace `Collection 1` with the appropriate collection directory (`Collection 2`, `Collection 3`, etc.).

The container processes the collection found in the `/app/input` directory and generates the output in `/app/output`. Note that both input and output are mapped to the same collection directory for convenience.

## Configuration

Input JSON structure:

```json
{
  "challenge_info": {
    "challenge_id": "round_1b_XXX",
    "test_case_name": "specific_test_case"
  },
  "documents": [{"filename": "doc.pdf", "title": "Document Title"}],
  "persona": {"role": "User Persona"},
  "job_to_be_done": {"task": "Task description"}
}
```

Output JSON structure:

```json
{
  "metadata": {
    "input_documents": ["doc1.pdf", "doc2.pdf"],
    "persona": "User Persona",
    "job_to_be_done": "Task description",
    "processing_timestamp": "2025-07-27T10:30:15.123456"
  },
  "extracted_sections": [
    {
      "document": "doc1.pdf",
      "section_title": "Important Section",
      "importance_rank": 1,
      "page_number": 3
    }
  ],
  "subsection_analysis": [
    {
      "document": "doc1.pdf",
      "refined_text": "Detailed content...",
      "page_number": 3
    }
  ]
}
```

## Performance Considerations

- Processing time scales with document complexity and collection size
- Typical processing time: 10-45 seconds for 5-10 documents
- Model cache improves performance for subsequent runs
- The system uses a maximum of 1GB for models and processing

## Sample Collections

The solution includes three test collections:

1. **Collection 1: Travel Planning**
   - 7 PDFs about South of France
   - Persona: Travel Planner
   - Task: Plan a 4-day trip for 10 college friends

2. **Collection 2: Adobe Acrobat Learning**
   - 15 PDFs about Adobe Acrobat features
   - Persona: HR Professional
   - Task: Create and manage fillable forms

3. **Collection 3: Recipe Collection**
   - 9 PDFs with food recipes
   - Persona: Food Contractor
   - Task: Prepare vegetarian buffet-style dinner menu

Each collection demonstrates the system's ability to extract relevant information across different domains and use cases.
