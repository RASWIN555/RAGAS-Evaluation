# High-level script for Ragas evaluation project

import os
from retriever.rag_chain import create_index, add_embeddings
from retriever.retriever_config import RetrieverConfig
from generator.qa_chain import RetrievalQA
from evaluation.run_ragas_eval import run_evaluation

def main():
    # Load configuration
    config = RetrieverConfig()
    
    # Create Pinecone index and add embeddings
    create_index(config)
    add_embeddings(config)

    # Initialize the QA chain
    qa_chain = RetrievalQA(config)

    # Run the evaluation
    results = run_evaluation(qa_chain)

    # Save or display results
    print("Evaluation Results:", results)

if __name__ == "__main__":
    main()