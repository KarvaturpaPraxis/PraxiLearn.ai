# prepare_embeddings.py

import os
import pickle
import numpy as np
import faiss
from process_pdf import (
    extract_text_from_pdf,
    parse_pdf_to_structure,
    clean_and_split_text,
    generate_embedding,
    create_embeddings_store,
    build_faiss_index
)

def create_embeddings_store_from_structure(structured_data):
    embeddings_store = []
    for idx, entry in enumerate(structured_data):
        text = entry['content'].strip()
        if not text:
            print(f"Skipping entry {idx}: Text is empty or whitespace only.")
            continue

        print(f"Generating embedding for entry {idx}: {text[:100]}... [truncated]")
        embedding = generate_embedding(text)
        if embedding is not None:
            embeddings_store.append({
                'id': idx,
                'chapter': entry.get('chapter', ''),
                'section': entry.get('section', ''),
                'subsection': entry.get('subsection', ''),
                'content': text,
                'embedding': embedding
            })
        else:
            print(f"Failed to generate embedding for entry {idx}.")
    return embeddings_store



"""def prepare_embeddings(pdf_path, embeddings_path, index_path):
    # Extract and process text
    text = extract_text_from_pdf(pdf_path)
    chunks = clean_and_split_text(text)
    
    # Generate embeddings
    embeddings_store = create_embeddings_store(chunks)
    
    # Build FAISS index
    index, _ = build_faiss_index(embeddings_store)
    
    # Save embeddings_store
    with open(embeddings_path, 'wb') as f:
        pickle.dump(embeddings_store, f)
    
    # Save FAISS index
    faiss.write_index(index, index_path)"""

def prepare_embeddings(pdf_path, embeddings_path, index_path):
    print("Parsing PDF into structured format...")
    structured_data = parse_pdf_to_structure(pdf_path)

    if not structured_data:
        print("No structured data extracted from the PDF.")
        return

    print("Generating embeddings from structured data...")
    embeddings_store = create_embeddings_store_from_structure(structured_data)

    if not embeddings_store:
        print("No embeddings generated.")
        return

    print("Building FAISS index...")
    dimension = len(embeddings_store[0]['embedding'])
    index = faiss.IndexFlatL2(dimension)
    embeddings_matrix = np.array([item['embedding'] for item in embeddings_store])
    index.add(embeddings_matrix)

    print("Saving embeddings store...")
    with open(embeddings_path, 'wb') as f:
        pickle.dump(embeddings_store, f)

    print("Saving FAISS index...")
    faiss.write_index(index, index_path)


if __name__ == '__main__':
    pdf_path = 'JOHT_KK_22.pdf'  # Update with your actual PDF file path
    embeddings_path = 'embeddings_store_johtkk22.pkl'
    index_path = 'faiss_index.index'

    prepare_embeddings(pdf_path, embeddings_path, index_path)