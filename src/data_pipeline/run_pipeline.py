import logging
import json
import os
from dotenv import load_dotenv
from collections import defaultdict

# Load env variables FIRST
load_dotenv()

from src.data_pipeline.dataset_builder import DatasetBuilder
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.embedding_model import EmbeddingModel
from src.retrieval.vector_index import VectorIndex
from src.import_map import DocumentChunk, DocumentSource

# Logging config
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def deserialize_chunks(chunks_data: list) -> list:
    """Convert raw JSON chunks to DocumentChunk objects."""
    chunks = []

    for chunk_dict in chunks_data:
        try:
            chunk = DocumentChunk(
                chunk_id=chunk_dict["chunk_id"],
                text=chunk_dict["text"],
                source=DocumentSource(chunk_dict["source"]),
                document_name=chunk_dict["document_name"],
                chunk_index=chunk_dict.get("chunk_index", 0),
                page_number=chunk_dict["page_number"],
                metadata=chunk_dict.get("metadata", {})
            )
            chunks.append(chunk)

        except Exception as e:
            logger.warning(f"Skipping chunk {chunk_dict.get('chunk_id')}: {e}")

    return chunks


def main():
    print("=" * 60)
    print("PHASE 1: DOCUMENT CHUNKING")
    print("=" * 60)

    builder = DatasetBuilder(
        output_dir="data/chunks",
        chunk_size=500,
        chunk_overlap=100
    )

    source_dirs = {
        "personal_tax": "data/raw/income_tax",
        "corporate_tax": "data/raw/corporate_tax",
        "gst": "data/raw/gst",
        "investment": "data/raw/investment",
        "regulatory": "data/raw/regulatory",
    }

    try:
        result = builder.build_dataset(
            source_directories=source_dirs,
            output_filename="chunks.json"
        )

        print(f"\n[OK] Phase 1 Complete: {result['total_chunks']} chunks")

        with open(result["output_path"], "r", encoding="utf-8") as f:
            chunks_data = json.load(f)

    except Exception as e:
        logger.error(f"Phase 1 failed: {e}")
        return

    print("\n" + "=" * 60)
    print("PHASE 2: FEDERATED VECTOR DATABASE (PER-DOMAIN FAISS)")
    print("=" * 60)

    try:
        chunks = deserialize_chunks(chunks_data)

        if len(chunks) == 0:
            raise ValueError("No valid chunks found")

        print(f"Loaded {len(chunks)} chunks")

        # Initialize embedding model
        embedding_model = EmbeddingModel()

        # Get embedding dimension
        print("\nDetecting embedding dimension...")
        dimension = embedding_model.get_embedding_dimension()
        print(f"[OK] Dimension: {dimension}")

        # GROUP CHUNKS BY DOMAIN
        domain_chunks = defaultdict(list)

        for chunk in chunks:
            domain_chunks[chunk.source.value].append(chunk)

        # PROCESS EACH DOMAIN (create separate indices)
        total_indexed = 0
        for domain, domain_data in domain_chunks.items():

            print(f"\n--- DOMAIN: {domain} ---")
            print(f"Chunks: {len(domain_data)}")

            # Initialize FAISS for this domain
            vector_index = VectorIndex(dimension=dimension)

            hybrid_retriever = HybridRetriever(
                embedding_model=embedding_model,
                vector_index=vector_index,
                chunks=domain_data
            )

            print(f"Embedding {len(domain_data)} chunks...")
            hybrid_retriever.index_documents(domain_data)

            # VALIDATION
            if vector_index.index.ntotal != len(domain_data):
                raise ValueError(
                    f"Mismatch in {domain}: {vector_index.index.ntotal} != {len(domain_data)}"
                )

            print(f"[OK] {domain}: {len(domain_data)} embeddings stored")
            total_indexed += len(domain_data)

            # SAVE PER DOMAIN
            base_path = f"data/vector_store/{domain}"
            os.makedirs(base_path, exist_ok=True)

            faiss_path = f"{base_path}/index.faiss"
            metadata_path = f"{base_path}/metadata.json"

            vector_index.save_index(faiss_path)

            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(vector_index.metadata, f, indent=2)

            print(f"[OK] Saved {domain} to {base_path}/")

        print("\n" + "=" * 60)
        print("[SUCCESS] FEDERATED PIPELINE COMPLETE")
        print("=" * 60)
        print(f"Total chunks indexed: {total_indexed}")
        print(f"Domains: {list(domain_chunks.keys())}")
        print(f"Storage: data/vector_store/{{domain}}/index.faiss")
        print("\nNext: Run 'python run_demo.py' to test the system")

    except Exception as e:
        logger.error(f"Phase 2 failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()