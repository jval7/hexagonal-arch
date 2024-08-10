from app.adapters.openai_adapter import OpenAIAdapter
from app.adapters.chromadb_adapter import ChromaDBAdapter
from app import usecases


def get_rag_service() -> usecases.RAGService:
    openai_adapter = OpenAIAdapter(api_key="your-api-key")
    document_repo = ChromaDBAdapter()
    return usecases.RAGService(document_repo=document_repo, openai_adapter=openai_adapter)
