from typing import List
from unittest.mock import Mock
from app import usecases
from app.core import ports, models


class FakeDocumentRepo(ports.DocumentRepositoryPort):
    def __init__(self) -> None:
        self._repo = {}

    def save_document(self, document: models.Document) -> None:
        _id = document.id
        self._repo[_id] = document

    def get_documents(
        self, query: str, n_results: int | None = None
    ) -> List[models.Document]:
        return list(self._repo.values())

class FakeLlmPort(ports.LlmPort):
    def __init__(self) -> None:
        self._history = []

    def generate_text(self, prompt: str, retrieval_context: str) -> str:
        self._history.append((prompt, retrieval_context))
        return "Respuesta Mockeada"

def test_should_save_document_when_calling_rag_service_save_method():
    # Arrange
    document_repo = Mock(spec=ports.DocumentRepositoryPort)
    llm_stub = FakeLlmPort()

    rag_service = usecases.RAGService(
        document_repo=document_repo, openai_adapter=llm_stub
    )

    # Act
    rag_service.save_document(
        content="this is a test fo save document in vectordatabase"
    )

    # Assert
    document_repo.save_document.assert_called_once()


def test_should_generate_answer_when_calling_rag_service_generate_answer_method():
    # Arrange
    document_repo = Mock(spec=ports.DocumentRepositoryPort)
    llm_stub = FakeLlmPort()

    content1 = "this is a test fo save document in vectordatabase 1"
    content2 = "this is a test fo save document in vectordatabase 2"

    document_repo.get_documents.return_value = [
        models.Document(content=content1),
        models.Document(content=content2),
    ]

    rag_service = usecases.RAGService(
        document_repo=document_repo, openai_adapter=llm_stub
    )

    # Act
    rag_service.generate_answer(query="test")

    # Assert
    assert llm_stub._history[-1] == ("test", content1 + " " + content2)
