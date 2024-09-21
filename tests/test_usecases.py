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


def test_should_save_document_when_calling_rag_service_save_method():
    # Arrange
    document_repo = FakeDocumentRepo()
    llm_mock = Mock(spec=ports.LlmPort)

    rag_service = usecases.RAGService(
        document_repo=document_repo, openai_adapter=llm_mock
    )

    # Act
    rag_service.save_document(
        content="this is a test fo save document in vectordatabase"
    )

    # Assert
    documents = document_repo.get_documents(query="test")

    assert len(documents) > 0


def test_should_generate_answer_when_calling_rag_service_generate_answer_method():
    # Arrange
    document_repo = FakeDocumentRepo()
    llm_mock = Mock(spec=ports.LlmPort)

    content1 = "this is a test fo save document in vectordatabase 1"
    content2 = "this is a test fo save document in vectordatabase 2"
    retrieval_context = " ".join([content1, content2])

    document_repo.save_document(document=models.Document(content=content1))
    document_repo.save_document(document=models.Document(content=content2))
    rag_service = usecases.RAGService(
        document_repo=document_repo, openai_adapter=llm_mock
    )

    # Act
    rag_service.generate_answer(query="test")

    # Assert
    llm_mock.generate_text.assert_called_once_with(
        prompt="test", retrieval_context=retrieval_context
    )
