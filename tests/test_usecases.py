from typing import List
from unittest.mock import Mock
from app import usecases
from app.core import ports, models


class FakeLLM(ports.LlmPort):
    def __init__(self) -> None:
        self.prompt_history = []
        self.retrieval_context = ""

    def generate_text(self, prompt: str, retrieval_context: str) -> str:
        self.prompt_history.append(prompt)
        self.retrieval_context = retrieval_context
        return "Generated answer for: " + prompt


def test_should_save_document_when_calling_rag_service_save_method():
    # Arrange
    document_repo = Mock(spec=ports.DocumentRepositoryPort)
    FakeLLM_ = FakeLLM()

    rag_service = usecases.RAGService(
        document_repo=document_repo, openai_adapter=FakeLLM_
    )

    # Act
    rag_service.save_document(
        content="this is a test fo save document in vectordatabase"
    )

    # Assert
    document_repo.save_document.assert_called_once()
    assert len(FakeLLM_.prompt_history) == 0


def test_should_generate_answer_when_calling_rag_service_generate_answer_method():
    # Arrange
    document_repo = Mock(spec=ports.DocumentRepositoryPort)
    FakeLLM_ = FakeLLM()

    content1 = "this is a test fo save document in vectordatabase 1"
    content2 = "this is a test fo save document in vectordatabase 2"
    retrieval_context = " ".join([content1, content2])

    document_repo.get_documents.return_value = [
        models.Document(content=content1),
        models.Document(content=content2),
    ]

    rag_service = usecases.RAGService(
        document_repo=document_repo, openai_adapter=FakeLLM_
    )

    # Act
    rag_service.generate_answer(query="test")

    # Assert
    document_repo.get_documents.assert_called_once_with('test')
    assert FakeLLM_.retrieval_context == retrieval_context
    assert len(FakeLLM_.prompt_history) > 0
    assert FakeLLM_.prompt_history[-1] == "test"
