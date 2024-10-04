import unittest
from unittest.mock import MagicMock, patch

from app.adapters.chromadb_adapter import ChromaDBAdapter
from app.adapters.openai_adapter import OpenAIAdapter
from app.core.models import Document
from app.core import ports
from app.usecases import RAGService

class TestRAG(unittest.TestCase):

    def setUp(self):
        #Crear mocks
        self.document_repo_mock = MagicMock(spec=ports.DocumentRepositoryPort)
        self.openai_adapter_mock = MagicMock(spec=ports.LlmPort)

        self.rag_service = RAGService(
            document_repo = self.document_repo_mock,
            openai_adapter = self.openai_adapter_mock
        )

    def test_generate_answer_when_calling_generate_answer_in_RAG(self):
        #Arrange
        query = "Que es dragon ball?"
        mock_documento = [
            Document(content = "Una serie creada por Akira Toriyama"),
            Document(content="La serie trata de Goku, un saiyajin")
        ]

        self.document_repo_mock.get_documents.return_value = mock_documento

        self.openai_adapter_mock.generate_text.return_value = \
            ("Dragon Ball z es una serie de anime creada por "
             "Akira Toriyama en la que Goku y sus amigos pelean contra enemigos poderosos")

        #Act
        result = self.rag_service.generate_answer(query)

        #Assert
        self.document_repo_mock.get_documents.assert_called_once_with(query)

        expected_context = "Una serie creada por Akira Toriyama La serie trata de Goku, un saiyajin"
        self.openai_adapter_mock.generate_text.assert_called_once_with(
            prompt = query, retrieval_context = expected_context
        )

        self.assertEqual(result, "Dragon Ball z es una serie de anime creada por "
                                 "Akira Toriyama en la que Goku y sus amigos pelean contra enemigos poderosos")




    def test_save_document_when_calling_save_document_in_RAG(self):
        #Arrange
        content = "Este es un documento de prueba"

        #Act
        self.rag_service.save_document(content)

        #Assert
        self.document_repo_mock.save_document.assert_called_once()
        saved_document = self.document_repo_mock.save_document.call_args[0][0]
        self.assertIsInstance(saved_document, Document)
        self.assertEqual(saved_document.content, content)


class TestChromaDBAdapter(unittest.TestCase):

    def setUp(self):
        # Crear un mock de ChromaDBAdapter
        self.chromadb_adapter = ChromaDBAdapter(number_of_vectorial_results=5)

        # Crear mocks para el cliente y la colección de ChromaDB
        self.chromadb_adapter.client = MagicMock()
        self.chromadb_adapter.collection = MagicMock()

    def test_save_document(self):
        # Arrange
        document = Document(id="123", content="Este es un documento de prueba")

        # Act
        self.chromadb_adapter.save_document(document)

        # Assert
        self.chromadb_adapter.collection.add.assert_called_once_with(
            ids=["123"],
            documents=["Este es un documento de prueba"]
        )

    def test_get_documents(self):

        # Arrange
        mock_result = {
            'ids': [['123']],
            'documents': [["Este es un documento de prueba"]]
        }
        self.chromadb_adapter.collection.query.return_value = mock_result

        query = "test query"
        expected_documents = [Document(id="123", content="Este es un documento de prueba")]

        # Act
        result = self.chromadb_adapter.get_documents(query=query, n_results=1)

        #Assert
        self.chromadb_adapter.collection.query.assert_called_once_with(
            query_texts=[query], n_results=1
        )

        self.assertEqual(result, expected_documents)




class TestOpenAIAdapter(unittest.TestCase):

    @patch('openai.OpenAI')
    def test_generate_text(self, mock_openai):

        # Arrange
        mock_completion = MagicMock()
        mock_openai.return_value.chat.completions.create.return_value = mock_completion

        mock_completion.choices = [MagicMock(message=MagicMock(content="Respuesta simulada de OpenAI"))]

        adapter = OpenAIAdapter(api_key="fake-api-key", model="gpt-3.5-turbo", max_tokens=100, temperature=0.7)


        # Act
        result = adapter.generate_text(prompt="¿Qué es Dragon Ball?", retrieval_context="Dragon Ball es un anime")

        # Assert
        mock_openai.return_value.chat.completions.create.assert_called_once_with(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system",
                 "content": "The context is: Dragon Ball es un anime, please respond to the following question: "},
                {"role": "user", "content": "¿Qué es Dragon Ball?"}
            ],
            max_tokens=100,
            temperature=0.7,
        )

        self.assertEqual(result, "Respuesta simulada de OpenAI")