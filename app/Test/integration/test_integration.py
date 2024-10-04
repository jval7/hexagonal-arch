import unittest
from unittest.mock import MagicMock
from app.core.models import Document
from app.usecases import RAGService
from app.core import ports


class TestRAGServiceIntegration(unittest.TestCase):
    def setUp(self):
        self.mock_document_repo = MagicMock(spec=ports.DocumentRepositoryPort)
        self.mock_openai_adapter = MagicMock(spec=ports.LlmPort)
        self.rag_service = RAGService(
            document_repo=self.mock_document_repo,
            openai_adapter=self.mock_openai_adapter
        )

    def test_integration_generate_answer_about_cali(self):
        query = "¿Cuál es la historia de la ciudad de Cali?"

        mock_documents = [
            Document(content="Cali fue fundada en 1536 por Sebastián de Belalcázar."),
            Document(content="Cali es conocida como la capital mundial de la salsa.")
        ]
        self.mock_document_repo.get_documents.return_value = mock_documents

        self.mock_openai_adapter.generate_text.return_value = "Cali, fundada en 1536, es una de las principales ciudades de Colombia y es reconocida por su música salsa."

        response = self.rag_service.generate_answer(query)

        expected_context = "Cali fue fundada en 1536 por Sebastián de Belalcázar. Cali es conocida como la capital mundial de la salsa."
        self.mock_openai_adapter.generate_text.assert_called_once_with(
            prompt=query, retrieval_context=expected_context
        )
        self.assertEqual(response,
                         "Cali, fundada en 1536, es una de las principales ciudades de Colombia y es reconocida por su música salsa.")

    def test_integration_generate_answer_about_uao(self):
        query = "¿Qué es la Universidad Autónoma de Occidente?"

        mock_documents = [
            Document(
                content="La Universidad Autónoma de Occidente es una institución privada en Cali, fundada en 1970."),
            Document(content="La UAO es conocida por sus programas de ingeniería y comunicación social.")
        ]
        self.mock_document_repo.get_documents.return_value = mock_documents

        self.mock_openai_adapter.generate_text.return_value = "La Universidad Autónoma de Occidente, fundada en 1970, es reconocida por sus facultades de ingeniería y comunicación social."

        response = self.rag_service.generate_answer(query)

        expected_context = "La Universidad Autónoma de Occidente es una institución privada en Cali, fundada en 1970. La UAO es conocida por sus programas de ingeniería y comunicación social."
        self.mock_openai_adapter.generate_text.assert_called_once_with(
            prompt=query, retrieval_context=expected_context
        )
        self.assertEqual(response,
                         "La Universidad Autónoma de Occidente, fundada en 1970, es reconocida por sus facultades de ingeniería y comunicación social.")


if __name__ == '__main__':
    unittest.main()
