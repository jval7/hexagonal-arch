import unittest
from unittest.mock import MagicMock
from app.usecases import RAGService
from app.core.models import Document
from app.core import ports

class TestRAGService(unittest.TestCase):
    def setUp(self):
        self.mock_document_repo = MagicMock(spec=ports.DocumentRepositoryPort)
        self.mock_openai_adapter = MagicMock(spec=ports.LlmPort)
        self.rag_service = RAGService(
            document_repo=self.mock_document_repo,
            openai_adapter=self.mock_openai_adapter
        )

    def test_generate_answer_with_empty_documents(self):
        query = "¿Qué es Python?"
        self.mock_document_repo.get_documents.return_value = []
        response = self.rag_service.generate_answer(query)
        self.mock_openai_adapter.generate_text.assert_called_once_with(prompt=query, retrieval_context="")
        self.assertEqual(response, "")

    def test_generate_answer_with_documents(self):
        query = "¿Qué es Python?"
        mock_documents = [Document(content="Python es un lenguaje de programación.")]
        self.mock_document_repo.get_documents.return_value = mock_documents
        self.mock_openai_adapter.generate_text.return_value = "Python es un lenguaje de programación popular."
        response = self.rag_service.generate_answer(query)
        self.mock_openai_adapter.generate_text.assert_called_once_with(
            prompt=query, retrieval_context="Python es un lenguaje de programación."
        )
        self.assertEqual(response, "Python es un lenguaje de programación popular.")

    def test_save_document(self):
        content = "Este es un nuevo documento."
        self.rag_service.save_document(content)
        self.mock_document_repo.save_document.assert_called_once()
        saved_document = self.mock_document_repo.save_document.call_args[0][0]
        self.assertEqual(saved_document.content, content)

    def test_generate_answer_with_multiple_documents(self):
        query = "¿Cómo funciona un motor de búsqueda?"
        mock_documents = [
            Document(content="Un motor de búsqueda usa crawlers para indexar páginas."),
            Document(content="Los algoritmos de un motor de búsqueda determinan la relevancia de los resultados.")
        ]
        self.mock_document_repo.get_documents.return_value = mock_documents
        self.mock_openai_adapter.generate_text.return_value = "Los motores de búsqueda son sistemas complejos que indexan y clasifican páginas."
        response = self.rag_service.generate_answer(query)
        expected_context = "Un motor de búsqueda usa crawlers para indexar páginas. Los algoritmos de un motor de búsqueda determinan la relevancia de los resultados."
        self.mock_openai_adapter.generate_text.assert_called_once_with(prompt=query, retrieval_context=expected_context)
        self.assertEqual(response, "Los motores de búsqueda son sistemas complejos que indexan y clasifican páginas.")

    def test_generate_answer_openai_error(self):
        query = "Explícanos sobre la inteligencia artificial"
        mock_documents = [Document(content="La inteligencia artificial se basa en algoritmos avanzados.")]
        self.mock_document_repo.get_documents.return_value = mock_documents
        self.mock_openai_adapter.generate_text.side_effect = Exception("OpenAI API error")
        with self.assertRaises(Exception):
            self.rag_service.generate_answer(query)

if __name__ == '__main__':
    unittest.main()
