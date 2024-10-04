import unittest
from unittest.mock import MagicMock
from app.core.models import Document
from app.usecases import RAGService
from app.core import ports

class TestRAGServiceIntegration(unittest.TestCase):
    def setUp(self):
        # Crear mocks para el repositorio y el adaptador
        self.mock_document_repo = MagicMock(spec=ports.DocumentRepositoryPort)
        self.mock_openai_adapter = MagicMock(spec=ports.LlmPort)

        # Crear instancia del servicio con los mocks
        self.rag_service = RAGService(
            document_repo=self.mock_document_repo,
            openai_adapter=self.mock_openai_adapter
        )

    def test_integration_generate_answer(self):
        query = "¿Qué es la inteligencia artificial?"

        # Simular el comportamiento del repositorio (retorna documentos)
        mock_documents = [
            Document(content="La inteligencia artificial se basa en algoritmos avanzados.")
        ]
        self.mock_document_repo.get_documents.return_value = mock_documents

        # Simular el comportamiento del adaptador OpenAI
        self.mock_openai_adapter.generate_text.return_value = "La IA es un campo de estudio que involucra el diseño de algoritmos avanzados."

        # Llamar al método del servicio
        response = self.rag_service.generate_answer(query)

        # Verificar que los documentos fueron recuperados correctamente
        self.mock_document_repo.get_documents.assert_called_once_with(query)

        # Verificar que se generó la respuesta con el contexto adecuado
        expected_context = "La inteligencia artificial se basa en algoritmos avanzados."
        self.mock_openai_adapter.generate_text.assert_called_once_with(
            prompt=query, retrieval_context=expected_context
        )

        # Verificar que la respuesta es correcta
        self.assertEqual(response, "La IA es un campo de estudio que involucra el diseño de algoritmos avanzados.")

    def test_integration_save_document(self):
        content = "Este es un documento de prueba para IA."

        # Llamar al método del servicio para guardar el documento
        self.rag_service.save_document(content)

        # Verificar que el documento fue guardado correctamente en el repositorio
        self.mock_document_repo.save_document.assert_called_once()

        # Obtener el documento que se pasó al repositorio para verificar el contenido
        saved_document = self.mock_document_repo.save_document.call_args[0][0]
        self.assertEqual(saved_document.content, content)

if __name__ == "__main__":
    unittest.main()
