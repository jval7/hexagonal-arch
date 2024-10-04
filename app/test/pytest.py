import unittest
from unittest.mock import MagicMock
from app.core.models import Document
from app.core import ports
from app.usecases import RAGService

class TestRAGService(unittest.TestCase):
    def setUp(self):
        # Crear mocks para las dependencias
        self.document_repo_mock = MagicMock(spec=ports.DocumentRepositoryPort)
        self.openai_adapter_mock = MagicMock(spec=ports.LlmPort)

        # Inicializar el servicio con los mocks
        self.rag_service = RAGService(self.document_repo_mock, self.openai_adapter_mock)

    def test_generate_answer(self):
        # Configurar el query y los documentos de ejemplo
        query = "Cual es la capital de Colombia"
        documents = [Document(content="Bogota es la capital de Colombia.")]
        self.document_repo_mock.get_documents.return_value = documents
        self.openai_adapter_mock.generate_text.return_value = "Bogota"

        # Llama al método que estamos probando
        result = self.rag_service.generate_answer(query)

        # Verificar que el repo fue llamado con el query correcto
        self.document_repo_mock.get_documents.assert_called_once_with(query)

        # Verificar que el adaptador de OpenAI fue llamado con el prompt y el contexto correcto
        expected_context = "Bogota es la capital de Colombia."
        self.openai_adapter_mock.generate_text.assert_called_once_with(prompt=query, retrieval_context=expected_context)

        # Verificar que el resultado sea el esperado
        self.assertEqual(result, "Bogota")

    def test_save_document(self):
        # Configurar el contenido de ejemplo
        content = "Este es un documento de pruebas."

        # Llamar al método que estamos probando
        self.rag_service.save_document(content)

        # Verificar que el repositorio fue llamado correctamente con el documento
        expected_document = Document(content=content)
        self.document_repo_mock.save_document.assert_called_once()

        # Verificar que el documento fue creado correctamente
        saved_document = self.document_repo_mock.save_document.call_args[0][0]
        self.assertEqual(saved_document.content, expected_document.content)

if __name__ == "__main__":
    unittest.main()
