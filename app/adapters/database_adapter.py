from app.core import ports, models
from pymongo import MongoClient


class MongoDbAdapter(ports.DatabasePort):
    def __init__(self, url: str) -> None:
        self.client = MongoClient(url)
        self.db = self.client["rag_db"]
        self.users = self.db["users"]
        self.documents = self.db["documents"]

    def save_user(self, username: str, password: str) -> None:
        self.users.insert_one({"username": username, "password": password})

    def get_user(self, username: str) -> models.User:
        user = self.users.find_one({"username": username})
        return models.User(username=user["username"], password=user["password"])

    def save_document(self, document: models.Document) -> None:
        self.documents.insert_one({"id": document.id, "content": document.content})

    def get_document(self, document_id: str) -> models.Document:
        document = self.documents.find_one({"id": document_id})
        return models.Document(id=document["id"], content=document["content"])

    def get_documents(self) -> list[models.Document]:
        documents = self.documents.find()
        return [models.Document(id=document["id"], content=document["content"]) for document in documents]

    def delete_document(self, document_id: str) -> None:
        self.documents.delete_one({"id": document_id})

    def update_document(self, document_id: str, content: str) -> None:
        self.documents.update_one({"id": document_id}, {"$set": {"content": content}})

    def get_user_documents(self, username: str) -> list[models.Document]:
        documents = self.documents.find({"username": username})
        return [models.Document(id=document["id"], content=document["content"]) for document in documents]

    def get_user_document(self, username: str, document_id: str) -> models.Document:
        document = self.documents.find_one({"username": username, "id": document_id})
        return models.Document(id=document["id"], content=document["content"]) if document else None
