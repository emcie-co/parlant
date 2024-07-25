from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Mapping, Sequence, Type

from emcie.server.base_models import DefaultBaseModel
from emcie.server.core.persistence.common import ObjectId, Where


class DocumentDatabase(ABC):

    @abstractmethod
    def create_collection(
        self,
        name: str,
        schema: Type[DefaultBaseModel],
    ) -> DocumentCollection:
        """
        Creates a new collection with the given name and returns the collection.
        """
        ...

    @abstractmethod
    def get_collection(
        self,
        name: str,
    ) -> DocumentCollection:
        """
        Retrieves an existing collection by its name.
        """
        ...

    @abstractmethod
    def get_or_create_collection(
        self,
        name: str,
        schema: Type[DefaultBaseModel],
    ) -> DocumentCollection:
        """
        Retrieves an existing collection by its name or creates a new one if it does not exist.
        """
        ...

    @abstractmethod
    def delete_collection(
        self,
        name: str,
    ) -> None:
        """
        Deletes a collection by its name.
        """
        ...


class DocumentCollection(ABC):

    @abstractmethod
    async def find(
        self,
        filters: Where,
    ) -> Sequence[Mapping[str, Any]]:
        """
        Finds all documents that match the given filters.
        """
        ...

    @abstractmethod
    async def find_one(
        self,
        filters: Where,
    ) -> Mapping[str, Any]:
        """
        Returns the first document that matches the query criteria.
        """
        ...

    @abstractmethod
    async def insert_one(
        self,
        document: Mapping[str, Any],
    ) -> ObjectId:
        """
        Inserts a single document into the collection.
        """
        ...

    @abstractmethod
    async def update_one(
        self,
        filters: Where,
        updated_document: Mapping[str, Any],
        upsert: bool = False,
    ) -> ObjectId:
        """
        Updates the first document that matches the query criteria. If upsert is True,
        inserts the document if it does not exist.
        """
        ...

    @abstractmethod
    async def delete_one(
        self,
        filters: Where,
    ) -> None:
        """
        Deletes the first document that matches the query criteria.
        """
        ...


class VectorCollection(DocumentCollection):
    @abstractmethod
    async def find_similar_documents(
        self,
        filters: Where,
        query: str,
        k: int,
    ) -> Sequence[Mapping[str, Any]]:
        """
        Finds the k most similar documents to the given query in the collection.
        """
        ...
