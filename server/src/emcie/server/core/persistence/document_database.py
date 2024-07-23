from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Mapping, Sequence, Type

from emcie.server.base_models import DefaultBaseModel
from emcie.server.core.persistence.common import (
    ObjectId,
    Where,
)


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
        Get or create a new collection
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
    ) -> Sequence[Mapping[str, Any]]: ...

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
    ) -> ObjectId: ...

    @abstractmethod
    async def update_one(
        self,
        filters: Where,
        updated_document: Mapping[str, Any],
        upsert: bool = False,
    ) -> ObjectId:
        """
        Updates the first document that matches the query criteria.
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
