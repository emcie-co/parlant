from abc import abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import NewType, Optional, Sequence


TermId = NewType("TermId", str)


@dataclass(frozen=True)
class Term:
    id: TermId
    creation_utc: datetime
    name: str
    description: str
    synonyms: Optional[list[str]]

    def __repr__(self) -> str:
        term_string = f"Name: {self.name}, Description: {self.description}"
        if self.synonyms:
            term_string += f", Sysnonyms: {", ".join(self.synonyms)}"
        return term_string

    def __hash__(self) -> int:
        return hash(self.id)


class TerminologyStore:
    @abstractmethod
    async def create_term(
        self,
        term_set: str,
        name: str,
        description: str,
        creation_utc: Optional[datetime] = None,
        synonyms: Optional[Sequence[str]] = None,
    ) -> Term: ...

    @abstractmethod
    async def update_term(
        self,
        term_set: str,
        name: str,
        description: str,
        synonyms: Sequence[str],
    ) -> Term: ...

    @abstractmethod
    async def read_term(
        self,
        term_set: str,
        name: str,
    ) -> Term: ...

    @abstractmethod
    async def list_terms(
        self,
        term_set: str,
    ) -> Sequence[Term]: ...

    @abstractmethod
    async def delete_term(
        self,
        term_set: str,
        name: str,
    ) -> TermId: ...

    @abstractmethod
    async def find_relevant_terms(
        self,
        term_set: str,
        query: str,
    ) -> Sequence[Term]: ...
