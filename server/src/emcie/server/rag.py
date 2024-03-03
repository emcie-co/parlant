# Copyright (c) 2024 Emcie
# All rights reserved.
#
# This file and its contents are the property of Emcie and are strictly confidential.
# No part of this file may be reproduced, distributed, or transmitted in any form or by any means,
# including photocopying, recording, or other electronic or mechanical methods,
# without the prior written permission of Emcie.
#
# Website: https://emcie.co

import heapq
from typing import List, Dict, Iterable, NewType, Optional, Tuple, Union
from scipy import spatial
from tinydb.storages import MemoryStorage
from pydantic import BaseModel
from tinydb import TinyDB, table

from emcie.server.models import TextEmbeddingModel


class RagDocument(BaseModel):
    id: str
    metadata: Optional[dict] = None
    document: str
    vector: List[float]


class RagStore:
    def __init__(
        self,
        embedding_model: TextEmbeddingModel,
        db: Optional[TinyDB] = None,
    ) -> None:
        self.db = db or TinyDB(storage=MemoryStorage)
        self.embedding_model = embedding_model

    @staticmethod
    def distance(first_vec: Iterable[float], second_vec: Iterable[float]):
        return spatial.distance.cosine(first_vec, second_vec)

    async def upsert(self, document: dict) -> Dict:
        document_vector = (await self.embedding_model.embed(document["document"]))[0]
        full_document = {
            **{"document_vector": document_vector},
            **document,
        }
        self.db.upsert(
            table.Document(
                full_document,
                doc_id=document["id"],
            )
        )
        return document

    async def query(self, query: str, k: int = 3) -> Iterable[Dict]:
        query_embedding = (await self.embedding_model.embed(query))[0]
        docs_with_distance = [
            {
                **doc,
                **{"distance": self.distance(query_embedding, doc["document_vector"])},
            }
            for doc in self.get_all_documents()
        ]
        top_k = heapq.nlargest(
            k,
            docs_with_distance,
            key=lambda doc: 1 - doc["distance"],
        )

        return sorted(top_k, key=lambda doc: doc["distance"])

    def get_all_documents(self) -> Iterable[Dict] | None:
        return self.db.all()
