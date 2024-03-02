from abc import ABC, abstractmethod
import os
from typing import List
from scipy import spatial
from sklearn.metrics.pairwise import cosine_similarity
import openai


class EmbedderModel(ABC):
    @abstractmethod
    def get_embedding(
        self,
        text: str,
    ) -> List[int]:
        return []

    @staticmethod
    def get_similarity(first_vec: List[float], second_vec: List[float]):
        return spatial.distance.cosine(first_vec, second_vec)


class OpenAIEmbedder(EmbedderModel):
    """
    engine for example: "text-embedding-ada-002", "text-search-davinci-*-001"
    """

    def __init__(self, embedding_model="text-embedding-ada-002"):
        self.embedding_model = embedding_model
        self.client = openai.AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

    async def get_embedding(self, text: str) -> List[int]:
        response = await self.client.embeddings.create(input=[text], model=self.embedding_model)
        return response.data[0].embedding
