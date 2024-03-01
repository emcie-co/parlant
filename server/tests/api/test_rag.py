import json
import os
from typing import Any, Dict
from fastapi.testclient import TestClient
from fastapi import status
import time
from emcie.server.rag import RagDocument

from emcie.server.api.agents import ReactionDTO
from emcie.server.api.threads import MessageDTO


def test_upsert_rag(
    client: TestClient,
) -> None:

    response = client.post(
        "/rag",
        json={
            "id": "123",
            "metadata": {"id": 123, "user_name": "dorzo"},
            "document": "hamburger is Dor's favourite food",
        },
    )

    assert response.status_code == status.HTTP_200_OK

    response = client.get("/rag")

    assert response.status_code == status.HTTP_200_OK

    data = response.json()

    assert len(data) == 1
    assert data[0]["document"] == "hamburger is Dor's favourite food"
    assert data[0]["metadata"] == {"id": 123, "user_name": "dorzo"}

    response = client.post(
        "/rag",
        json={"id": "124", "document": "Pizza and Hamburgers are both considered fast food"},
    )

    assert response.status_code == status.HTTP_200_OK

    response = client.get("/rag")

    data = response.json()

    assert response.status_code == status.HTTP_200_OK

    assert len(data) == 2
    assert data[1]["document"] == "Pizza and Hamburgers are both considered fast food"
    assert data[1]["metadata"] == None

    response = client.post(
        "/rag",
        json={"id": "123", "metadata": {}, "document": "Pizza is Dor's favourite food"},
    )

    assert response.status_code == status.HTTP_200_OK

    response = client.get("/rag")

    data = response.json()

    assert response.status_code == status.HTTP_200_OK

    assert len(data) == 2
    assert data[0]["document"] == "Pizza is Dor's favourite food"  # train need to be sent first
    assert data[1]["document"] == "Pizza and Hamburgers are both considered fast food"


def test_query_rag(
    client: TestClient,
) -> None:
    response = client.post(
    "/rag",
    json={
        "id": "1",
        "metadata": {"id": 1, "user_name": "dorzo"},
        "document": "A hamburger is a delicious meal made from ground meat, often beef, served between two slices of bread or a bun. It's a popular choice in many countries and can be customized with a variety of toppings like lettuce, tomato, onions, cheese, and condiments such as ketchup, mustard, and mayonnaise.",
    },
    )
    response = client.post(
        "/rag",
        json={
            "id": "2",
            "metadata": {"id": 2, "user_name": "dorzo"},
            "document": "Meat, a staple in many diets, comes from animals, with beef being one of the most common sources. Cows are raised in various environments around the world, and their meat is processed in numerous ways to create cuts that vary in flavor and texture.",
        },
    )
    response = client.post(
        "/rag",
        json={
            "id": "3",
            "metadata": {"id": 3, "user_name": "dorzo"},
            "document": "Toppings made from meat, such as pepperoni, sausage, and bacon, significantly enhance the flavor of pizza, making it a beloved choice for many. These toppings add a rich and savory taste, complementing the cheese and tomato sauce beautifully.",
        },
    )
    response = client.post(
        "/rag",
        json={
            "id": "4",
            "metadata": {"id": 4, "user_name": "dorzo"},
            "document": "Pizza is a versatile and beloved dish that is easy to prepare, making it a favorite for quick dinners or social gatherings. With a simple base of dough, sauce, and cheese, it can be customized with an endless variety of toppings to suit any taste.",
        },
    )

    response = client.post(
        "/rag",
        json={
            "id": "5",
            "metadata": {"id": 5, "user_name": "dorzo"},
            "document": "Restaurants that serve both pizzas and hamburgers often face criticism for not specializing in one type of cuisine, leading to perceptions of lower quality. However, many such establishments successfully cater to a wide range of tastes, offering delicious options for everyone.",
        },)

    response = client.get(f"/rag?query={"Is pepperoni considered a good choice of topping?"}")

    assert response.status_code == status.HTTP_200_OK

    assert len(response.json()) == 3

    docs = response.json()
    assert docs[0]["document"] == "Toppings made from meat, such as pepperoni, sausage, and bacon, significantly enhance the flavor of pizza, making it a beloved choice for many. These toppings add a rich and savory taste, complementing the cheese and tomato sauce beautifully."


