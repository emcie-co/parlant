from fastapi import status
from fastapi.testclient import TestClient

from parlant.core.agents import AgentId


def test_that_a_term_can_be_created(
    client: TestClient,
    agent_id: AgentId,
) -> None:
    name = "guideline"
    description = "when and then statements"
    synonyms = ["rule", "principle"]

    response = client.post(
        f"/agents/{agent_id}/terms/",
        json={
            "name": name,
            "description": description,
            "synonyms": synonyms,
        },
    )

    assert response.status_code == status.HTTP_201_CREATED

    assert "term" in response.json()
    data = response.json()["term"]

    assert data["name"] == name
    assert data["description"] == description
    assert data["synonyms"] == synonyms


def test_that_a_term_can_be_createdwithout_synonyms(
    client: TestClient,
    agent_id: AgentId,
) -> None:
    name = "guideline"
    description = "when and then statements"

    response = client.post(
        f"/agents/{agent_id}/terms/",
        json={
            "name": name,
            "description": description,
        },
    )

    assert response.status_code == status.HTTP_201_CREATED

    assert "term" in response.json()
    data = response.json()["term"]

    assert data["name"] == name
    assert data["description"] == description
    assert data["synonyms"] == []


def test_that_a_term_can_be_read(
    client: TestClient,
    agent_id: AgentId,
) -> None:
    name = "guideline"
    description = "when and then statements"
    synonyms = ["rule", "principle"]

    create_response = client.post(
        f"/agents/{agent_id}/terms/",
        json={
            "name": name,
            "description": description,
            "synonyms": synonyms,
        },
    )
    assert create_response.status_code == status.HTTP_201_CREATED
    term = create_response.json()["term"]

    read_response = client.get(f"agents/{agent_id}/terms/{term['id']}")
    assert read_response.status_code == status.HTTP_200_OK

    data = read_response.json()
    assert data["name"] == name
    assert data["description"] == description
    assert data["synonyms"] == synonyms


def test_that_a_term_can_be_read_without_synonyms(
    client: TestClient,
    agent_id: AgentId,
) -> None:
    name = "guideline"
    description = "when and then statements"

    create_response = client.post(
        f"/agents/{agent_id}/terms/",
        json={
            "name": name,
            "description": description,
        },
    )
    assert create_response.status_code == status.HTTP_201_CREATED
    term = create_response.json()["term"]

    read_response = client.get(f"/agents/{agent_id}/terms/{term['id']}")
    assert read_response.status_code == status.HTTP_200_OK

    data = read_response.json()
    assert data["name"] == name
    assert data["description"] == description
    assert data["synonyms"] == []


def test_that_terms_can_be_listed_for_an_agent(
    client: TestClient,
    agent_id: AgentId,
) -> None:
    terms = [
        {"name": "guideline1", "description": "description 1", "synonyms": ["synonym1"]},
        {"name": "guideline2", "description": "description 2", "synonyms": ["synonym2"]},
    ]

    for term in terms:
        response = client.post(
            f"/agents/{agent_id}/terms",
            json={
                "name": term["name"],
                "description": term["description"],
                "synonyms": term["synonyms"],
            },
        )
        assert response.status_code == status.HTTP_201_CREATED

    list_response = client.get(f"/agents/{agent_id}/terms/")
    assert list_response.status_code == status.HTTP_200_OK

    data = list_response.json()
    returned_terms = data["terms"]
    assert len(returned_terms) == 2

    assert {
        "name": returned_terms[1]["name"],
        "description": returned_terms[1]["description"],
        "synonyms": returned_terms[1]["synonyms"],
    } in terms

    assert {
        "name": returned_terms[0]["name"],
        "description": returned_terms[0]["description"],
        "synonyms": returned_terms[0]["synonyms"],
    } in terms


def test_that_a_term_can_be_updated(
    client: TestClient,
    agent_id: AgentId,
) -> None:
    name = "guideline"
    description = "when and then statements"
    synonyms = ["rule", "principle"]

    term = (
        client.post(
            f"/agents/{agent_id}/terms/",
            json={
                "name": name,
                "description": description,
                "synonyms": synonyms,
            },
        )
        .raise_for_status()
        .json()["term"]
    )

    updated_name = "updated guideline"
    updated_description = "Updated guideline description"
    updated_synonyms = ["instruction"]

    update_response = client.patch(
        f"/agents/{agent_id}/terms/{term["id"]}",
        json={
            "name": updated_name,
            "description": updated_description,
            "synonyms": updated_synonyms,
        },
    )

    assert update_response.status_code == status.HTTP_200_OK

    data = update_response.json()
    assert data["name"] == updated_name
    assert data["description"] == updated_description
    assert data["synonyms"] == updated_synonyms


def test_that_a_term_can_be_deleted(
    client: TestClient,
    agent_id: AgentId,
) -> None:
    name = "guideline"
    description = "when and then statements"
    synonyms = ["rule", "principle"]

    term = (
        client.post(
            f"/agents/{agent_id}/terms",
            json={
                "name": name,
                "description": description,
                "synonyms": synonyms,
            },
        )
        .raise_for_status()
        .json()["term"]
    )

    delete_response = (
        client.delete(f"/agents/{agent_id}/terms/{term['id']}").raise_for_status().json()
    )
    assert delete_response["term_id"] == term["id"]

    read_response = client.get(f"/agents/{agent_id}/terms/{name}")
    assert read_response.status_code == status.HTTP_404_NOT_FOUND
