import dateutil.parser
from fastapi import status
import httpx
from lagom import Container
from pytest import raises

from parlant.core.common import ItemNotFoundError
from parlant.core.fragments import FragmentStore, FragmentField
from parlant.core.tags import TagStore


async def test_that_a_fragment_can_be_created(
    async_client: httpx.AsyncClient,
) -> None:
    payload = {
        "value": "Your account balance is {balance}",
        "fields": [
            {
                "name": "balance",
                "description": "Account's balance",
                "examples": ["9000"],
            }
        ],
    }

    response = await async_client.post("/fragments", json=payload)
    assert response.status_code == status.HTTP_201_CREATED

    fragment = response.json()

    assert fragment["value"] == payload["value"]
    assert fragment["fields"] == payload["fields"]

    assert "id" in fragment
    assert "creation_utc" in fragment


async def test_that_a_fragment_can_be_read(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    fragment_store = container[FragmentStore]

    value = "Your account balance is {balance}"
    fields = [FragmentField(name="balance", description="Account's balance", examples=["9000"])]

    fragment = await fragment_store.create_fragment(value=value, fields=fields)

    response = await async_client.get(f"/fragments/{fragment.id}")
    assert response.status_code == status.HTTP_200_OK

    data = response.json()
    assert data["id"] == fragment.id
    assert data["value"] == value

    assert len(data["fields"]) == 1
    fragment_field = data["fields"][0]
    assert fragment_field["name"] == fields[0].name
    assert fragment_field["description"] == fields[0].description
    assert fragment_field["examples"] == fields[0].examples

    assert dateutil.parser.parse(data["creation_utc"]) == fragment.creation_utc


async def test_that_all_fragments_can_be_listed(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    fragment_store = container[FragmentStore]

    first_value = "Your account balance is {balance}"
    first_fields = [
        FragmentField(name="balance", description="Account's balance", examples=["9000"])
    ]

    second_value = "It will take {days_number} days to deliver to {address}"
    second_fields = [
        FragmentField(
            name="days_number", description="Time required for delivery in days", examples=["8"]
        ),
        FragmentField(name="address", description="Customer's address", examples=["Some Address"]),
    ]

    await fragment_store.create_fragment(value=first_value, fields=first_fields)
    await fragment_store.create_fragment(value=second_value, fields=second_fields)

    response = await async_client.get("/fragments")
    assert response.status_code == status.HTTP_200_OK
    fragments = response.json()

    assert len(fragments) >= 2
    assert any(f["value"] == first_value for f in fragments)
    assert any(f["value"] == second_value for f in fragments)


async def test_that_a_fragment_can_be_updated(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    fragment_store = container[FragmentStore]

    value = "Your account balance is {balance}"
    fields = [FragmentField(name="balance", description="Account's balance", examples=["9000"])]

    fragment = await fragment_store.create_fragment(value=value, fields=fields)

    update_payload = {
        "value": "Updated balance: {balance}",
        "fields": [
            {
                "name": "balance",
                "description": "Updated account balance",
                "examples": ["10000"],
            }
        ],
    }

    response = await async_client.patch(f"/fragments/{fragment.id}", json=update_payload)
    assert response.status_code == status.HTTP_200_OK

    updated_fragment = response.json()
    assert updated_fragment["value"] == update_payload["value"]
    assert updated_fragment["fields"] == update_payload["fields"]


async def test_that_a_fragment_can_be_deleted(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    fragment_store = container[FragmentStore]

    value = "Your account balance is {balance}"
    fields = [FragmentField(name="balance", description="Account's balance", examples=["9000"])]

    fragment = await fragment_store.create_fragment(value=value, fields=fields)

    delete_response = await async_client.delete(f"/fragments/{fragment.id}")
    assert delete_response.status_code == status.HTTP_204_NO_CONTENT

    with raises(ItemNotFoundError):
        await fragment_store.read_fragment(fragment.id)


async def test_that_a_tag_can_be_added_to_a_fragment(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    fragment_store = container[FragmentStore]
    tag_store = container[TagStore]

    tag = await tag_store.create_tag(name="VIP")

    value = "Your account balance is {balance}"
    fields = [FragmentField(name="balance", description="Account's balance", examples=["9000"])]

    fragment = await fragment_store.create_fragment(value=value, fields=fields)

    response = await async_client.patch(
        f"/fragments/{fragment.id}", json={"tags": {"add": [tag.id]}}
    )
    assert response.status_code == status.HTTP_200_OK

    updated_fragment = await fragment_store.read_fragment(fragment.id)
    assert tag.id in updated_fragment.tags


async def test_that_a_tag_can_be_removed_from_a_fragment(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    fragment_store = container[FragmentStore]
    tag_store = container[TagStore]

    tag = await tag_store.create_tag(name="VIP")

    value = "Your account balance is {balance}"
    fields = [FragmentField(name="balance", description="Account's balance", examples=["9000"])]

    fragment = await fragment_store.create_fragment(value=value, fields=fields)

    await fragment_store.add_tag(fragment_id=fragment.id, tag_id=tag.id)
    response = await async_client.patch(
        f"/fragments/{fragment.id}", json={"tags": {"remove": [tag.id]}}
    )
    assert response.status_code == status.HTTP_200_OK

    updated_fragment = await fragment_store.read_fragment(fragment.id)
    assert tag.id not in updated_fragment.tags


async def test_that_fragments_can_be_filtered_by_tags(
    async_client: httpx.AsyncClient,
    container: Container,
) -> None:
    fragment_store = container[FragmentStore]
    tag_store = container[TagStore]

    tag_vip = await tag_store.create_tag(name="VIP")
    tag_finance = await tag_store.create_tag(name="Finance")
    tag_greeting = await tag_store.create_tag(name="Greeting")

    first_fragment = await fragment_store.create_fragment(
        value="Welcome {username}!",
        fields=[
            FragmentField(name="username", description="User's name", examples=["Alice", "Bob"])
        ],
    )
    await fragment_store.add_tag(first_fragment.id, tag_greeting.id)

    second_fragment = await fragment_store.create_fragment(
        value="Your balance is {balance}",
        fields=[
            FragmentField(name="balance", description="Account balance", examples=["5000", "10000"])
        ],
    )
    await fragment_store.add_tag(second_fragment.id, tag_finance.id)

    third_fragment = await fragment_store.create_fragment(
        value="Exclusive VIP offer for {username}",
        fields=[FragmentField(name="username", description="VIP customer", examples=["Charlie"])],
    )
    await fragment_store.add_tag(third_fragment.id, tag_vip.id)

    response = await async_client.get(f"/fragments?tags={tag_greeting.id}")
    assert response.status_code == status.HTTP_200_OK
    fragments = response.json()
    assert len(fragments) == 1
    assert fragments[0]["value"] == "Welcome {username}!"

    response = await async_client.get(f"/fragments?tags={tag_finance.id}&tags={tag_vip.id}")
    assert response.status_code == status.HTTP_200_OK
    fragments = response.json()
    assert len(fragments) == 2
    values = {f["value"] for f in fragments}
    assert "Your balance is {balance}" in values
    assert "Exclusive VIP offer for {username}" in values

    response = await async_client.get("/fragments?tags=non_existent_tag")
    assert response.status_code == status.HTTP_200_OK
    fragments = response.json()
    assert len(fragments) == 0
