"""The live catalog records attachments, never executable callbacks or results."""

from dataclasses import FrozenInstanceError, asdict

import pytest

from parlant.core.retrievers import RetrieverRegistry


def test_that_agent_replacement_updates_metadata_without_duplicating_the_attachment() -> None:
    registry = RetrieverRegistry()
    first = registry.register_agent("agent-a", "profile", title="Old", description=None)
    replacement = registry.register_agent(
        "agent-a", "profile", title="Profile", description="Account facts"
    )
    assert replacement.id == first.id
    assert registry.list() == (replacement,)
    assert replacement.owner.kind == "agent"
    assert replacement.owner.id == "agent-a"
    assert replacement.title == "Profile"


def test_that_same_local_names_on_other_owners_cannot_alias() -> None:
    registry = RetrieverRegistry()
    a = registry.register_agent("a:b", "c", title=None, description=None)
    b = registry.register_agent("a", "b:c", title=None, description=None)
    c = registry.register_rule("a:b", "c", title=None, description=None)
    assert len({a.id, b.id, c.id}) == 3
    assert c.owner.kind == "rule"


def test_that_repeated_rule_hook_attachments_are_preserved_in_registration_order() -> None:
    registry = RetrieverRegistry()
    first = registry.register_rule("rule-a", "facts", title="First", description=None)
    second = registry.register_rule("rule-a", "facts", title="Second", description=None)
    assert first.id != second.id
    assert registry.list() == (first, second)


def test_that_listings_are_immutable_snapshots_containing_only_authored_metadata() -> None:
    registry = RetrieverRegistry()
    entry = registry.register_rule("rule-a", "facts", title=None, description=None)
    snapshot = registry.list()
    assert asdict(entry) == {
        "id": entry.id,
        "name": "facts",
        "title": None,
        "description": None,
        "owner": {"kind": "rule", "id": "rule-a"},
    }
    with pytest.raises(FrozenInstanceError):
        setattr(entry, "title", "Changed")
    registry.register_agent("a", "next", title=None, description=None)
    assert snapshot == (entry,)
    assert RetrieverRegistry().list() == ()
