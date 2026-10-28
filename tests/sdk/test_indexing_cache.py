# Copyright 2026 Emcie Co Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The SDK rule-indexing cache must be sensitive to the glossary terms a
rule's indexing depends on: a term added or edited at runtime should
invalidate exactly the cached indexing results whose relevant terms changed."""

from dataclasses import replace
from datetime import datetime, timezone

from parlant.core.common import generate_id
from parlant.core.glossary import Term, TermId
from parlant.sdk import _CachedIndexer, _RuleText


def _term(name: str, description: str) -> Term:
    now = datetime.now(timezone.utc)
    return Term(
        id=TermId(generate_id()),
        creation_utc=now,
        modified_utc=now,
        name=name,
        description=description,
        synonyms=[],
        groups=[],
    )


def _hash(glossary_fingerprint: str) -> str:
    indexer = object.__new__(_CachedIndexer)
    return indexer._hash_rule_indexing_request(
        g=_RuleText(condition="a condition", content="a description"),
        relativity_proposition=True,
        signal_proposition=True,
        title_proposition=False,
        title=None,
        agent_id=None,
        glossary_fingerprint=glossary_fingerprint,
    )


def test_that_the_indexing_cache_hash_is_sensitive_to_the_glossary_fingerprint() -> None:
    assert _hash("fingerprint-a") == _hash("fingerprint-a")
    assert _hash("fingerprint-a") != _hash("fingerprint-b")


def test_that_the_glossary_fingerprint_reflects_term_content_but_not_order() -> None:
    indexer = object.__new__(_CachedIndexer)
    prs = _term("PRS", "Pinewood Rash Syndrome.")
    rma = _term("RMA-3", "The medical escalation procedure.")

    fingerprint = indexer._glossary_fingerprint([prs, rma])

    assert fingerprint == indexer._glossary_fingerprint([rma, prs])
    assert fingerprint != indexer._glossary_fingerprint(
        [prs, replace(rma, description="A changed procedure.")]
    )
    assert indexer._glossary_fingerprint([]) != fingerprint
