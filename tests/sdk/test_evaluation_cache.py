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

"""The SDK rule-evaluation cache must be sensitive to the glossary terms a
rule's evaluation depends on: a term added or edited at runtime should
invalidate exactly the cached evaluations whose relevant terms changed."""

from dataclasses import replace
from datetime import datetime, timezone

from parlant.core.common import generate_id
from parlant.core.glossary import Term, TermId
from parlant.core.rules import RuleContent
from parlant.sdk import _CachedEvaluator


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
    evaluator = object.__new__(_CachedEvaluator)
    return evaluator._hash_rule_evaluation_request(
        g=RuleContent(condition="a condition", action="an action", description=None),
        tool_ids=[],
        journey_state_propositions=False,
        properties_proposition=True,
        signal_proposition=True,
        title_proposition=False,
        title=None,
        agent_id=None,
        glossary_fingerprint=glossary_fingerprint,
    )


def test_that_the_evaluation_cache_hash_is_sensitive_to_the_glossary_fingerprint() -> None:
    assert _hash("fingerprint-a") == _hash("fingerprint-a")
    assert _hash("fingerprint-a") != _hash("fingerprint-b")


def test_that_the_glossary_fingerprint_reflects_term_content_but_not_order() -> None:
    evaluator = object.__new__(_CachedEvaluator)
    prs = _term("PRS", "Pinewood Rash Syndrome.")
    rma = _term("RMA-3", "The medical escalation procedure.")

    fingerprint = evaluator._glossary_fingerprint([prs, rma])

    assert fingerprint == evaluator._glossary_fingerprint([rma, prs])
    assert fingerprint != evaluator._glossary_fingerprint(
        [prs, replace(rma, description="A changed procedure.")]
    )
    assert evaluator._glossary_fingerprint([]) != fingerprint
