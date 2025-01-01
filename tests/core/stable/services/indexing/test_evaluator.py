# Copyright 2024 Emcie Co Ltd.
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

import asyncio
from typing import cast

from lagom import Container
from pytest import raises

from parlant.core.agents import Agent
from parlant.core.evaluations import (
    EvaluationListener,
    EvaluationStatus,
    EvaluationStore,
    GuidelinePayload,
    GuidelineInvoiceData,
    PayloadDescriptor,
    PayloadKind,
    StyleGuidePayload,
)
from parlant.core.guidelines import GuidelineContent, GuidelineStore
from parlant.core.services.indexing.behavioral_change_evaluation import (
    BehavioralChangeEvaluator,
    EvaluationValidationError,
)

from parlant.core.style_guides import StyleGuideContent, StyleGuideStore
from tests.conftest import NoCachedGenerations

AMOUNT_OF_TIME_TO_WAIT_FOR_EVALUATION_TO_START_RUNNING = 0.3


async def test_that_a_new_guideline_evaluation_starts_with_a_pending_status(
    container: Container,
    agent: Agent,
) -> None:
    evaluation_service = container[BehavioralChangeEvaluator]
    evaluation_store = container[EvaluationStore]

    evaluation_id = await evaluation_service.create_evaluation_task(
        agent=agent,
        payload_descriptors=[
            PayloadDescriptor(
                kind=PayloadKind.GUIDELINE,
                payload=GuidelinePayload(
                    content=GuidelineContent(
                        condition="the customer greets you",
                        action="greet them back with 'Hello'",
                    ),
                    operation="add",
                    coherence_check=True,
                    connection_proposition=True,
                ),
            ),
        ],
    )

    evaluation = await evaluation_store.read_evaluation(evaluation_id)

    assert evaluation.status == EvaluationStatus.PENDING


async def test_that_a_new_style_guide_evaluation_starts_with_a_pending_status(
    container: Container,
    agent: Agent,
) -> None:
    evaluation_service = container[BehavioralChangeEvaluator]
    evaluation_store = container[EvaluationStore]

    evaluation_id = await evaluation_service.create_evaluation_task(
        agent=agent,
        payload_descriptors=[
            PayloadDescriptor(
                kind=PayloadKind.STYLE_GUIDE,
                payload=StyleGuidePayload(
                    content=StyleGuideContent(
                        principle="the customer greets you",
                        examples=[],
                    ),
                    operation="add",
                    coherence_check=True,
                ),
            ),
        ],
    )

    evaluation = await evaluation_store.read_evaluation(evaluation_id)

    assert evaluation.status == EvaluationStatus.PENDING


async def test_that_a_guideline_evaluation_completes_when_all_invoices_have_data(
    container: Container,
    agent: Agent,
) -> None:
    evaluation_service = container[BehavioralChangeEvaluator]
    evaluation_store = container[EvaluationStore]
    evaluation_listener = container[EvaluationListener]

    evaluation_id = await evaluation_service.create_evaluation_task(
        agent=agent,
        payload_descriptors=[
            PayloadDescriptor(
                PayloadKind.GUIDELINE,
                GuidelinePayload(
                    content=GuidelineContent(
                        condition="the customer greets you",
                        action="greet them back with 'Hello'",
                    ),
                    operation="add",
                    coherence_check=True,
                    connection_proposition=True,
                ),
            )
        ],
    )

    assert await evaluation_listener.wait_for_completion(evaluation_id)

    evaluation = await evaluation_store.read_evaluation(evaluation_id)

    assert evaluation.status == EvaluationStatus.COMPLETED

    assert len(evaluation.invoices) == 1

    assert evaluation.invoices[0].approved

    assert evaluation.invoices[0].data
    invoice_data = cast(GuidelineInvoiceData, evaluation.invoices[0].data)

    assert invoice_data.coherence_checks == []
    assert invoice_data.connection_propositions is None


async def test_that_a_style_guide_evaluation_completes_when_all_invoices_have_data(
    container: Container,
    agent: Agent,
) -> None:
    evaluation_service = container[BehavioralChangeEvaluator]
    evaluation_store = container[EvaluationStore]
    evaluation_listener = container[EvaluationListener]

    evaluation_id = await evaluation_service.create_evaluation_task(
        agent=agent,
        payload_descriptors=[
            PayloadDescriptor(
                PayloadKind.STYLE_GUIDE,
                payload=StyleGuidePayload(
                    content=StyleGuideContent(
                        principle="the customer greets you",
                        examples=[],
                    ),
                    operation="add",
                    coherence_check=True,
                ),
            )
        ],
    )

    assert await evaluation_listener.wait_for_completion(evaluation_id)

    evaluation = await evaluation_store.read_evaluation(evaluation_id)

    assert evaluation.status == EvaluationStatus.COMPLETED

    assert len(evaluation.invoices) == 1

    assert evaluation.invoices[0].approved

    assert evaluation.invoices[0].data
    invoice_data = cast(GuidelineInvoiceData, evaluation.invoices[0].data)

    assert invoice_data.coherence_checks == []


async def test_that_an_evaluation_of_a_coherent_guideline_completes_with_an_approved_invoice(
    container: Container,
    agent: Agent,
) -> None:
    guideline_store = container[GuidelineStore]
    evaluation_service = container[BehavioralChangeEvaluator]
    evaluation_store = container[EvaluationStore]
    evaluation_listener = container[EvaluationListener]

    await guideline_store.create_guideline(
        guideline_set=agent.id,
        condition="a customer inquires about upgrading their service package",
        action="provide information on available upgrade options and benefits",
    )

    evaluation_id = await evaluation_service.create_evaluation_task(
        agent=agent,
        payload_descriptors=[
            PayloadDescriptor(
                PayloadKind.GUIDELINE,
                GuidelinePayload(
                    content=GuidelineContent(
                        condition="a customer needs assistance with understanding their billing statements",
                        action="guide them through the billing details and explain any charges",
                    ),
                    operation="add",
                    coherence_check=True,
                    connection_proposition=True,
                ),
            )
        ],
    )

    assert await evaluation_listener.wait_for_completion(evaluation_id)

    evaluation = await evaluation_store.read_evaluation(evaluation_id)

    assert evaluation.status == EvaluationStatus.COMPLETED

    assert len(evaluation.invoices) == 1

    assert evaluation.invoices[0].approved

    assert evaluation.invoices[0].data
    invoice_data = cast(GuidelineInvoiceData, evaluation.invoices[0].data)

    assert invoice_data.coherence_checks == []
    assert invoice_data.connection_propositions is None


async def test_that_an_evaluation_of_an_incoherent_guideline_completes_with_an_unapproved_invoice(
    container: Container,
    agent: Agent,
) -> None:
    guideline_store = container[GuidelineStore]
    evaluation_service = container[BehavioralChangeEvaluator]
    evaluation_store = container[EvaluationStore]
    evaluation_listener = container[EvaluationListener]

    await guideline_store.create_guideline(
        guideline_set=agent.id,
        condition="A VIP customer requests a specific feature that aligns with their business needs but is not on the current product roadmap",
        action="Escalate the request to product management for special consideration",
    )

    evaluation_id = await evaluation_service.create_evaluation_task(
        agent=agent,
        payload_descriptors=[
            PayloadDescriptor(
                PayloadKind.GUIDELINE,
                GuidelinePayload(
                    content=GuidelineContent(
                        condition="Any customer requests a feature not available in the current version",
                        action="Inform them that upcoming features are added only according to the roadmap",
                    ),
                    operation="add",
                    coherence_check=True,
                    connection_proposition=True,
                ),
            )
        ],
    )

    assert await evaluation_listener.wait_for_completion(evaluation_id)

    evaluation = await evaluation_store.read_evaluation(evaluation_id)

    assert evaluation.status == EvaluationStatus.COMPLETED

    assert len(evaluation.invoices) == 1

    assert evaluation.invoices[0]
    assert not evaluation.invoices[0].approved

    assert evaluation.invoices[0].data
    invoice_data = cast(GuidelineInvoiceData, evaluation.invoices[0].data)

    assert len(invoice_data.coherence_checks) == 1
    assert invoice_data.connection_propositions is None


async def test_that_an_evaluation_of_incoherent_proposed_guidelines_completes_with_an_unapproved_invoice(
    container: Container,
    agent: Agent,
) -> None:
    evaluation_service = container[BehavioralChangeEvaluator]
    evaluation_store = container[EvaluationStore]
    evaluation_listener = container[EvaluationListener]

    evaluation_id = await evaluation_service.create_evaluation_task(
        agent=agent,
        payload_descriptors=[
            PayloadDescriptor(
                PayloadKind.GUIDELINE,
                GuidelinePayload(
                    content=GuidelineContent(
                        condition="any customer requests a feature not available in the current version",
                        action="Inform them that upcoming features are added only according to the roadmap",
                    ),
                    operation="add",
                    coherence_check=True,
                    connection_proposition=True,
                ),
            ),
            PayloadDescriptor(
                PayloadKind.GUIDELINE,
                GuidelinePayload(
                    content=GuidelineContent(
                        condition="a VIP customer requests a specific feature that aligns with their business needs but is not on the current product roadmap",
                        action="escalate the request to product management for special consideration",
                    ),
                    operation="add",
                    coherence_check=True,
                    connection_proposition=True,
                ),
            ),
        ],
    )

    assert await evaluation_listener.wait_for_completion(evaluation_id)

    evaluation = await evaluation_store.read_evaluation(evaluation_id)

    assert evaluation.status == EvaluationStatus.COMPLETED

    assert len(evaluation.invoices) == 2

    assert evaluation.invoices[0]
    assert not evaluation.invoices[0].approved

    assert evaluation.invoices[0].data
    assert len(evaluation.invoices[0].data.coherence_checks) == 1

    assert evaluation.invoices[1].data
    assert len(evaluation.invoices[1].data.coherence_checks) == 1


async def test_that_an_evaluation_of_multiple_payloads_completes_with_an_invoice_containing_data_for_each(
    container: Container,
    agent: Agent,
) -> None:
    evaluation_service = container[BehavioralChangeEvaluator]
    evaluation_store = container[EvaluationStore]
    evaluation_listener = container[EvaluationListener]

    evaluation_id = await evaluation_service.create_evaluation_task(
        agent=agent,
        payload_descriptors=[
            PayloadDescriptor(
                PayloadKind.GUIDELINE,
                GuidelinePayload(
                    content=GuidelineContent(
                        condition="the customer greets you",
                        action="greet them back with 'Hello'",
                    ),
                    operation="add",
                    coherence_check=True,
                    connection_proposition=True,
                ),
            ),
            PayloadDescriptor(
                PayloadKind.GUIDELINE,
                GuidelinePayload(
                    content=GuidelineContent(
                        condition="the customer asks about the weather",
                        action="provide a weather update",
                    ),
                    operation="add",
                    coherence_check=True,
                    connection_proposition=True,
                ),
            ),
        ],
    )

    assert await evaluation_listener.wait_for_completion(evaluation_id)

    evaluation = await evaluation_store.read_evaluation(evaluation_id)

    assert evaluation.status == EvaluationStatus.COMPLETED
    assert len(evaluation.invoices) == 2

    for invoice in evaluation.invoices:
        assert invoice.approved

        assert invoice.data
        assert invoice.data.coherence_checks == []
        assert cast(GuidelineInvoiceData, invoice.data).connection_propositions is None


async def test_that_an_evaluation_that_failed_due_to_already_running_evaluation_task_contains_its_error_details(
    container: Container,
    agent: Agent,
    no_cache: NoCachedGenerations,
) -> None:
    evaluation_service = container[BehavioralChangeEvaluator]
    evaluation_store = container[EvaluationStore]

    first_evaluation_id = await evaluation_service.create_evaluation_task(
        agent=agent,
        payload_descriptors=[
            PayloadDescriptor(
                PayloadKind.GUIDELINE,
                GuidelinePayload(
                    content=GuidelineContent(
                        condition="the customer greets you",
                        action="greet them back with 'Hello'",
                    ),
                    operation="add",
                    coherence_check=True,
                    connection_proposition=True,
                ),
            ),
            PayloadDescriptor(
                PayloadKind.GUIDELINE,
                GuidelinePayload(
                    content=GuidelineContent(
                        condition="the customer greets you",
                        action="greet them back with 'Hola'",
                    ),
                    operation="add",
                    coherence_check=True,
                    connection_proposition=True,
                ),
            ),
        ],
    )

    second_payloads = [
        GuidelinePayload(
            content=GuidelineContent(
                condition="the customer asks about the weather",
                action="provide a weather update",
            ),
            operation="add",
            coherence_check=True,
            connection_proposition=True,
        )
    ]

    await asyncio.sleep(AMOUNT_OF_TIME_TO_WAIT_FOR_EVALUATION_TO_START_RUNNING)

    second_evaluation_id = await evaluation_service.create_evaluation_task(
        agent=agent,
        payload_descriptors=[PayloadDescriptor(PayloadKind.GUIDELINE, p) for p in second_payloads],
    )

    await asyncio.sleep(AMOUNT_OF_TIME_TO_WAIT_FOR_EVALUATION_TO_START_RUNNING)

    evaluation = await evaluation_store.read_evaluation(second_evaluation_id)

    assert evaluation.status == EvaluationStatus.FAILED
    assert evaluation.error == f"An evaluation task '{first_evaluation_id}' is already running."


async def test_that_an_evaluation_validation_failed_due_to_guidelines_duplication_in_the_payloads_contains_relevant_error_details(
    container: Container,
    agent: Agent,
) -> None:
    evaluation_service = container[BehavioralChangeEvaluator]

    duplicate_payload = GuidelinePayload(
        content=GuidelineContent(
            condition="the customer greets you",
            action="greet them back with 'Hello'",
        ),
        operation="add",
        coherence_check=True,
        connection_proposition=True,
    )

    with raises(EvaluationValidationError) as exc:
        await evaluation_service.create_evaluation_task(
            agent=agent,
            payload_descriptors=[
                PayloadDescriptor(PayloadKind.GUIDELINE, p)
                for p in [
                    duplicate_payload,
                    duplicate_payload,
                ]
            ],
        )
        await asyncio.sleep(AMOUNT_OF_TIME_TO_WAIT_FOR_EVALUATION_TO_START_RUNNING)

    assert str(exc.value) == "Duplicate guideline found among the provided guidelines."


async def test_that_an_evaluation_validation_failed_due_to_style_guides_duplication_in_the_payloads_contains_relevant_error_details(
    container: Container,
    agent: Agent,
) -> None:
    evaluation_service = container[BehavioralChangeEvaluator]

    duplicate_payload = StyleGuidePayload(
        content=StyleGuideContent(
            principle="Be extremely formal",
            examples=[],
        ),
        operation="add",
        coherence_check=True,
    )

    with raises(EvaluationValidationError) as exc:
        await evaluation_service.create_evaluation_task(
            agent=agent,
            payload_descriptors=[
                PayloadDescriptor(PayloadKind.STYLE_GUIDE, p)
                for p in [
                    duplicate_payload,
                    duplicate_payload,
                ]
            ],
        )
        await asyncio.sleep(AMOUNT_OF_TIME_TO_WAIT_FOR_EVALUATION_TO_START_RUNNING)

    assert str(exc.value) == "Duplicate style guide found among the provided style guides."


async def test_that_an_evaluation_validation_failed_due_to_duplicate_guidelines_with_existing_contains_relevant_error_details(
    container: Container,
    agent: Agent,
) -> None:
    evaluation_service = container[BehavioralChangeEvaluator]
    guideline_store = container[GuidelineStore]

    await guideline_store.create_guideline(
        guideline_set=agent.id,
        condition="the customer greets you",
        action="greet them back with 'Hello'",
    )

    with raises(EvaluationValidationError) as exc:
        await evaluation_service.create_evaluation_task(
            agent=agent,
            payload_descriptors=[
                PayloadDescriptor(
                    PayloadKind.GUIDELINE,
                    GuidelinePayload(
                        content=GuidelineContent(
                            condition="the customer greets you",
                            action="greet them back with 'Hello'",
                        ),
                        operation="add",
                        coherence_check=True,
                        connection_proposition=True,
                    ),
                )
            ],
        )
        await asyncio.sleep(AMOUNT_OF_TIME_TO_WAIT_FOR_EVALUATION_TO_START_RUNNING)

    assert (
        str(exc.value)
        == f"Duplicate guideline found against existing guideline: When the customer greets you, then greet them back with 'Hello' in {agent.id} guideline_set"
    )


async def test_that_an_evaluation_validation_failed_due_to_duplicate_style_guides_with_existing_contains_relevant_error_details(
    container: Container,
    agent: Agent,
) -> None:
    evaluation_service = container[BehavioralChangeEvaluator]
    style_guide_store = container[StyleGuideStore]

    await style_guide_store.create_style_guide(
        style_guide_set=agent.id,
        principle="Be extremely formal",
        examples=[],
    )

    with raises(EvaluationValidationError) as exc:
        await evaluation_service.create_evaluation_task(
            agent=agent,
            payload_descriptors=[
                PayloadDescriptor(
                    PayloadKind.STYLE_GUIDE,
                    StyleGuidePayload(
                        content=StyleGuideContent(
                            principle="Be extremely formal",
                            examples=[],
                        ),
                        operation="add",
                        coherence_check=True,
                    ),
                )
            ],
        )
        await asyncio.sleep(AMOUNT_OF_TIME_TO_WAIT_FOR_EVALUATION_TO_START_RUNNING)

    assert (
        str(exc.value)
        == f"Duplicate style guide found against existing style guide: Be extremely formal in {agent.id} style_guide_set"
    )


async def test_that_an_evaluation_completes_and_contains_a_connection_proposition_with_an_existing_guideline(
    container: Container,
    agent: Agent,
) -> None:
    guideline_store = container[GuidelineStore]
    evaluation_service = container[BehavioralChangeEvaluator]
    evaluation_store = container[EvaluationStore]
    evaluation_listener = container[EvaluationListener]

    await guideline_store.create_guideline(
        guideline_set=agent.id,
        condition="the customer asks about the weather",
        action="provide the current weather update",
    )

    evaluation_id = await evaluation_service.create_evaluation_task(
        agent=agent,
        payload_descriptors=[
            PayloadDescriptor(
                PayloadKind.GUIDELINE,
                GuidelinePayload(
                    content=GuidelineContent(
                        condition="providing the weather update",
                        action="mention the best time to go for a walk",
                    ),
                    operation="add",
                    coherence_check=True,
                    connection_proposition=True,
                ),
            )
        ],
    )

    assert await evaluation_listener.wait_for_completion(evaluation_id)

    evaluation = await evaluation_store.read_evaluation(evaluation_id)

    assert evaluation.status == EvaluationStatus.COMPLETED

    assert len(evaluation.invoices) == 1
    assert evaluation.invoices[0].data
    invoice_data = cast(GuidelineInvoiceData, evaluation.invoices[0].data)

    assert invoice_data.connection_propositions
    assert len(invoice_data.connection_propositions) == 1
    assert (
        invoice_data.connection_propositions[0].check_kind == "connection_with_existing_guideline"
    )

    assert invoice_data.connection_propositions
    assert (
        invoice_data.connection_propositions[0].source.action
        == "provide the current weather update"
    )
    assert (
        invoice_data.connection_propositions[0].target.condition == "providing the weather update"
    )


async def test_that_an_evaluation_completes_and_contains_connection_proposition_between_evaluated_guidelines(
    container: Container,
    agent: Agent,
) -> None:
    evaluation_service = container[BehavioralChangeEvaluator]
    evaluation_store = container[EvaluationStore]
    evaluation_listener = container[EvaluationListener]

    evaluation_id = await evaluation_service.create_evaluation_task(
        agent=agent,
        payload_descriptors=[
            PayloadDescriptor(
                PayloadKind.GUIDELINE,
                GuidelinePayload(
                    content=GuidelineContent(
                        condition="the customer asks about the weather",
                        action="provide the current weather update",
                    ),
                    operation="add",
                    coherence_check=True,
                    connection_proposition=True,
                ),
            ),
            PayloadDescriptor(
                PayloadKind.GUIDELINE,
                GuidelinePayload(
                    content=GuidelineContent(
                        condition="providing the weather update",
                        action="mention the best time to go for a walk",
                    ),
                    operation="add",
                    coherence_check=True,
                    connection_proposition=True,
                ),
            ),
        ],
    )

    assert await evaluation_listener.wait_for_completion(evaluation_id)

    evaluation = await evaluation_store.read_evaluation(evaluation_id)

    assert evaluation.status == EvaluationStatus.COMPLETED

    assert len(evaluation.invoices) == 2
    assert evaluation.invoices[0].data
    invoice_data = cast(GuidelineInvoiceData, evaluation.invoices[0].data)

    assert invoice_data.connection_propositions
    assert len(invoice_data.connection_propositions) == 1
    assert (
        invoice_data.connection_propositions[0].check_kind
        == "connection_with_another_evaluated_guideline"
    )
    assert (
        invoice_data.connection_propositions[0].source.condition
        == "the customer asks about the weather"
    )
    assert (
        invoice_data.connection_propositions[0].target.condition == "providing the weather update"
    )

    assert evaluation.invoices[1].data
    invoice_data = cast(GuidelineInvoiceData, evaluation.invoices[1].data)

    assert invoice_data.connection_propositions
    assert len(invoice_data.connection_propositions) == 1
    assert (
        invoice_data.connection_propositions[0].check_kind
        == "connection_with_another_evaluated_guideline"
    )

    assert (
        invoice_data.connection_propositions[0].source.action
        == "provide the current weather update"
    )
    assert (
        invoice_data.connection_propositions[0].target.condition == "providing the weather update"
    )
