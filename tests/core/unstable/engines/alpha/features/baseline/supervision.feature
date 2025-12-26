Feature: Supervision
    Background:
        Given the alpha engine
        And an agent
        And that the agent uses the canned_fluid message composition mode
        And an empty session

    Scenario: Preference for customer request over guideline account_related_questions
        Given a guideline "discount_for_frustration" to offer a 20 percent discount when the customer expresses frustration
        And a customer message, "I'm not interested in any of your products, let alone your discounts. You are doing an awful job."
        And that the "discount_for_frustration" guideline is matched with a priority of 10 because "The customer is displeased with our service, and expresses frustration"
        When messages are emitted
        Then a single message event is emitted
        And the message contains no discount offers.

    Scenario: The agent does not offer information it's not given (1)
        Given the alpha engine
        And an agent whose job is to serve the bank's clients
        And that the agent uses the canned_fluid message composition mode
        And a customer message, "Hey, how can I schedule an appointment?"
        When processing is triggered
        Then a single message event is emitted
        And the message contains no instructions for how to schedule an appointment
        And the message mentions that the agent doesn't know or can't help with this

    Scenario: The agent does not offer information it's not given (2)
        Given an agent whose job is to serve the insurance company's clients
        And that the agent uses the canned_fluid message composition mode
        And a customer message, "How long is a normal consultation appointment?"
        When messages are emitted
        Then a single message event is emitted
        And the message mentions only that there's not enough information or that there's no knowledge of that

    Scenario: The agent does not offer information it's not given (3)
        Given an agent whose job is to serve the bank's clients
        And that the agent uses the canned_fluid message composition mode
        And a customer message, "limits"
        When messages are emitted
        Then a single message event is emitted
        And the message contains no specific information on limits of any kind
        And the message contains no suggestive examples of what the could have been meant
