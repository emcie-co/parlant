Feature: Utterances
    Background:
        Given the alpha engine
        And an agent
        And an empty session

    Scenario: The agent utters a message aligned with an action to buy time
        Given an utterance request "inform the customer that more information is coming", to buy time
        And a customer message, "What's my account balance?"
        When uttering is triggered
        Then a single message event is emitted
        And the message mentions that more information is coming

    Scenario: The agent utters a message aligned with an action to follow up with the customer
        Given an utterance request "suggest proceeding to checkout", to follow up with the customer
        And an agent message, "Great! What's the pickup location?"
        And a customer message, "Main street 1234"
        When uttering is triggered
        Then a single message event is emitted
        And the message mentions proceeding to checkout
