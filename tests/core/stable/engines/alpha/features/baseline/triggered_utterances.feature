Feature: Utterances
    Background:
        Given the alpha engine
        And an agent
        And an empty session

    Scenario: The agent utters a message aligned with an action to buy time
        Given a canned response request "inform the customer that more information is coming", to buy time
        When uttering is triggered
        Then a single message event is emitted
        And the message mentions that more information is coming

    Scenario: The agent utters a message aligned with an action to follow up with the customer
        Given a canned response request "suggest proceeding to checkout", to follow up with the customer
        When uttering is triggered
        Then a single message event is emitted
        And the message mentions proceeding to checkout
