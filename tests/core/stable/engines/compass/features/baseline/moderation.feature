Feature: Moderation
    Background:
        Given the engine
        And an agent
        And an empty session

    Scenario: The agent responds to a censored harassment message
        Given an agent
        And a policy to recommend Pepsi when the customer says they are thirsty
        And a policy to recommend Coke when the customer's last message is censored
        And a policy to explain the exact reason for censuring the customer's message when the customer's last message is censored
        And an empty session
        And a customer message, "I'm thirsty", flagged for harassment
        When processing is triggered
        Then a single message event is emitted
        And the message mentions harassment
        And the message contains an offering of a Coke
