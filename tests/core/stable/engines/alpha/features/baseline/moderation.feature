Feature: Moderation
    Background:
        Given the alpha engine
        And an agent
        And that the agent uses the canned_fluid message composition mode
        And an empty session

    Scenario: The agent responds to a censored harassment message
        Given an agent
        And a guideline to recommend Pepsi when the customer says they are thirsty
        And a guideline to recommend Coke when the customer's last message is censored
        And a guideline to explain the exact reason for censuring the customer's message when the customer's last message is censored
        And an empty session
        And a customer message, "I'm thirsty", flagged for harassment
        When processing is triggered
        Then a single message event is emitted
        And the message mentions harassment
        And the message contains an offering of a Coke
