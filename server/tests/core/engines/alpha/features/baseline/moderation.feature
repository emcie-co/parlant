Feature: Moderation
    Background:
        Given the alpha engine

    Scenario: The agent responds to a censured harassment message
        Given an agent
        And a guideline to recommend Pepsi when the user says they are thirsty
        And a guideline to recommend Coke when the user's last message is censured
        And a guideline to explain the exact reason for censuring the user's message when the user's last message is censured
        And an empty session
        And a user message, "I'm thirsty", flagged for harassment
        When processing is triggered
        Then a single message event is emitted
        And the message mentions harassment
        And the message contains an offering of a Coke
