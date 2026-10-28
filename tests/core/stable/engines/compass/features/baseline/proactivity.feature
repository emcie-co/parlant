Feature: Proactivity
    Background:
        Given the engine
        And an agent
        And an empty session

    Scenario: The agent starts a conversation based on context values
        Given a context variable "account_balance" set to "-$207.05"
        And a policy to offer the customer a loan when the customer's account is overdrawn
        When processing is triggered
        Then a single message event is emitted
        And the message contains an offering of a loan
