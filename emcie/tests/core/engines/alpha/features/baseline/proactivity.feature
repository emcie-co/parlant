Feature: Proactivity
    Background:
        Given the alpha engine
        And an agent
        And an empty session

    Scenario: The agent does not start a conversation if no proactive guidelines exist
        Given a context variable "account_balance" with a value of "-$207.05"
        When processing is triggered
        Then a typing status event is not emitted
        And no message events are emitted


    Scenario: The agent starts a conversation based on context values
        Given a context variable "account_balance" with a value of "-$207.05"
        And a guideline to offer the user a loan when the user's account is overdrawn
        When processing is triggered
        Then a single message event is emitted
        And the message contains an offering of a loan
