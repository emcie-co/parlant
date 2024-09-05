Feature: Proactive agents
    Scenario: The agent does not start a conversation if no proactive guidelines exist
        Given the alpha engine
        And an agent
        And an empty session
        And a context variable "account_balance" with a value of "-$207.05"
        When processing is triggered
        Then no events are emitted


    Scenario: The agent starts a conversation based on context values
        Given the alpha engine
        And an agent
        And an empty session
        And a context variable "account_balance" with a value of "-$207.05"
        And a guideline to offer the user a loan when the user's account is overdrawn
        When processing is triggered
        Then a single message event is emitted
        And the message contains an offering of a loan
