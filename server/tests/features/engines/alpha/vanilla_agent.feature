Feature: Alpha Engine
    Scenario: No message is emitted for an empty session
        Given the alpha engine
        And an agent
        And an empty session
        When processing is triggered
        Then no events are emitted

    Scenario: A single message event is emitted for a session with a user message
        Given the alpha engine
        And an agent
        And a session with a single user message
        When processing is triggered
        Then a single message event is emitted

    Scenario: A single message event is emitted for a session with a few messages
        Given the alpha engine
        And an agent
        And a session with a few messages
        When processing is triggered
        Then a single message event is emitted
