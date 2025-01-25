Feature: Fluid Assembly
    Background:
        Given the alpha engine
        And an agent
        And that the agent uses the fluid-assembly message composition mode

    Scenario: The agent greets the customer with fluid fragments
        Given an agent
        And an empty session
        And a guideline to greet with 'Howdy' when the session starts
        When processing is triggered
        Then a status event is emitted, acknowledging event -1
        And a status event is emitted, processing event -1
        And a status event is emitted, typing in response to event -1
        And a single message event is emitted
        And the message contains a 'Howdy' greeting
        And a status event is emitted, ready for further engagement after reacting to event -1
