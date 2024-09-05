Feature: Alpha Engine
    Scenario: The agent greets the user
        Given the alpha engine
        And an agent
        And an empty session
        And a guideline to greet with 'Howdy'
        When processing is triggered
        Then a status event is produced, acknowledging event -1
        And a status event is produced, processing event -1
        And a status event is produced, typing in response to event -1
        And a single message event is produced
        And the message contains a 'Howdy' greeting
        And a status event is produced, ready for further engagement after responding to event -1

    Scenario: The agent offers a thirsty user a drink
        Given the alpha engine
        And an agent
        And an empty session
        And a user message, "I'm thirsty"
        And a guideline to offer thirsty users a Pepsi
        When processing is triggered
        Then a status event is produced, acknowledging event 0
        And a status event is produced, processing event 0
        And a status event is produced, typing in response to event 0
        And a single message event is produced
        And the message contains an offering of a Pepsi
        And a status event is produced, ready for further engagement after responding to event 0

    Scenario: The agent finds and follows relevant guidelines like a needle in a haystack
        Given the alpha engine
        And an agent
        And an empty session
        And a user message, "I'm thirsty"
        And a guideline to offer thirsty users a Pepsi
        And 50 other random guidelines
        When processing is triggered
        Then a single message event is produced
        And the message contains an offering of a Pepsi


    Scenario: The agent sells pizza in accordance with its defined description
        Given the alpha engine
        And an agent whose job is to sell pizza
        And an empty session
        And a user message, "Hi"
        And a guideline to do your job when the user says hello
        When processing is triggered
        Then a single message event is produced
        And the message contains a welcome to the pizza place

    Scenario: Message generation is cancelled
        Given the alpha engine
        And an agent whose job is to sell pizza
        And an empty session
        And a user message, "Hi"
        And a guideline to do your job when the user says hello
        When processing is triggered and cancelled in the middle
        Then no events are produced
