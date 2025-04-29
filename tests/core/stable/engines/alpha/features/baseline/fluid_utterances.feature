Feature: Fluid Utterance
    Background:
        Given the alpha engine
        And an agent
        And that the agent uses the fluid_utterance message composition mode
        And an empty session

    Scenario: The agent greets the customer (fluid utterance)
        Given a guideline to greet with 'Howdy' when the session starts
        When processing is triggered
        Then a status event is emitted, acknowledging event -1
        And a status event is emitted, processing event -1
        And a status event is emitted, typing in response to event -1
        And a single message event is emitted
        And the message contains a 'Howdy' greeting
        And a status event is emitted, ready for further engagement after reacting to event -1

    Scenario: Responding based on data the user is providing (fluid utterance)
        Given a customer message, "I say that a banana is green, and an apple is purple. What did I say was the color of a banana?"
        And an utterance, "Sorry, I do not know"
        And an utterance, "The answer is {{generative.answer}}"
        When messages are emitted
        Then the message doesn't contain the text "I do not know"
        And the message mentions the color green
