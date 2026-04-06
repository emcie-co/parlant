Feature: Relationship

    Background:
        Given the alpha engine
        And an agent
        And that the agent uses the canned_fluid message composition mode
        And an empty session


    Scenario: The agent follows a guideline that is entailed by another guideline
        Given the alpha engine
        And an agent whose job is to sell pizza
        And an empty session
        And a customer message, "Hi"
        And a guideline "howdy" to greet the customer with "Howdy" when the customer says hello
        And a guideline "good_sir" to add "good sir" when saying "Howdy"
        And a guideline relationship whereby "howdy" entails "good_sir"
        When processing is triggered
        Then a single message event is emitted
        And the message contains a greeting with "Howdy" and "good sir"
