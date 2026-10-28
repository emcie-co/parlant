Feature: Relationship

    Background:
        Given the engine
        And an agent
        And an empty session


    Scenario: The agent follows a policy that is entailed by another policy
        Given the engine
        And an agent whose job is to sell pizza
        And an empty session
        And a customer message, "Hi"
        And a policy "howdy" to greet the customer with "Howdy" when the customer says hello
        And a policy "good_sir" to add "good sir" when saying "Howdy"
        And a policy relationship whereby "howdy" entails "good_sir"
        When processing is triggered
        Then a single message event is emitted
        And the message contains a greeting with "Howdy" and "good sir"
