Feature: Guideline Connections
    Scenario: The agent follows a guideline that is entailed by another guideline
        Given the alpha engine
        And an agent whose job is to sell pizza
        And an empty session
        And a user message, "Hi"
        And a guideline "howdy", to greet the user with "Howdy" when the user says hello
        And a guideline "good_sir", to add "good sir" when saying "Howdy"
        And a guideline connection whereby "howdy" entails "good_sir"
        When processing is triggered
        Then a single message event is produced
        And the message contains a greeting with "Howdy" and "good sir"