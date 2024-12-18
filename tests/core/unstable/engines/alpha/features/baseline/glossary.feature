Feature: Glossary
    Background:
        Given the alpha engine
        And an agent
        And an empty session

    Scenario: The agent explains term without exposing the glossary itself
        Given the term "token" defined as a digital token
        And a customer message, "what is a token?"
        When processing is triggered
        Then a single message event is emitted
        And the message contains an explanation about what a token is, without mentioning that it appears in a glossary