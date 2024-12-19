Feature: Utters
    Background:
        Given the alpha engine
        And an agent
        And an empty session

    Scenario: A buy-time message is determined by the actions sent from utter engine operation
        Given an utter action of "inform the user that more information is coming" with reason of "BUY_TIME"
        When triggering utter action is requested
        Then a single message event is emitted
        And the message contains that more information is coming

    Scenario: A follow-up message is determined by the actions sent from utter engine operation
        Given an utter action of "ask the user if he need assistant with the blue-yellow feature" with reason of "FOLLOW_UP"
        When triggering utter action is requested
        Then a single message event is emitted
        And the message contains asking the user if he need help with the blue-yellow feature