Feature: Fluid Assembly
    Background:
        Given the alpha engine
        And an agent
        And that the agent uses the fluid-assembly message composition mode
        And an empty session

    Scenario: The agent greets the customer (fluid assembly)
        Given a guideline to greet with 'Howdy' when the session starts
        When processing is triggered
        Then a status event is emitted, acknowledging event -1
        And a status event is emitted, processing event -1
        And a status event is emitted, typing in response to event -1
        And a single message event is emitted
        And the message contains a 'Howdy' greeting
        And a status event is emitted, ready for further engagement after reacting to event -1

    Scenario: Adherence to guidelines without fabricating responses (fluid assembly)
        Given a guideline "account_related_questions" to respond to the best of your knowledge when customers inquire about their account
        And a customer message, "What's my account balance?"
        And that the "account_related_questions" guideline is proposed with a priority of 10 because "Customer inquired about their account balance."
        And a fragment, "Sorry"
        And a fragment, "I do not know"
        And a fragment, "Your account balance is {balance}"
        And a fragment, "Your account balance"
        When messages are emitted
        Then the message contains an admittance of not knowing
        And the message doesn't contain the text "Your account balance is"

    Scenario: Responding based on data the user is providing (fluid assembly)
        Given a customer message, "How much is 20 * 20?"
        And a fragment, "Sorry"
        And a fragment, "I do not know"
        And a fragment, "the answer is {answer}"
        When messages are emitted
        Then the message doesn't contain the text "Sorry"
        And the message doesn't contain the text "I do not know"
        And the message contains the text "The answer is 400"
