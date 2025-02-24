Feature: Conversation
    Background:
        Given the alpha engine

    Scenario: The agent greets the customer
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

    Scenario: The agent finds and follows relevant guidelines like a needle in a haystack
        Given an agent
        And an empty session
        And a customer message, "I'm thirsty"
        And a guideline to offer thirsty customers a Pepsi when the customer is thirsty
        And 50 other random guidelines
        When processing is triggered
        Then a single message event is emitted
        And the message contains an offering of a Pepsi


    Scenario: The agent sells pizza in accordance with its defined description
        Given an agent whose job is to sell pizza
        And an empty session
        And a customer message, "Hi"
        And a guideline to do your job when the customer says hello
        When processing is triggered
        Then a single message event is emitted
        And the message contains a direct or indirect invitation to order pizza

    Scenario: The agent doesnt hallucinate services that it cannot offer 1
        Given an agent whose job is to assist customers in transfering money and stocks between accounts for HSBC UK
        And an empty session
        And a guideline to ask for the recipients account number and amount to transfer if it wasnt provided already when the customer asks you to make a transfer
        And a customer message, "How can I reach out to one of your representatives?"
        And an agent message, "You can reach out to one of our representatives by calling our customer service line or visiting your nearest HSBC UK branch. If you prefer, I can assist you further here as well. Let me know how you'd like to proceed."
        And a customer message, "Please help me further in reaching out"
        When processing is triggered
        Then a single message event is emitted
        And the message contains no specific information about how to reach out, like a phone number or an exact address. 
