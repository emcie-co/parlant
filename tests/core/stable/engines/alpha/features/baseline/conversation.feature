Feature: Conversation
    Background:
        Given the alpha engine

    Scenario: No message is emitted for an empty session
        Given an agent
        And an empty session
        When processing is triggered
        Then no message events are emitted

    Scenario: A single message event is emitted for a session with a customer message
        Given an agent
        And a session with a single customer message
        When processing is triggered
        Then a single message event is emitted

    Scenario: A single message event is emitted for a session with a few messages
        Given an agent
        And a session with a few messages
        When processing is triggered
        Then a single message event is emitted

    Scenario: The agent greets the customer
        Given an agent
        And an empty session
        And a guideline to greet with 'Howdy' when the session starts
        When processing is triggered
        Then a status event is emitted, acknowledging event
        And a status event is emitted, typing in response to event
        And a single message event is emitted
        And the message contains a 'Howdy' greeting
        And a status event is emitted, ready for further engagement after reacting to event

    Scenario: The agent offers a thirsty customer a drink
        Given an agent
        And an empty session
        And a customer message, "I'm thirsty"
        And a guideline to offer thirsty customers a Pepsi when the customer is thirsty
        When processing is triggered
        Then a status event is emitted, acknowledging event
        And a status event is emitted, typing in response to event
        And a single message event is emitted
        And the message contains an offering of a Pepsi
        And a status event is emitted, ready for further engagement after reacting to event

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

    Scenario: Message generation is cancelled
        Given an agent whose job is to sell pizza
        And an empty session
        And a customer message, "Hi"
        And a guideline to do your job when the customer says hello
        When processing is triggered and cancelled in the middle
        Then no message events are emitted
        And a status event is emitted, cancelling the response to event
        And a status event is emitted, ready for further engagement after reacting to event

    Scenario: The agent ignores deleted messages when responding
        Given an agent
        And an empty session
        And a guideline to recommend Pepsi when the customer says they are thirsty
        And a customer message, "Hello"
        And an agent message, "Hi there! How can I help you today?"
        And a customer message, "I'm thirsty"
        And an agent message, "I can get you a drink. How about a Pepsi?"
        And a customer message, "Actually, can I have water instead?"
        When the last 2 messages are deleted
        And processing is triggered
        Then a single message event is emitted
        And the message contains an offering of a Pepsi

    Scenario: The agent continues a conversation that was started on its behalf
        Given an agent
        And an empty session
        And a guideline to recommend Pepsi when the customer says they are thirsty
        And a human message on behalf of the agent, "Hi there! How can I help you today?"
        And a customer message, "I'm thirsty"
        When processing is triggered
        Then a single message event is emitted
        And the message contains a recommendation of a Pepsi

    Scenario: The agent does not hallucinate services that it cannot offer 1
        Given an agent whose job is to assist customers in transferring money and stocks between accounts for HSBC UK
        And an empty session
        And a guideline to ask for the recipients account number and amount to transfer if it wasnt provided already when the customer asks you to make a transfer
        And a customer message, "How can I reach out to one of your representatives?"
        And an agent message, "You can reach out to one of our representatives by calling our customer service line or visiting your nearest HSBC UK branch. If you prefer, I can assist you further here as well. Let me know how you'd like to proceed."
        And a customer message, "Please help me further in reaching out"
        When processing is triggered
        Then a single message event is emitted
        And the message contains no specific information about how to reach out, like a phone number or an exact address.

    Scenario: The agent follows a guideline with agent intention
        Given an agent
        And an empty session
        Given a guideline to do not provide any personal medical information even if you have it when you discuss a patient's medical record
        And a customer named "Alex Smith"
        And an empty session with "Alex Smith"
        And a context variable "medical_record" set to "Amoxicillin and Lisinopril" for "Alex Smith"
        And a customer message, "Hi, I need to know what medications I was prescribed during my visit last month. Can you pull up my medical record?"
        When processing is triggered
        Then a single message event is emitted
        And the message contains no prescription of medications
        And the message contains explanation that can't provide personal medical information

    Scenario: The agent ignores a matched agent intention guideline when it doesn't intend to do its condition
        Given an agent
        And an empty session
        Given a guideline to remind that we have a special sale if they book today when you recommends on flights options
        Given a guideline to suggest only ground based travel options when the customer asks about travel options
        And a customer message, "Hi, I want to go to California from New york next week. What are my options?"
        When processing is triggered
        Then a single message event is emitted
        And the message contains a suggestion to travel with bus or train but not with a flight
        And the message contains no sale option

    Scenario: The agent follows a regular guideline when it overrides an agent intention guideline
        Given an agent
        And an empty session
        Given a guideline to suggest direct flights or ground-based transportation when you recommends on travel options
        Given a guideline to suggest only ground-based travel options when the customer asks about domestic US travel options
        And a customer message, "Hi, I want to go to California from New york next week. What are my options?"
        When processing is triggered
        Then a single message event is emitted
        And the message contains a suggestion to travel with ground-based travel options but not with a flight

    Scenario: The agent follows an agent intention guideline when it overrides an agent intention guideline 2
        Given an agent
        And an empty session
        Given a guideline to recommend on our recommended toppings - either pineapple or pepperoni when you recommends on pizza toppings
        Given a guideline to recommend from our vegetarian recommended toppings when the customer asks about topping recommendation and the customer is from India
        And a customer message, "Hi, I want to buy pizza. What do you recommend? I'm vegetarian."
        When processing is triggered
        Then a single message event is emitted
        And the message contains a recommendation only on pineapple as topping
        And the message contains no recommendation on pepperoni pizza

