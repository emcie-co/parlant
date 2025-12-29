Feature: Conversation
    Background:
        Given the alpha engine
        And an agent
        And that the agent uses the canned_fluid message composition mode
        And an empty session

    Scenario: The agent follows a regular guideline when it overrides an agent intention guideline 2
        Given a guideline to recommend on our recommended toppings - either pineapple or pepperoni when you recommend pizza toppings
        And a guideline to recommend from our vegetarian recommended toppings when the customer asks about topping recommendation and the customer is from India
        And a customer message, "Hi, I want to buy pizza. What do you recommend? I'm from India if it matters."
        When processing is triggered
        Then a single message event is emitted
        And the message contains a recommendation only on pineapple as topping
        And the message contains no recommendation on pepperoni pizza

    Scenario: The agent follows an agent intention guideline when it overrides an agent intention guideline
        Given a guideline to suggest direct flights or ground-based transportation when you recommend travel options
        And a guideline to suggest only ground-based travel options when you recommend domestic US travel options
        And a customer message, "Hi, I want to go to California from New york next week. What are my options?"
        When processing is triggered
        Then a single message event is emitted
        And the message contains a suggestion to travel with ground-based travel options but not with a flight
