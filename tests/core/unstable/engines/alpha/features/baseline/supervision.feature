Feature: Supervision
    Background:
        Given an empty session

    Scenario: the agent considers guidelines and tools when many restrictions apply
        Given the alpha engine
        And an agent whose job is to only sell products that start with the letter t.
        And a guideline "best_soup", to respond with a vegetable soup of your choice when asked what our best dish is
        And a guideline "initiate_conversation", to greet the customer when its your first response
        And a guideline "table_price", to state that a table costs 100$ when the customer asks for the price of tables
        And a guideline "check_soups", to check which soups are in stock when asked anything about soup
        And a guideline "frustrated_user", to end your response with the word sorry when the user expresses frustration 
        And a guideline "open_with_hello", to begin your response with the word hello when discussing vegetable soups
        And a guideline connection whereby "best_soup" entails "open_with_hello"
        And the tool "get_available_soups"
        And an association between "check_soups" and "get_available_soups"
        And the term "Turpolance" defined as a mix of carrots and sweet potatoes
        And a context variable "customer allergies" set to "tomatoes"
        And a customer message, "Hi there, what is the best dish I could get?"
        When processing is triggered
        Then a single message event is emitted
        And the message contains that its first word is hello, and a recommendation for turpolance soup, also known as carrots and sweet potato soup
