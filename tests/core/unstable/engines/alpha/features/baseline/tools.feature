Feature: Tools
    Background:
        Given the alpha engine
        And an agent

    Scenario: Guideline matcher and tool caller understand that a Q&A tool needs to be called multiple times to answer different questions
        Given an empty session
        And a guideline "answer_questions" to look up the answer and, if found, when the customer has a question related to the bank's services
        And the tool "find_answer"
        And an association between "answer_questions" and "find_answer"
        And a customer message, "How do I pay my credit card bill?"
        And an agent message, "You can just tell me the last 4 digits of the desired card and I'll help you with that."
        And a customer message, "Thank you! And I imagine this applies also if my card is currently lost, right?"
        When processing is triggered
        Then a single tool calls event is emitted
        And the tool calls event contains 1 tool call(s)
        And the tool calls event contains a call to "find_answer" with an inquiry about a situation in which a card is lost

    Scenario: Relevant guidelines are refreshed based on tool results
        Given a guideline "retrieve_account_information" to retrieve account information when customers inquire about account-related information
        And the tool "get_account_balance"
        And an association between "retrieve_account_information" and "get_account_balance"
        And a customer message, "What is the balance of Scooby Doo's account?"
        And a guideline "apologize_for_missing_data" to apologize for missing data when the account balance has the value of -555
        When processing is triggered
        Then a single message event is emitted
        And the message contains an apology for missing data
    
    Scenario: No tool call emitted when data is ambiguios (transfer_coins)
        Given an empty session
        And a guideline "make_transfer" to make a transfer when asked to transfer money from one account to another
        And the tool "transfer_coins"
        And an association between "make_transfer" and "transfer_coins"
        And a customer message, "My name is Mark Corrigan and I want to transfer about 200-300 dollars from my account to Sophie Chapman account. My pincode is 1234"
        When processing is triggered
        Then no tool calls event is emitted

    Scenario: The agent correctly chooses to call the right tool
        Given an agent whose job is to sell groceries
        And the term "carrot" defined as a kind of fruit
        And a guideline "check_prices" to reply with the price of the item when a customer asks about an items price
        And the tool "check_fruit_price"
        And the tool "check_vegetable_price"
        And an association between "check_prices" and "check_fruit_price"
        And an association between "check_prices" and "check_vegetable_price"
        And a customer message, "What's the price of 1 kg of carrots?"
        When processing is triggered
        Then a single tool calls event is emitted
        And the tool calls event contains 1 tool call(s)
        And the tool calls event contains a call with tool_id of "local:check_fruit_price"
        And a single message event is emitted
        And the message contains that the price of 1 kg of carrots is 10 dollars