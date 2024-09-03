Feature: Single Tool Event
    Background:
        Given the alpha engine
        And an agent
        And an empty session

    Scenario Outline: Single tool is being called once
        Given a guideline "<GUIDELINE>", to sell pizza when interacting with users
        And the tool "<TOOL>"
        And an association between "<GUIDELINE>" and "<TOOL>"
        And a user message, "Hey, can I order a large pepperoni pizza with Sprite?"
        When processing is triggered
        Then a single tool calls event is produced
        And the tool calls event contains 1 tool call(s)
        And the tool calls event contains <EXPECTED_CONTENT>
        Examples:
            | GUIDELINE               | TOOL                   | EXPECTED_CONTENT                           |
            | check_drinks_in_stock   | get_available_drinks   | Sprite and Coca Cola as available drinks   |
            | check_toppings_in_stock | get_available_toppings | Mushrooms and Olives as available toppings |

    Scenario: Single tool is being called multiple times
        Given a guideline "sell_pizza", to sell pizza when interacting with users
        And a guideline "check_stock", to check if toppings or drinks are available in stock when a client asks for toppings or drinks
        And the tool "get_available_product_by_type"
        And an association between "check_stock" and "get_available_product_by_type"
        And a user message, "Hey, Can I order large pepperoni pizza with Sprite?"
        When processing is triggered
        Then a single tool calls event is produced
        And the tool calls event contains 2 tool call(s)
        And the tool calls event contains Sprite and Coca Cola as drinks, and Pepperoni, Mushrooms and Olives as toppings

    Scenario: Add tools called twice
        Given a guideline "calculate_sum", to calculate sums when the user seeks to add numbers
        And the tool "add"
        And an association between "calculate_sum" and "add"
        And a user message, "What is 8+2 and 4+6?"
        When processing is triggered
        Then a single tool calls event is produced
        And the tool calls event contains 2 tool call(s)
        And the tool calls event contains the numbers 8 and 2 in the first tool call
        And the tool calls event contains the numbers 4 and 6 in the second tool call

    Scenario: Drinks and toppings tools called from same guideline
        Given a guideline "sell_pizza", to sell pizza when interacting with users
        And a guideline "check_drinks_or_toppings_in_stock", to check for drinks or toppings in stock when the user specifies toppings or drinks
        And the tool "get_available_drinks"
        And the tool "get_available_toppings"
        And an association between "check_drinks_or_toppings_in_stock" and "get_available_drinks"
        And an association between "check_drinks_or_toppings_in_stock" and "get_available_toppings"
        And a user message, "Hey, can I order a large pepperoni pizza with Sprite?"
        When processing is triggered
        Then a single tool calls event is produced
        And the tool calls event contains 2 tool call(s)
        And the tool calls event contains Sprite and Coca Cola under "get_available_drinks"
        And the tool calls event contains Pepperoni, Mushrooms, and Olives under "get_available_toppings"

    Scenario: Drinks and toppings tools called from different guidelines
        Given a guideline "sell_pizza", to sell pizza when interacting with users
        And a guideline "check_drinks_in_stock", to check for drinks in stock when the user specifies drinks
        And a guideline "check_toppings_in_stock", to check for toppings in stock when the user specifies toppings
        And the tool "get_available_drinks"
        And the tool "get_available_toppings"
        And an association between "check_drinks_in_stock" and "get_available_drinks"
        And an association between "check_toppings_in_stock" and "get_available_toppings"
        And a user message, "Hey, can I order a large pepperoni pizza with Sprite?"
        When processing is triggered
        Then a single tool calls event is produced
        And the tool calls event contains 2 tool call(s)
        And the tool calls event contains Sprite and Coca Cola under "get_available_drinks"
        And the tool calls event contains Pepperoni, Mushrooms, and Olives under "get_available_toppings"

    Scenario: Add and multiply tools called once each
        Given a guideline "calculate_addition_or_multiplication", to calculate addition or multiplication when users ask arithmetic questions
        And the tool "add"
        And the tool "multiply"
        And an association between "calculate_addition_or_multiplication" and "add"
        And an association between "calculate_addition_or_multiplication" and "multiply"
        And a user message, "What is 8+2 and 4*6?"
        When processing is triggered
        Then a single tool calls event is produced
        And the tool calls event contains 2 tool call(s)
        And the tool calls event contains the numbers 8 and 2 in the "add" tool call
        And the tool calls event contains the numbers 4 and 6 in the "multiply" tool call

    Scenario: Add and multiply tools called multiple times each
        Given a guideline "calculate_addition_or_multiplication", to calculate addition or multiplication when users ask arithmetic questions
        And the tool "add"
        And the tool "multiply"
        And an association between "calculate_addition_or_multiplication" and "add"
        And an association between "calculate_addition_or_multiplication" and "multiply"
        And a user message, "What is 8+2 and 4*6? also, 9+5 and 10+2 and 3*5"
        When processing is triggered
        Then a single tool calls event is produced
        And the tool calls event contains 5 tool call(s)
        And the tool calls event contains 3 calls to "add", one with 8 and 2, the second with 9 and 5, and the last with 10 and 2
        And the tool calls event contains 2 calls to "multiply", one with 4 and 6, and the other with 3 and 5

    Scenario: Tool called again by context after user response
        Given a guideline "retrieve_account_information", to retrieve account information when users inquire about account-related information
        And the tool "get_account_balance"
        And an association between "retrieve_account_information" and "get_account_balance"
        And a user message, "What is the balance of Larry David's account?"
        And a tool event with data, [{ "tool_calls": { "tool_name": "get_account_balance", "parameters": { "account_name": "Larry David"}, "result": { "data": 451000000, "metadata": {} }}}]
        And a server message, "Larry David currently has 451 million dollars."
        And a user message, "And what about now?"
        When processing is triggered
        Then a single tool calls event is produced
        And the tool calls event contains 1 tool call(s)
        And the tool calls event contains a call to "get_account_balance" with Larry David's current balance

    Scenario: Tool call takes context variables into consideration
        Given a guideline "retrieve_account_information", to retrieve account information when users inquire about account-related information
        And the tool "get_account_balance"
        And an association between "retrieve_account_information" and "get_account_balance"
        And a context variable "user_account_name" with a value of "Jerry Seinfeld"
        And a user message, "What's my account balance?"
        When processing is triggered
        Then a single tool calls event is produced
        And the tool calls event contains 1 tool call(s)
        And the tool calls event contains a call to "get_account_balance" with Jerry Seinfeld's current balance

    Scenario: Relevant guidelines are refreshed based on tool results
        Given a guideline "retrieve_account_information", to retrieve account information when users inquire about account-related information
        And the tool "get_account_balance"
        And an association between "retrieve_account_information" and "get_account_balance"
        And a user message, "What is the balance of Scooby Doo's account?"
        And a guideline "apologize_for_missing_data", to apologize for missing data when the account balance has the value of -1
        When processing is triggered
        Then a single message event is produced
        And the message contains an apology for missing data

    Scenario: The tool call is correlated with the message with which it was generated
        Given a guideline "sell_pizza", to sell pizza when interacting with users
        And a guideline "check_stock", to check if toppings or drinks are available in stock when a client asks for toppings or drinks
        And the tool "get_available_product_by_type"
        And an association between "check_stock" and "get_available_product_by_type"
        And a user message, "Hey, Can I order large pepperoni pizza with Sprite?"
        When processing is triggered
        Then a single tool calls event is produced
        And a single message event is produced
        And the tool calls event is correlated with the message event
