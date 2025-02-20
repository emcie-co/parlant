Feature: Tools
    Background:
        Given the alpha engine
        And an agent
        And an empty session

    Scenario: Single tool get_available_drinks is being called once
        Given the guideline called "check_drinks_in_stock"
        And the tool "get_available_drinks"
        And an association between "check_drinks_in_stock" and "get_available_drinks"
        And a customer message, "Hey, can I order a large pepperoni pizza with Sprite?"
        When processing is triggered
        Then a single tool calls event is emitted
        And the tool calls event contains 1 tool call(s)
        And the tool calls event contains Sprite and Coca Cola as available drinks

    Scenario: Single tool get_available_toppings is being called once
        Given the guideline called "check_toppings_in_stock"
        And the tool "get_available_toppings"
        And an association between "check_toppings_in_stock" and "get_available_toppings"
        And a customer message, "Hey, can I order a large pepperoni pizza with Sprite?"
        When processing is triggered
        Then a single tool calls event is emitted
        And the tool calls event contains 1 tool call(s)
        And the tool calls event contains Mushrooms and Olives as available toppings

    Scenario: Single tool is being called multiple times
        Given a guideline "sell_pizza" to sell pizza when interacting with customers
        And a guideline "check_stock" to check if toppings or drinks are available in stock when a client asks for toppings or drinks
        And the tool "get_available_product_by_type"
        And an association between "check_stock" and "get_available_product_by_type"
        And a customer message, "Hey, Can I order a large pizza with pepperoni and Sprite on the side?"
        When processing is triggered
        Then a single tool calls event is emitted
        And the tool calls event contains 2 tool call(s)
        And the tool calls event contains Sprite and Coca Cola as drinks, and Pepperoni, Mushrooms and Olives as toppings

    Scenario: Add tool called twice
        Given a guideline "calculate_sum" to calculate sums when the customer seeks to add numbers
        And the tool "add"
        And an association between "calculate_sum" and "add"
        And a customer message, "What is 8+2 and 4+6?"
        When processing is triggered
        Then a single tool calls event is emitted
        And the tool calls event contains 2 tool call(s)
        And the tool calls event contains the numbers 8 and 2 in the first tool call
        And the tool calls event contains the numbers 4 and 6 in the second tool call

    Scenario: Drinks and toppings tools called from same guideline
        Given a guideline "sell_pizza" to sell pizza when interacting with customers
        And a guideline "check_drinks_or_toppings_in_stock" to check for drinks or toppings in stock when the customer specifies toppings or drinks
        And the tool "get_available_drinks"
        And the tool "get_available_toppings"
        And an association between "check_drinks_or_toppings_in_stock" and "get_available_drinks"
        And an association between "check_drinks_or_toppings_in_stock" and "get_available_toppings"
        And a customer message, "Hey, can I order a large pepperoni pizza with Sprite?"
        When processing is triggered
        Then a single tool calls event is emitted
        And the tool calls event contains 2 tool call(s)
        And the tool calls event contains Sprite and Coca Cola under "get_available_drinks"
        And the tool calls event contains Pepperoni, Mushrooms, and Olives under "get_available_toppings"

    Scenario: Drinks and toppings tools called from different guidelines
        Given a guideline "sell_pizza" to sell pizza when interacting with customers
        And a guideline "check_drinks_in_stock" to check for drinks in stock when the customer specifies drinks
        And a guideline "check_toppings_in_stock" to check for toppings in stock when the customer specifies toppings
        And the tool "get_available_drinks"
        And the tool "get_available_toppings"
        And an association between "check_drinks_in_stock" and "get_available_drinks"
        And an association between "check_toppings_in_stock" and "get_available_toppings"
        And a customer message, "Hey, can I order a large pepperoni pizza with Sprite?"
        When processing is triggered
        Then a single tool calls event is emitted
        And the tool calls event contains 2 tool call(s)
        And the tool calls event contains Sprite and Coca Cola under "get_available_drinks"
        And the tool calls event contains Pepperoni, Mushrooms, and Olives under "get_available_toppings"

    Scenario: Add and multiply tools called once each
        Given a guideline "calculate_addition_or_multiplication" to calculate addition or multiplication when customers ask arithmetic questions
        And the tool "add"
        And the tool "multiply"
        And an association between "calculate_addition_or_multiplication" and "add"
        And an association between "calculate_addition_or_multiplication" and "multiply"
        And a customer message, "What is 8+2 and 4*6?"
        When processing is triggered
        Then a single tool calls event is emitted
        And the tool calls event contains 2 tool call(s)
        And the tool calls event contains the numbers 8 and 2 in the "add" tool call
        And the tool calls event contains the numbers 4 and 6 in the "multiply" tool call

    Scenario: Add and multiply tools called multiple times each
        Given a guideline "calculate_addition_or_multiplication" to calculate addition or multiplication when customers ask arithmetic questions
        And the tool "add"
        And the tool "multiply"
        And an association between "calculate_addition_or_multiplication" and "add"
        And an association between "calculate_addition_or_multiplication" and "multiply"
        And a customer message, "What is 8+2 and 4*6? also, 9+5 and 10+2 and 3*5"
        When processing is triggered
        Then a single tool calls event is emitted
        And the tool calls event contains 5 tool call(s)
        And the tool calls event contains 3 calls to "add", one with 8 and 2, the second with 9 and 5, and the last with 10 and 2
        And the tool calls event contains 2 calls to "multiply", one with 4 and 6, and the other with 3 and 5

    Scenario: Tool call takes context variables into consideration
        Given a guideline "retrieve_account_information" to retrieve account information when customers inquire about account-related information
        And the tool "get_account_balance"
        And an association between "retrieve_account_information" and "get_account_balance"
        And a context variable "customer_account_name" set to "Jerry Seinfeld"
        And a customer message, "What's my account balance?"
        When processing is triggered
        Then a single tool calls event is emitted
        And the tool calls event contains 1 tool call(s)
        And the tool calls event contains a call to "get_account_balance" with Jerry Seinfeld's current balance

    Scenario: Relevant guidelines are refreshed based on tool results
        Given a guideline "retrieve_account_information" to retrieve account information when customers inquire about account-related information
        And the tool "get_account_balance"
        And an association between "retrieve_account_information" and "get_account_balance"
        And a customer message, "What is the balance of Scooby Doo's account?"
        And a guideline "apologize_for_missing_data" to apologize for missing data when the account balance has the value of -555
        When processing is triggered
        Then a single message event is emitted
        And the message contains an apology for missing data

    Scenario: The tool call is correlated with the message with which it was generated
        Given a guideline "sell_pizza" to sell pizza when interacting with customers
        And a guideline "check_stock" to check if toppings or drinks are available in stock when a client asks for toppings or drinks
        And the tool "get_available_product_by_type"
        And an association between "check_stock" and "get_available_product_by_type"
        And a customer message, "Hey, Can I order large pepperoni pizza with Sprite?"
        When processing is triggered
        Then a single tool calls event is emitted
        And a single message event is emitted
        And the tool calls event is correlated with the message event

    Scenario: Relevant guidelines are not refreshed based on tool results if no second iteration of proposing a new guideline is made
        Given an agent with a maximum of 1 engine iterations
        And a guideline "retrieve_account_information" to retrieve account information when customers inquire about account-related information
        And the tool "get_account_balance"
        And an association between "retrieve_account_information" and "get_account_balance"
        And a customer message, "What is the balance of Scooby Doo's account?"
        And a guideline "apologize_for_missing_data" to apologize for missing data when the account balance has the value of -555
        When processing is triggered
        Then a single message event is emitted
        And the message contains that the balance of Scooby Doo is -$555

    Scenario: The agent distinguishes between tools from different services
        Given a guideline "system_check_scheduling" to schedule a system check if the error is critical when the customer complains about an error
        And a guideline "cs_meeting_scheduleing" to schedule a new customer success meeting when the customer gives feedback regarding their use of the system
        And the tool "schedule" from "first_service"
        And the tool "schedule" from "second_service"
        And an association between "system_check_scheduling" and "schedule" from "first_service"
        And an association between "cs_meeting_scheduleing" and "schedule" from "second_service"
        And a customer message, "I’m really happy about the system"
        When processing is triggered
        Then a single tool calls event is emitted
        And the tool calls event contains 1 tool call(s)
        And the tool calls event contains a call with tool_id of "second_service:schedule"

    Scenario: The agent correctly calls tools from an entailed guideline
        Given a guideline "suggest_toppings" to suggest pineapple when the customer asks for topping recommendations
        And a guideline "check_stock" to check if the product is available in stock, and only suggest it if it is when suggesting products
        And the tool "get_available_toppings"
        And an association between "check_stock" and "get_available_toppings"
        And a guideline connection whereby "suggest_toppings" entails "check_stock"
        And a customer message, "What pizza topping should I take?"
        When processing is triggered
        Then a single tool calls event is emitted
        And the tool calls event contains 1 tool call(s)
        And the tool calls event contains a call with tool_id of "local:get_available_toppings"
        And a single message event is emitted
        And the message contains a recommendation for toppings which do not include pineapple

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

    Scenario: The agent uses tools correctly when many are available
        Given a guideline "retrieve_account_information" to retrieve account information when customers inquire about account-related information
        And the tool "get_account_balance"
        And the tool "check_fruit_price"
        And the tool "get_available_toppings"
        And the tool "schedule" from "first_service"
        And the tool "schedule" from "second_service"
        And the tool "get_available_product_by_type"
        And the tool "multiply"
        And an association between "retrieve_account_information" and "get_account_balance"
        And an association between "retrieve_account_information" and "check_fruit_price"
        And an association between "retrieve_account_information" and "get_available_toppings"
        And an association between "retrieve_account_information" and "schedule" from "first_service"
        And an association between "retrieve_account_information" and "schedule" from "second_service"
        And an association between "retrieve_account_information" and "get_available_product_by_type"
        And an association between "retrieve_account_information" and "multiply"
        And a customer message, "Does Larry David have enough money in his account to buy a kilogram of apples?"
        When processing is triggered
        Then a single tool calls event is emitted
        And the tool calls event contains 2 tool call(s)
        And the tool calls event contains a call to "local:get_account_balance" with Larry David's current balance
        And the tool calls event contains a call to "local:check_fruit_price" with the price of apples

    Scenario: Tool call takes enum parameter into consideration
        Given a guideline "get_available_products_by_category" to get all products by a specific category when a customer asks for the availability of products from a certain category
        And the tool "available_products_by_category" from "ksp"
        And an association between "get_available_products_by_category" and "available_products_by_category" from "ksp"
        And a customer message, "What available keyboards do you have?"
        When processing is triggered
        Then a single tool calls event is emitted
        And the tool calls event contains 1 tool call(s)
        And the tool calls event contains a call to "available_products_by_category" with category "peripherals"

    Scenario: The agent chooses to consult the policy when the user asks about product returns
        Given a guideline "handle_policy_questions" to consult policy and answer when the user asks policy-related matters
        And the tool "consult_policy"
        And the tool "other_inquiries"
        And an association between "handle_policy_questions" and "consult_policy"
        And an association between "handle_policy_questions" and "other_inquiries"
        And a customer message, "I'd like to return a product please?"
        When processing is triggered
        Then a single tool calls event is emitted
        And the tool calls event contains 1 tool call(s)
        And the tool calls event contains a call to "local:consult_policy" regarding return policies
        And a single message event is emitted
        And the message contains that the return policy allows returns within 4 days and 4 hours from the time of purchase

    Scenario: Tool called again by context after customer response
        Given an empty session
        And a guideline "retrieve_account_information" to retrieve account information when customers inquire about account-related information
        And the tool "get_account_balance"
        And an association between "retrieve_account_information" and "get_account_balance"
        And a customer message, "What is the balance of Larry David's account?"
        And a tool event with data, { "tool_calls": [{ "tool_id": "local:get_account_balance", "arguments": { "account_name": "Larry David"}, "result": { "data": 451000000, "metadata": {} }}]}
        And an agent message, "Larry David currently has 451 million dollars."
        And a customer message, "And what about now?"
        When processing is triggered
        Then a single tool calls event is emitted
        And the tool calls event contains 1 tool call(s)
        And the tool calls event contains a call to "get_account_balance" with Larry David's current balance

    Scenario: Tool caller does not over-optimistically assume an argument's value
        Given a customer with the name "Vax"
        And an empty session with "Vax"
        And a context variable "Current Date" set to "January 17th, 2025" for "Vax"
        And a guideline "pay_cc_bill_guideline" to help a customer make the payment when they want to pay their credit card bill
        And the tool "pay_cc_bill"
        And an association between "pay_cc_bill_guideline" and "pay_cc_bill"
        And a customer message, "Let's please pay my credit card bill"
        When processing is triggered
        Then no tool calls event is emitted

    Scenario: Tool caller correctly infers an argument's value (1)
        Given a customer with the name "Vax"
        And an empty session with "Vax"
        And a context variable "Current Date" set to "January 17th, 2025" for "Vax"
        And a guideline "pay_cc_bill_guideline" to help a customer make the payment when they want to pay their credit card bill
        And the tool "pay_cc_bill"
        And an association between "pay_cc_bill_guideline" and "pay_cc_bill"
        And a customer message, "Let's please pay my credit card bill immediately"
        When processing is triggered
        Then a single tool calls event is emitted
        And the tool calls event contains 1 tool call(s)
        And the tool calls event contains a call to "pay_cc_bill" with date 17-01-2025

    Scenario: Tool caller correctly infers an argument's value (2)
        Given a customer with the name "Vax"
        And an empty session with "Vax"
        And a context variable "Current Date" set to "January 17th, 2025" for "Vax"
        And a guideline "pay_cc_bill_guideline" to help a customer make the payment when they want to pay their credit card bill
        And the tool "pay_cc_bill"
        And an association between "pay_cc_bill_guideline" and "pay_cc_bill"
        And a customer message, "Let's please pay my credit card bill. Payment date is tomorrow."
        When processing is triggered
        Then a single tool calls event is emitted
        And the tool calls event contains 1 tool call(s)
        And the tool calls event contains a call to "pay_cc_bill" with date 18-01-2025

    Scenario: Guideline proposer and tool caller understand that a Q&A tool needs to be called multiple times to answer different questions
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

    Scenario: Message generator understands and communicates that required information is missing
        Given an empty session
        And a guideline "pay_cc_bill_guideline" to help a customer make the payment when they want to pay their credit card bill
        And the tool "pay_cc_bill"
        And an association between "pay_cc_bill_guideline" and "pay_cc_bill"
        And a customer message, "Let's please pay my credit card bill."
        When processing is triggered
        Then no tool calls event is emitted
        And a single message event is emitted
        And the message mentions that a date is missing
