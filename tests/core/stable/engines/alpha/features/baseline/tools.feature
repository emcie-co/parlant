Feature: Tools
    Background:
        Given the alpha engine
        And an agent
        And that the agent uses the canned_fluid message composition mode
        And an empty session

    Scenario: Single tool get_available_drinks is being called once
        Given the guideline called "check_drinks_in_stock"
        And the tool "get_available_drinks"
        And an association between "check_drinks_in_stock" and "get_available_drinks"
        And a customer message, "Hey, can I order a large pepperoni pizza with Sprite?"
        When processing is triggered
        Then a single tool calls event is emitted
        And no tool error has occurred
        And the tool calls event contains 1 tool call(s)
        And the tool calls event contains Sprite and Coca Cola as available drinks

    Scenario: Single tool get_available_toppings is being called once
        Given the guideline called "check_toppings_in_stock"
        And the tool "get_available_toppings"
        And an association between "check_toppings_in_stock" and "get_available_toppings"
        And a customer message, "Hey, can I order a large pepperoni pizza with Sprite?"
        When processing is triggered
        Then a single tool calls event is emitted
        And no tool error has occurred
        And the tool calls event contains 1 tool call(s)
        And the tool calls event contains Mushrooms and Olives as available toppings

    Scenario: Single tool is being called multiple times
        Given a guideline "sell_pizza" to sell pizza when interacting with customers
        And a guideline "check_stock" to check if toppings or drinks are available in stock when a client asks for toppings or drinks
        And the tool "get_available_product_by_type"
        And an association between "check_stock" and "get_available_product_by_type"
        And a customer message, "Hey, Can I order a large pizza with pepperoni and Sprite on the side?"
        When processing is triggered
        Then no tool error has occurred
        And the tool calls event contains 2 tool call(s)
        And the tool calls event contains Sprite and Coca Cola as drinks, and Pepperoni, Mushrooms and Olives as toppings

    Scenario: Add tool called twice
        Given a guideline "calculate_sum" to calculate sums when the customer seeks to add numbers
        And the tool "add"
        And an association between "calculate_sum" and "add"
        And a customer message, "What is 8+2 and 4+6?"
        When processing is triggered
        Then no tool error has occurred
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
        Then no tool error has occurred
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
        Then no tool error has occurred
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
        Then no tool error has occurred
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
        Then no tool error has occurred
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
        And no tool error has occurred
        And the tool calls event contains 1 tool call(s)
        And the tool calls event contains a call to "get_account_balance" with Jerry Seinfeld's current balance

    Scenario: The tool call is traced with the message with which it was generated
        Given a guideline "sell_pizza" to sell pizza when interacting with customers
        And a guideline "check_stock" to check if toppings or drinks are available in stock when a client asks for toppings or drinks
        And the tool "get_available_product_by_type"
        And an association between "check_stock" and "get_available_product_by_type"
        And a customer message, "Hey, can I order large pizza with a pepperoni topping and Sprite on the side?"
        When processing is triggered
        Then no tool error has occurred
        And the tool calls event contains 2 tool call(s)
        And a single message event is emitted
        And the tool calls event is traced with the message event

    Scenario: Relevant guidelines are not refreshed based on tool results if no second iteration of matching a new guideline is made
        Given an agent with a maximum of 1 engine iterations
        And that the agent uses the canned_fluid message composition mode
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
        And a guideline "cs_meeting_scheduling" to schedule a new customer success meeting when the customer gives feedback regarding their use of the system
        And the tool "schedule" from "first_service"
        And the tool "schedule" from "second_service"
        And an association between "system_check_scheduling" and "schedule" from "first_service"
        And an association between "cs_meeting_scheduling" and "schedule" from "second_service"
        And a customer message, "I'm really happy about the system"
        When processing is triggered
        Then a single tool calls event is emitted
        And no tool error has occurred
        And the tool calls event contains 1 tool call(s)
        And the tool calls event contains a call with tool_id of "second_service:schedule"

    Scenario: The agent correctly calls tools from an entailed guideline
        Given a guideline "suggest_toppings" to suggest pineapple when the customer asks for topping recommendations
        And a guideline "check_stock" to check if the product is available in stock, and only suggest it if it is when suggesting products
        And the tool "get_available_toppings"
        And an association between "check_stock" and "get_available_toppings"
        And a guideline relationship whereby "suggest_toppings" entails "check_stock"
        And a customer message, "What pizza topping should I take?"
        When processing is triggered
        Then a single tool calls event is emitted
        And the tool calls event contains 1 tool call(s)
        And the tool calls event contains a call with tool_id of "local:get_available_toppings"
        And a single message event is emitted
        And the message contains a recommendation for toppings which do not include pineapple

    Scenario: The agent uses tools correctly when many are available
        Given a guideline "retrieve_account_information" to retrieve account information when customers inquire about account-related information
        And the tool "get_account_balance"
        And the tool "check_fruit_price"
        And the tool "get_available_toppings"
        And the tool "schedule" from "first_service"
        And the tool "schedule" from "second_service"
        And the tool "get_available_product_by_type"
        And the tool "multiply"
        And a cross-service tool relationship whereby "schedule" from "first_service" overlaps with "schedule" from "second_service"
        And an association between "retrieve_account_information" and "get_account_balance"
        And an association between "retrieve_account_information" and "check_fruit_price"
        And an association between "retrieve_account_information" and "get_available_toppings"
        And an association between "retrieve_account_information" and "schedule" from "first_service"
        And an association between "retrieve_account_information" and "schedule" from "second_service"
        And an association between "retrieve_account_information" and "get_available_product_by_type"
        # And an association between "retrieve_account_information" and "multiply"
        And a customer message, "Does Larry David have enough money in his account to buy a kilogram of apples?"
        When processing is triggered
        Then no tool error has occurred
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
        And no tool error has occurred
        And the tool calls event contains 1 tool call(s)
        And the tool calls event contains a call to "available_products_by_category" with category "peripherals"

    Scenario: The agent chooses to consult the policy when the user asks about product returns
        Given a guideline "handle_policy_questions" to consult policy and answer when the user asks policy-related matters
        And the tool "consult_policy"
        And the tool "other_inquiries"
        And an association between "handle_policy_questions" and "consult_policy"
        And an association between "handle_policy_questions" and "other_inquiries"
        And a customer message, "I'd like to return a product please?"
        And a tool relationship whereby "consult_policy" overlaps with "other_inquiries"
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
        And that the "retrieve_account_information" guideline was matched in the previous iteration
        When detection and processing are triggered
        Then a single tool calls event is emitted
        And no tool error has occurred
        And the tool calls event contains 1 tool call(s)
        And the tool calls event contains a call to "get_account_balance" with Larry David's current balance

    Scenario: Tool caller does not over-optimistically assume an argument's value
        Given a customer named "Vax"
        And an empty session with "Vax"
        And a context variable "Current Date" set to "January 17th, 2025" for "Vax"
        And a guideline "pay_cc_bill_guideline" to help a customer make the payment when they want to pay their credit card bill
        And the tool "pay_cc_bill"
        And an association between "pay_cc_bill_guideline" and "pay_cc_bill"
        And a customer message, "Let's please pay my credit card bill"
        When processing is triggered
        Then no tool calls event is emitted

    Scenario: Tool caller correctly infers an argument's value (1)
        Given a customer named "Vax"
        And an empty session with "Vax"
        And a context variable "Current Date" set to "January 17th, 2025" for "Vax"
        And a guideline "pay_cc_bill_guideline" to help a customer make the payment when they want to pay their credit card bill
        And the tool "pay_cc_bill"
        And an association between "pay_cc_bill_guideline" and "pay_cc_bill"
        And a customer message, "Let's please pay my credit card bill immediately"
        When processing is triggered
        Then a single tool calls event is emitted
        And no tool error has occurred
        And the tool calls event contains 1 tool call(s)
        And the tool calls event contains a call to "pay_cc_bill" with date 17-01-2025

    Scenario: Tool caller correctly infers an argument's value (2)
        Given a customer named "Vax"
        And an empty session with "Vax"
        And a context variable "Current Date" set to "January 17th, 2025" for "Vax"
        And a guideline "pay_cc_bill_guideline" to help a customer make the payment when they want to pay their credit card bill
        And the tool "pay_cc_bill"
        And an association between "pay_cc_bill_guideline" and "pay_cc_bill"
        And a customer message, "Let's please pay my credit card bill. Payment date is tomorrow."
        When processing is triggered
        Then a single tool calls event is emitted
        And no tool error has occurred
        And the tool calls event contains 1 tool call(s)
        And the tool calls event contains a call to "pay_cc_bill" with date 18-01-2025

    Scenario: Message generator understands and communicates that required information is missing
        Given an empty session
        And a guideline "pay_cc_bill_guideline" to help a customer make the payment when they want to pay their credit card bill
        And the tool "pay_cc_bill"
        And an association between "pay_cc_bill_guideline" and "pay_cc_bill"
        And a customer message, "Let's please pay my credit card bill."
        When processing is triggered
        Then no tool calls event is emitted
        And no tool error has occurred
        And a single message event is emitted
        And the message mentions that a date is missing

    Scenario: When multiple parameters are missing, the message generator communicates only the ones with the lowest precedence value (1)
        Given an empty session
        And a guideline "registering_for_a_sweepstake" to register to a sweepstake when the customer wants to participate in a sweepstake
        And the tool "register_for_sweepstake"
        And an association between "registering_for_a_sweepstake" and "register_for_sweepstake"
        And a customer message, "Hi, my first name is Sushi, Please register me for a sweepstake with 3 entries. Ask me right away regarding every missing detail."
        When processing is triggered
        Then no tool calls event is emitted
        And a single message event is emitted
        And the number of missing parameters is exactly 1
        And the message mentions last name

    Scenario: When multiple parameters are missing, the message generator communicates only the ones with the lowest precedence value (2)
        Given an empty session
        And a guideline "registering_for_a_sweepstake" to register to a sweepstake when the customer wants to participate in a sweepstake
        And the tool "register_for_confusing_sweepstake"
        And an association between "registering_for_a_sweepstake" and "register_for_confusing_sweepstake"
        And a customer message, "Hi, I live in middle earth, Please register me for a sweepstake with 666 satan-type entries. Ask me right away regarding every missing detail."
        When processing is triggered
        Then no tool calls event is emitted
        And a single message event is emitted
        And the message mentions that parameters are missing
        And the number of missing parameters is exactly 2
        And the message mentions father and mother

    Scenario: Tool caller correctly infers arguments's value (1) (transfer_coins)
        Given an empty session
        And a guideline "make_transfer" to make a transfer when asked to transfer money from one account to another
        And the tool "transfer_coins"
        And an association between "make_transfer" and "transfer_coins"
        And a customer message, "I want to transfer $1500 from my account to Sophie Chapman"
        And an agent message, "I need your name and your pin code please"
        And a customer message, "My name is Mark Corrigan, The pincode is 1234"
        And that the "make_transfer" guideline was matched in the previous iteration
        When detection and processing are triggered
        Then a single tool calls event is emitted
        And no tool error has occurred
        And the tool calls event contains a call to "transfer_coins" with amount 1500 and from_account Mark Corrigan and to_account Sophie Chapman and pincode 1234

    Scenario: Tool caller correctly infers arguments's value (2) (transfer_coins)
        Given an empty session
        And a guideline "make_transfer" to make a transfer when asked to transfer money from one account to another
        And the tool "transfer_coins"
        And an association between "make_transfer" and "transfer_coins"
        And a customer message, "My name is Mark Corrigan and I want to transfer about 200-300 dollars from my account to Sophie Chapman account. My pincode is 1234. Actually I want to transfer 400"
        When processing is triggered
        Then a single tool calls event is emitted
        And no tool error has occurred
        And the tool calls event contains a call to "transfer_coins" with amount 400 and from_account Mark Corrigan and to_account Sophie Chapman and pincode 1234

    Scenario: Tool caller correctly infers arguments's value (3) (transfer_coins)
        Given an empty session
        And a guideline "make_transfer" to make a transfer when asked to transfer money from one account to another
        And the tool "transfer_coins"
        And an association between "make_transfer" and "transfer_coins"
        And a customer message, "I want to transfer $1500 from my account to Sophie Chapman"
        And an agent message, "I need your name and your pin code please"
        And a customer message, "My name is Mark Corrigan, The pincode is 1234"
        And an agent message, "Can you confirm the transformation?"
        And a customer message, "Actually I want to transfer 2000 please"
        And that the "make_transfer" guideline was matched in the previous iteration
        When detection and processing are triggered
        Then a single tool calls event is emitted
        And no tool error has occurred
        And the tool calls event contains a call to "transfer_coins" with amount 2000 and from_account Mark Corrigan and to_account Sophie Chapman and pincode 1234

    Scenario: Tool caller call the tool again when previous call has irrelevant arguments (1) (transfer_coins)
        Given an empty session
        And a guideline "make_transfer" to make a transfer when asked to transfer money from one account to another
        And the tool "transfer_coins"
        And an association between "make_transfer" and "transfer_coins"
        And a customer message, "Im Mark Corrigan. I want to transfer $3200 from my account to Sophie Chapman. The pincode is 1234"
        And a tool event with data, { "tool_calls": [{ "tool_id": "local:transfer_coins", "arguments": [{"amount": 3200, "from_account": "Mark Corrigan", "to_account":"Sophie Chapman", "pincode": "1234"}], "result": { "data": "Transaction successful: Transaction number: 83933", "metadata": {} }}]}
        And an agent message, "The transaction was successful. Can I help with anything else"
        And a customer message, "I want to transfer $1500 from my account to Sophie Chapman"
        And an agent message, "I need your name and your pin code please"
        And a customer message, "My name is Mark Corrigan, The pincode is 1234"
        And a previously applied guideline "make_transfer"
        And that the "make_transfer" guideline was matched in the previous iteration
        When detection and processing are triggered
        Then a single tool calls event is emitted
        And no tool error has occurred
        And the tool calls event contains a call to "transfer_coins" with amount 1500 and from_account Mark Corrigan and to_account Sophie Chapman and pincode 1234

    Scenario: Tool caller call the tool again when previous call has irrelevant arguments (2) (transfer_coins)
        Given an empty session
        And a guideline "make_transfer" to make a transfer when asked to transfer money from one account to another
        And the tool "transfer_coins"
        And an association between "make_transfer" and "transfer_coins"
        And a customer message, "I want to transfer $1500 from Mark Jackobs account to Gal Gadot. The pincode is 1234"
        And a tool event with data, { "tool_calls": [{ "tool_id": "local:transfer_coins", "arguments": [{ "amount": 1500, "from_account": "Mark Jackobs", "to_account":"Gal Gadot", "pincode": "1234"}], "result": { "data": "Transaction successful: Transaction number: 83933", "metadata": {} }}]}
        And an agent message, "The transaction was successful. Can I help with anything else"
        And a customer message, "I want to transfer $1500 from my account to Sophie Chapman"
        And an agent message, "I need your name and your pin code please"
        And a customer message, "My name is Mark Corrigan, The pincode is 1234"
        And a previously applied guideline "make_transfer"
        And that the "make_transfer" guideline was matched in the previous iteration
        When detection and processing are triggered
        Then a single tool calls event is emitted
        And no tool error has occurred
        And the tool calls event contains a call to "transfer_coins" with amount 1500 and from_account Mark Corrigan and to_account Sophie Chapman and pincode 1234

    Scenario: No tool call emitted when there is missing data (1) (transfer_coins)
        Given an empty session
        And a guideline "make_transfer" to make a transfer when asked to transfer money from one account to another
        And the tool "transfer_coins"
        And an association between "make_transfer" and "transfer_coins"
        And a customer message, "I want to transfer $1500 from my account to Sophie Chapman"
        When processing is triggered
        Then no tool calls event is emitted

    Scenario: No tool call emitted when there is missing data (2) (transfer_coins)
        Given an empty session
        And a guideline "make_transfer" to make a transfer when asked to transfer money from one account to another
        And the tool "transfer_coins"
        And an association between "make_transfer" and "transfer_coins"
        And a customer message, "My name is Mark Corrigan I want to transfer $1500 from my account to Sophie Chapman account"
        When processing is triggered
        Then no tool calls event is emitted

    Scenario: No tool call emitted when there is missing data (3) (transfer_coins)
        Given an empty session
        And a guideline "make_transfer" to make a transfer when asked to transfer money from one account to another
        And the tool "transfer_coins"
        And an association between "make_transfer" and "transfer_coins"
        And a customer message, "My name is Mark Corrigan and I want to transfer money from my account to Sophie Chapman account. My pincode is 1234"
        When processing is triggered
        Then no tool calls event is emitted

    Scenario: Tool caller call the same tool twice when needed (transfer_coins)
        Given an empty session
        And a guideline "make_transfer" to make a transfer when asked to transfer money from one account to another
        And the tool "transfer_coins"
        And an association between "make_transfer" and "transfer_coins"
        And a customer message, "I want to transfer $1500 from my account to Sophie Chapman and $1700 to Margaret Thatcher"
        And an agent message, "I need your name and your pin code please"
        And a customer message, "My name is Mark Corrigan, The pincode is 1234"
        And that the "make_transfer" guideline was matched in the previous iteration
        When detection and processing are triggered
        Then the tool calls event contains 2 tool call(s)
        And no tool error has occurred

    Scenario: Tool caller don't call the tool when user asks about request but don't want to make one (transfer_coins)
        Given an empty session
        And a guideline "make_transfer" to make a transfer when asked to transfer money from one account to another
        And the tool "transfer_coins"
        And an association between "make_transfer" and "transfer_coins"
        And a customer message, "Can I make a transfer from my account to a different one?"
        And an agent message, "Absolutely! I can help you with that. Just let me know the details, and I'll assist you in making the transfer."
        And a customer message, "My name is Mark Corrigan, and I might want to send 10,101 dollars to my sister, Ruthie."
        And an agent message, "Got it, Mark! What's your pin code, please?"
        And a customer message, "It's 1234. But actually, I'm not sure if I want to do it right now. I may do it tomorrow instead. I'll keep you posted"
        And that the "make_transfer" guideline was matched in the previous iteration
        When detection and processing are triggered
        When processing is triggered
        Then no tool calls event is emitted

    Scenario: Tool call consider a guideline about tool parameters (1) (transfer_coins)
        Given an empty session
        And a guideline "make_transfer" to make a transfer when asked to transfer money from one account to another
        And the tool "transfer_coins"
        And an association between "make_transfer" and "transfer_coins"
        And a guideline to multiply amount by 2 when asked to make a transfer in euros
        And a customer message, "Can I make a transfer from my account to a different one?"
        And an agent message, "Absolutely! I can help you with that. Just let me know the details, and I'll assist you in making the transfer."
        And a customer message, "My name is Mark Corrigan, and I want to send 1500 euros to my sister, Sophie Chapman."
        And an agent message, "Got it, Mark! What's your pin code, please?"
        And a customer message, "It's 1234. "
        And that the "make_transfer" guideline was matched in the previous iteration
        When detection and processing are triggered
        Then the tool calls event contains a call to "transfer_coins" with amount 3000 and from_account Mark Corrigan and to_account Sophie Chapman and pincode 1234
        And no tool error has occurred

    Scenario: Tool call consider a guideline about tool parameters (2) (transfer_coins)
        Given an empty session
        And a guideline "make_transfer" to make a transfer when asked to transfer money 
        And the tool "transfer_coins"
        And an association between "make_transfer" and "transfer_coins"
        And a guideline to set the destination account to Sophie Chapman when asked to transfer money 
        And a customer message, "Hi, it's Mark Corrigan here. Can I make a transfer of 4500$?. You probably need my pincode, its 1234 "
        When processing is triggered
        Then the tool calls event contains a call to "transfer_coins" with amount 4500 and from_account Mark Corrigan and to_account Sophie Chapman and pincode 1234
        And no tool error has occurred

    Scenario: The tool caller infers parameters based on outputs from another tool (1) (transfer_coins)
        Given an empty session
        And a guideline "make_transfer" to make a transfer when asked to transfer money from one account to another
        And the tool "transfer_coins"
        And an association between "make_transfer" and "transfer_coins"
        And a customer message, "Hi, here Mark Corrigan. Can you check my account balance and transfer it all to Sophie Chapman? My pin code is 1234"
        And a tool event with data, { "tool_calls": [{ "tool_id": "local:get_account_balance", "arguments": { "account_name": "Mark Corrigan"}, "result": { "data": 1000, "metadata": {} }}]}
        When processing is triggered
        Then a single tool calls event is emitted
        And no tool error has occurred
        And the tool calls event contains a call to "transfer_coins" with amount 1000 and from_account Mark Corrigan and to_account Sophie Chapman and pincode 1234

    Scenario: The tool caller infers parameters based on outputs from another tool (2) (transfer_coins)
        Given an empty session
        And a guideline "make_transfer" to make a transfer when asked to transfer money from one account to another
        And the tool "transfer_coins"
        And an association between "make_transfer" and "transfer_coins"
        And a customer message, "Hi, here Mark Corrigan. I don't remember my sister name but I want to transfer her 100,000$. Can you check her name and make the transfer?. My pin code is 1234"
        And a tool event with data, { "tool_calls": [{ "tool_id": "local:get_user_sister_name", "arguments": { "user_name": "Mark Corrigan"}, "result": { "data": "Sophie Chapman", "metadata": {} }}]}
        When processing is triggered
        Then a single tool calls event is emitted
        And no tool error has occurred
        And the tool calls event contains a call to "transfer_coins" with amount 100,000 and from_account Mark Corrigan and to_account Sophie Chapman and pincode 1234

    Scenario: Tool call infer parameters from different conversation parts (transfer_coins)
        Given an empty session
        And a guideline "make_transfer" to make a transfer when asked to transfer money from one account to another
        And the tool "transfer_coins"
        And an association between "make_transfer" and "transfer_coins"
        And a customer message, "Hi, can I make a transfer from my account to a different one?"
        And an agent message, "Absolutely! I'd be happy to help with that. Just let me know the details—like who you want to send money to and how much—and I'll assist you with the transfer."
        And a customer message, "My name is Mark Corrigan. I'm not sure if I have enough money in my account, though. Can you help me figure that out?"
        And an agent message, "Of course, Mark! Would you like me to check your account balance for you?"
        And a customer message, "Not right now. I think I'll just go ahead and try to make a transfer anyway."
        And an agent message, " Alright, no problem! Can you tell me the name of the person you want to send money to, and how much you'd like to transfer?"
        And a customer message, "Actually, do you work tomorrow? What are your working hours?"
        And an agent message, "Yes, I'm available every day from 9 AM to 5 PM. Would you like to go ahead with the transfer now, or is there something else you need?"
        And a customer message, "If I come tomorrow will that be ok?"
        And an agent message, "Yes, that works! Feel free to come tomorrow. Would you like to go ahead with the transfer now, or is there anything else you need help with?"
        And a customer message, "Yeah, I think I'm ready. My name is Mark Corrigan. I need to transfer to my dear friend Sophie Chapman "
        And an agent message, "Hi again, Mark! So, how much would you like to transfer today?"
        And a customer message, "I think it'd be better if someone else helped me with the transfer."
        And an agent message, "Alright, if you'd like me to help with that, I'll just need to know how much would you want to transfer"
        And a customer message, "I'm thinking of sending $2000 right now."
        And an agent message, "Alright, if you'd like me to assist with that, I'll just need your pin code to proceed"
        And a customer message, "Sure, try 1234."
        And that the "make_transfer" guideline was matched in the previous iteration
        When detection and processing are triggered
        Then the tool calls event contains a call to "transfer_coins" with amount 2000 and from_account Mark Corrigan and to_account Sophie Chapman and pincode 1234
        And no tool error has occurred

    Scenario: Tool caller call tool once when there are 2 requests but one with missing data (transfer_coins)
        Given an empty session
        And a guideline "make_transfer" to make a transfer when asked to transfer money from one account to another
        And the tool "transfer_coins"
        And an association between "make_transfer" and "transfer_coins"
        And a customer message, "Hi, here Mark Corrigan. Can I transfer $1000 to Mark Scout? Also make another transfer of $2000 but not to Mark Scout? My pin code is 1234"
        When processing is triggered
        Then a single tool calls event is emitted
        And no tool error has occurred
        And the tool calls event contains a call to "transfer_coins" with amount 1000 and from_account Mark Corrigan and to_account Mark Scout and pincode 1234

    Scenario: Tool caller correctly infers arguments values with optional (1)
        Given a guideline "filter_electronic_products" to retrieve relevant products that match the asked attributes when customer is interested in electronic products with specific attributes
        And the tool "search_electronic_products"
        And an association between "filter_electronic_products" and "search_electronic_products"
        And a customer message, "Hey, do you have laptop that is not above $300?"
        When processing is triggered
        Then no tool error has occurred

    Scenario: Tool caller correctly infers arguments values with optional (2)
        Given a guideline "filter_electronic_products" to retrieve relevant products that match the asked attributes when customer is interested in electronic products with specific attributes
        And the tool "search_electronic_products"
        And an association between "filter_electronic_products" and "search_electronic_products"
        And a customer message, "Hey, do you have SSD of Samsung?"
        And an agent message, "Do you have a price limit? for example not more than $400?"
        And a customer message, "No"
        And that the "filter_electronic_products" guideline was matched in the previous iteration
        When detection and processing are triggered
        Then a single tool calls event is emitted
        And no tool error has occurred
        And the tool calls event contains 1 tool call(s)
        And the tool calls event contains SSD as keyword and Samsung as Vendor and no price limit

    Scenario: Tool caller doesnt call tool when optional arguments exist but required not (1)
        Given a guideline "filter_electronic_products" to retrieve relevant products that match the asked attributes when customer is interested in electronic products with specific attributes
        And the tool "search_electronic_products"
        And an association between "filter_electronic_products" and "search_electronic_products"
        And a customer message, "Hey, do you have product above $15?"
        When processing is triggered
        Then no tool calls event is emitted

    Scenario: Tool caller doesnt call tool when optional arguments exist but required not (2)
        Given a guideline "filter_electronic_products" to retrieve relevant products that match the asked attributes when customer is interested in electronic products with specific attributes
        And the tool "search_electronic_products"
        And an association between "filter_electronic_products" and "search_electronic_products"
        And a customer message, "Hey, do you have in stock a product that cost more than $15?"
        When processing is triggered
        Then no tool calls event is emitted

    Scenario: Tool caller consider a guideline about optional parameters (1)
        Given a guideline "filter_electronic_products" to retrieve relevant products that match the asked attributes when customer is interested in electronic products with specific attributes
        And the tool "search_electronic_products"
        And an association between "filter_electronic_products" and "search_electronic_products"
        And a guideline to check only products in stock when costumer is interested in electronic products with specific attributes
        And a customer message, "Hey, do you have laptop that is not above $300?"
        When processing is triggered
        Then a single tool calls event is emitted
        And no tool error has occurred
        And the tool calls event contains 1 tool call(s)
        And the tool calls event contains laptop as keyword and 300 as max price and in_stock_only is True

    Scenario: Tool caller consider h parameters (2)
        Given a guideline "filter_electronic_products" to retrieve relevant products that match the asked attributes when customer is interested in electronic products with specific attributes
        And the tool "search_electronic_products"
        And an association between "filter_electronic_products" and "search_electronic_products"
        And a guideline to check only products of Dell when customer is interested in laptops with specific attributes
        And a customer message, "Hey, do you have a laptop that costs no more than $300 but isn't too cheap, say around $10?"
        When processing is triggered
        Then a single tool calls event is emitted
        And the tool calls event contains 1 tool call(s)
        And the tool calls event contains laptop as keyword and Dell as vendor and 300 as max price and 10 as min price

    Scenario: Tool caller call tool twice with optional arguments
        Given a guideline "filter_electronic_products" to retrieve relevant products that match the asked attributes when customer is interested in electronic products with specific attributes
        And the tool "search_electronic_products"
        And an association between "filter_electronic_products" and "search_electronic_products"
        And a customer message, "Hey, do you have Dell laptop or Samsung SSD?"
        When processing is triggered
        Then no tool error has occurred
        And the tool calls event contains 2 tool call(s)
        And the tool calls event contains a call to "local:search_electronic_products" with Dell brand and laptop keyword
        And the tool calls event contains a call to "local:search_electronic_products" with Samsung brand and SSD keyword

    Scenario: When tool have a mix of missing and invalid parameters, the message generator communicates the invalids with the missing, by precedence
        Given an empty session
        And a guideline "registering_for_a_sweepstake" to register to a sweepstake when the customer wants to participate in a sweepstake
        And the tool "register_for_sweepstake"
        And an association between "registering_for_a_sweepstake" and "register_for_sweepstake"
        And a customer message, "Hi, my first name is Nushi, Please register me for a sweepstake with 3 entries"
        When processing is triggered
        Then no tool calls event is emitted
        And a single message event is emitted
        And the message mentions that parameters are missing
        And the number of missing parameters is exactly 1
        And the message mentions that parameters are invalid
        And the number of invalid parameters is exactly 1
        And the message mentions mentions last name

    Scenario: A tool with both missing and invalid parameters, some hidden and some have display names, communicate the problems correctly
        Given an empty session
        And a guideline "calculate your salary" to calculate the salary of a person when the customer wants to know their salary
        And the tool "calculate_salary"
        And an association between "calculate your salary" and "calculate_salary"
        And a customer message, "Hi, My name is Chris Pikrim, I work in Mike Andike's team. My mistress KittyKat and my friend Shuki asked me for my salary, so I would like you to calculate my salary. Please provide me with all details regarding missing or invalid data."
        When processing is triggered
        Then no tool calls event is emitted
        And a single message event is emitted
        And the message mentions that the robot parameter is missing or needs to be specified
        And the number of missing parameters is exactly 1
        And the message mentions that parameters are invalid
        And the number of invalid parameters is exactly 2
        And the message mentions the robot, mistress and homie


    Scenario: Tool caller chooses the right tool for scheduling when three are overlapping
        Given a customer named "Hailey"
        And an empty session with "Hailey"
        And a guideline "to_schedule_appointment" to schedule an appointment with a doctor when user asks to make an appointment
        And a guideline "to_reschedule_appointment" to reschedule existing appointment when user had an appointment and they want to change its time
        And a guideline "to_schedule_meeting" to schedule a meeting when customer asks to meet with someone
        And the tool "schedule_appointment"
        And an association between "to_schedule_appointment" and "schedule_appointment"
        And the tool "reschedule_appointment"
        And an association between "to_reschedule_appointment" and "reschedule_appointment"
        And the tool "schedule_meeting"
        And an association between "to_schedule_meeting" and "schedule_meeting"
        And a tool relationship whereby "reschedule_appointment" overlaps with "schedule_appointment"
        And a tool relationship whereby "schedule_meeting" overlaps with "schedule_appointment"
        And a context variable "Current Date" set to "April 10th, 2025" for "Hailey"
        And a customer message, "Hi I want to make an appointment with Dr Sara Goodman tomorrow at 19:00. Can you help me with that?"
        When processing is triggered
        Then a single tool calls event is emitted
        And the tool calls event contains 1 tool call(s)
        And the tool calls event contains a call to "local:schedule_appointment" to Hailey with Dr sara goodman at 11-04-2025 19:00

    Scenario: Tool caller use both tools for scheduling appointment correctly when they overlap
        Given a customer named "Hailey"
        And an empty session with "Hailey"
        And a guideline "to_schedule_appointment" to schedule an appointment with a doctor when user asks to make an appointment
        And a guideline "to_reschedule_appointment" to reschedule existing appointment when user had an appointment and they want to change its time
        And the tool "schedule_appointment"
        And an association between "to_schedule_appointment" and "schedule_appointment"
        And the tool "reschedule_appointment"
        And an association between "to_reschedule_appointment" and "reschedule_appointment"
        And a tool relationship whereby "reschedule_appointment" overlaps with "schedule_appointment"
        And a context variable "Current Date" set to "April 10th, 2025" for "Hailey"
        And a customer message, "Hi, I'd like to schedule an appointment for tomorrow at 18:00 with Dr. Gabi, please. Also, I have an appointment with Dr. Michael. Could you please reschedule with Dr. Michael for tomorrow at 3:00 PM? Thank you!"
        When processing is triggered
        Then the tool calls event contains 2 tool call(s)
        And the tool calls event contains a call to "local:reschedule_appointment" to Hailey with Dr. Michael at 11-04-2025 15:00 and contains a call to "local:schedule_appointment" to Hailey with Dr. Gabi at 11-04-2025 18:00

    Scenario: Drinks and toppings tools called from same guideline
        Given a guideline "sell_pizza" to sell pizza when interacting with customers
        And a guideline "check_drinks_or_toppings_in_stock" to check for drinks or toppings in stock when the customer specifies toppings or drinks
        And the tool "get_available_drinks"
        And the tool "get_available_toppings"
        And an association between "check_drinks_or_toppings_in_stock" and "get_available_drinks"
        And an association between "check_drinks_or_toppings_in_stock" and "get_available_toppings"
        And a customer message, "Hey, can I order a large pepperoni pizza with Sprite?"
        When processing is triggered
        Then the tool calls event contains 2 tool call(s)
        And the tool calls event contains Sprite and Coca Cola under "get_available_drinks"
        And the tool calls event contains Pepperoni, Mushrooms, and Olives under "get_available_toppings"

    Scenario: Tool caller use the more suitable tool for transfer when two overlap
        Given a guideline "do_transaction" to transfer money for the customer when customer asks to transfer money
        And the tool "transfer_shekels"
        And the tool "transfer_money"
        And an association between "do_transaction" and "transfer_shekels"
        And an association between "do_transaction" and "transfer_money"
        And a customer message, "Hey, can transfer to my friend Alisse 200 shekels? my name is Fredric"
        And a tool relationship whereby "transfer_shekels" overlaps with "transfer_money"
        When processing is triggered
        Then a single tool calls event is emitted
        And the tool calls event contains 1 tool call(s)
        And the tool calls event contains a call to "transfer_shekels" with 200 as amount from Fredric to Alisse

    Scenario: Tool caller want to use the more suitable tool for transfer when two overlap and there are missing parameters (1)
        Given a guideline "do_transaction" to transfer money for the customer when customer asks to transfer money
        And the tool "transfer_shekels"
        And the tool "transfer_money"
        And an association between "do_transaction" and "transfer_shekels"
        And an association between "do_transaction" and "transfer_money"
        And a tool relationship whereby "transfer_shekels" overlaps with "transfer_money"
        And a customer message, "Hey, can transfer to my friend Alisse 200 shekels?"
        When processing is triggered
        Then no tool calls event is emitted

    Scenario: Tool caller want to use the more suitable tool for transfer when two overlap and there are missing parameters (2)
        Given a guideline "do_transaction" to transfer money for the customer when customer asks to transfer money
        And the tool "transfer_shekels"
        And the tool "transfer_money"
        And an association between "do_transaction" and "transfer_shekels"
        And an association between "do_transaction" and "transfer_money"
        And a tool relationship whereby "transfer_shekels" overlaps with "transfer_money"
        And a customer message, "Hey, can transfer to $200?"
        When processing is triggered
        Then no tool calls event is emitted

    Scenario: Tool caller use both tools for the right transfer when two overlap
        Given a guideline "do_transaction" to transfer money for the customer when customer asks to transfer money
        And a guideline to choose the more specific coin when customer asks to transfer money
        And the tool "transfer_money"
        And the tool "transfer_shekels"
        And an association between "do_transaction" and "transfer_shekels"
        And an association between "do_transaction" and "transfer_money"
        And a customer message, "Hey, can transfer to my friend Alisse 200 shekels and to my friend Bob $300? my name is Fredric"
        And a tool relationship whereby "transfer_shekels" overlaps with "transfer_money"
        When processing is triggered
        Then the tool calls event contains 2 tool call(s)
        And the tool calls event contains a call to "transfer_shekels" with 200 from Fredric to Alisse and a call to "transfer_money" with 300 from Fredric to Bob and no call to "transfer_money" with 200

    Scenario: Tool caller use tools multiple times for the right transfer when two overlap
        Given a guideline "do_transaction" to transfer money for the customer when customer asks to transfer money
        And a guideline to choose the more specific coin when customer asks to transfer money
        And the tool "transfer_shekels"
        And the tool "transfer_money"
        And an association between "do_transaction" and "transfer_shekels"
        And an association between "do_transaction" and "transfer_money"
        And a tool relationship whereby "transfer_shekels" overlaps with "transfer_money"
        And a customer message, "Hey, can transfer to my friend Alisse 200 shekels and to my friend Bob $300 and also 100 shekels to Bob? my name is Fredric"
        When processing is triggered
        Then the tool calls event contains 3 tool call(s)
        And the tool calls event contains a call to "transfer_shekels" with 200 from Fredric to Alisse a call to "transfer_shekels" with 100 from Fredric to Bob and a call to "transfer_money" with 300 from Fredric to Bob and no call to "transfer_money" with 200

    Scenario: Tool caller use the more suitable tool for transfer when three overlap directly
        Given a guideline "do_transaction" to transfer money for the customer when customer asks to transfer money
        And the tool "transfer_shekels"
        And the tool "transfer_money"
        And the tool "transfer_dollars"
        And an association between "do_transaction" and "transfer_shekels"
        And an association between "do_transaction" and "transfer_money"
        And an association between "do_transaction" and "transfer_dollars"
        And a tool relationship whereby "transfer_shekels" overlaps with "transfer_money"
        And a tool relationship whereby "transfer_dollars" overlaps with "transfer_money"
        And a tool relationship whereby "transfer_dollars" overlaps with "transfer_shekels"
        And a customer message, "Hey, can transfer to my friend Alisse 200 shekels and to my friend Dan $40 and to my friend Ali 500 Dinar? my name is Fredric"
        When processing is triggered
        Then the tool calls event contains 3 tool call(s)
        And the tool calls event contains a call to "transfer_shekels" with 200 from Fredric to Alisse and a call to "transfer_dollars" with 40 from Fredric to Dan and a call to "transfer_money" with 500 from Fredric to Ali

    Scenario: Tool caller use the more suitable tool for transfer when three overlap indirectly
        Given a guideline "do_transaction" to transfer money for the customer when customer asks to transfer money
        And the tool "transfer_shekels"
        And the tool "transfer_money"
        And the tool "transfer_dollars"
        And an association between "do_transaction" and "transfer_shekels"
        And an association between "do_transaction" and "transfer_money"
        And an association between "do_transaction" and "transfer_dollars"
        And a tool relationship whereby "transfer_shekels" overlaps with "transfer_money"
        And a tool relationship whereby "transfer_dollars" overlaps with "transfer_money"
        And a customer message, "Hey, can transfer to my friend Alisse 200 shekels and to my friend Dan $40 and to my friend Ali 500 Dinar? my name is Fredric"
        When processing is triggered
        Then the tool calls event contains 3 tool call(s)
        And the tool calls event contains a call to "transfer_shekels" with 200 from Fredric to Alisse and a call to "transfer_dollars" with 40 from Fredric to Dan and a call to "transfer_money" with 500 from Fredric to Ali

    Scenario: Tool caller user the more suitable tool for searching when two overlap (1)
        Given a guideline "do_search" to retrieve relevant products that match the asked attributes when customer is interested in products with specific attributes
        And the tool "search_products"
        And the tool "search_electronic_products"
        And an association between "do_search" and "search_products"
        And an association between "do_search" and "search_electronic_products"
        And a tool relationship whereby "search_electronic_products" overlaps with "search_products"
        And a customer message, "Hey, Do you have trousers that costs no more than $5 for men?"
        When processing is triggered
        Then a single tool calls event is emitted
        And the tool calls event contains 1 tool call(s)
        And the tool calls event contains a call to "search_products" with trousers as keyword and max price of 5

    Scenario: Tool caller user the more suitable tool for searching when two overlap (2)
        Given a guideline "do_search" to retrieve relevant products that match the asked attributes when customer is interested in products with specific attributes
        And the tool "search_products"
        And the tool "search_electronic_products"
        And an association between "do_search" and "search_products"
        And an association between "do_search" and "search_electronic_products"
        And a tool relationship whereby "search_electronic_products" overlaps with "search_products"
        And a customer message, "Hey, Do you have laptops that costs no more than $5?"
        When processing is triggered
        Then a single tool calls event is emitted
        And the tool calls event contains 1 tool call(s)
        And the tool calls event contains a call to "search_electronic_products" with laptop as keyword and max price of 5

    Scenario: The agent correctly chooses to call the right tool
        Given an agent whose job is to sell groceries
        And the term "carrot" defined as a kind of fruit
        And a guideline "check_prices" to reply with the price of the item when a customer asks about an items price
        And the tool "check_fruit_price"
        And the tool "check_vegetable_price"
        And an association between "check_prices" and "check_fruit_price"
        And an association between "check_prices" and "check_vegetable_price"
        And a tool relationship whereby "check_fruit_price" overlaps with "check_vegetable_price"
        And a customer message, "What's the price of 1 kg of carrots?"
        When processing is triggered
        Then a single tool calls event is emitted
        And the tool calls event contains 1 tool call(s)
        And the tool calls event contains a call with tool_id of "local:check_fruit_price"
        And a single message event is emitted
        And the message contains that the price of 1 kg of carrots is 10 dollars

    Scenario: Tool caller chooses the right tool for scheduling when three are overlapping
        Given a customer named "Hailey"
        And an empty session with "Hailey"
        And a guideline "to_schedule_appointment" to schedule an appointment with a doctor when user asks to make an appointment
        And a guideline "to_reschedule_appointment" to reschedule existing appointment when user had an appointment and they want to change its time
        And a guideline "to_schedule_meeting" to schedule a meeting when customer asks to meet with someone
        And the tool "schedule_appointment"
        And an association between "to_schedule_appointment" and "schedule_appointment"
        And the tool "reschedule_appointment"
        And an association between "to_reschedule_appointment" and "reschedule_appointment"
        And the tool "schedule_meeting"
        And an association between "to_schedule_meeting" and "schedule_meeting"
        And a tool relationship whereby "reschedule_appointment" overlaps with "schedule_appointment"
        And a tool relationship whereby "schedule_meeting" overlaps with "schedule_appointment"
        And a context variable "Current Date" set to "April 10th, 2025" for "Hailey"
        And a customer message, "Hi I want to make an appointment with Dr Sara Goodman tomorrow at 19:00. Can you help me with that?"
        When processing is triggered
        Then a single tool calls event is emitted
        And the tool calls event contains 1 tool call(s)
        And the tool calls event contains a call to "local:schedule_appointment" to Hailey with Dr sara goodman at 11-04-2025 19:00

    Scenario: Tool caller use both tools for scheduling appointment correctly when they overlap
        Given a customer named "Hailey"
        And an empty session with "Hailey"
        And a guideline "to_schedule_appointment" to schedule an appointment with a doctor when user asks to make an appointment
        And a guideline "to_reschedule_appointment" to reschedule existing appointment when user had an appointment and they want to change its time
        And the tool "schedule_appointment"
        And an association between "to_schedule_appointment" and "schedule_appointment"
        And the tool "reschedule_appointment"
        And an association between "to_reschedule_appointment" and "reschedule_appointment"
        And a tool relationship whereby "reschedule_appointment" overlaps with "schedule_appointment"
        And a context variable "Current Date" set to "April 10th, 2025" for "Hailey"
        And a customer message, "Hi, I'd like to schedule an appointment for tomorrow at 18:00 with Dr. Gabi, please. Also, I have an appointment with Dr. Michael. Could you please reschedule with Dr. Michael for tomorrow at 3:00 PM? Thank you!"
        When processing is triggered
        Then the tool calls event contains 2 tool call(s)
        And the tool calls event contains a call to "local:reschedule_appointment" to Hailey with Dr. Michael at 11-04-2025 15:00 and contains a call to "local:schedule_appointment" to Hailey with Dr. Gabi at 11-04-2025 18:00

    Scenario: Drinks and toppings tools called from same guideline
        Given a guideline "sell_pizza" to sell pizza when interacting with customers
        And a guideline "check_drinks_or_toppings_in_stock" to check for drinks or toppings in stock when the customer specifies toppings or drinks
        And the tool "get_available_drinks"
        And the tool "get_available_toppings"
        And an association between "check_drinks_or_toppings_in_stock" and "get_available_drinks"
        And an association between "check_drinks_or_toppings_in_stock" and "get_available_toppings"
        And a customer message, "Hey, can I order a large pepperoni pizza with Sprite?"
        When processing is triggered
        Then the tool calls event contains 2 tool call(s)
        And the tool calls event contains Sprite and Coca Cola under "get_available_drinks"
        And the tool calls event contains Pepperoni, Mushrooms, and Olives under "get_available_toppings"

    Scenario: Tool caller use the more suitable tool for transfer when two overlap
        Given a guideline "do_transaction" to transfer money for the customer when customer asks to transfer money
        And the tool "transfer_shekels"
        And the tool "transfer_money"
        And an association between "do_transaction" and "transfer_shekels"
        And an association between "do_transaction" and "transfer_money"
        And a customer message, "Hey, can I transfer to my friend Alisse 200 shekels? my name is Fredric"
        And a tool relationship whereby "transfer_shekels" overlaps with "transfer_money"
        When processing is triggered
        Then a single tool calls event is emitted
        And the tool calls event contains 1 tool call(s)
        And the tool calls event contains a call to "transfer_shekels" with 200 as amount from Fredric to Alisse

    Scenario: Tool caller chooses the more suitable tool for transfer when two overlap and there are missing parameters (1)
        Given a guideline "do_transaction" to transfer money for the customer when customer asks to transfer money
        And the tool "transfer_shekels"
        And the tool "transfer_money"
        And an association between "do_transaction" and "transfer_shekels"
        And an association between "do_transaction" and "transfer_money"
        And a tool relationship whereby "transfer_shekels" overlaps with "transfer_money"
        And a customer message, "Hey, can I transfer to my friend Alisse 200 shekels?"
        When processing is triggered
        Then no tool calls event is emitted

    Scenario: Tool caller chooses the more suitable tool for transfer when two overlap and there are missing parameters (2)
        Given a guideline "do_transaction" to transfer money for the customer when customer asks to transfer money
        And the tool "transfer_shekels"
        And the tool "transfer_money"
        And an association between "do_transaction" and "transfer_shekels"
        And an association between "do_transaction" and "transfer_money"
        And a tool relationship whereby "transfer_shekels" overlaps with "transfer_money"
        And a customer message, "Hey, can transfer $200?"
        When processing is triggered
        Then no tool calls event is emitted

    Scenario: Tool caller use both tools for the right transfer when two overlap
        Given a guideline "do_transaction" to transfer money for the customer when customer asks to transfer money
        And a guideline to choose the more specific coin when customer asks to transfer money
        And the tool "transfer_money"
        And the tool "transfer_shekels"
        And an association between "do_transaction" and "transfer_shekels"
        And an association between "do_transaction" and "transfer_money"
        And a customer message, "Hey, can transfer to my friend Alisse 200 shekels and to my friend Bob $300? my name is Fredric"
        And a tool relationship whereby "transfer_shekels" overlaps with "transfer_money"
        When processing is triggered
        Then the tool calls event contains 2 tool call(s)
        And the tool calls event contains a call to "transfer_shekels" with 200 from Fredric to Alisse and a call to "transfer_money" with 300 from Fredric to Bob and no call to "transfer_money" with 200

    Scenario: Tool caller use tools multiple times for the right transfer when two overlap
        Given a guideline "do_transaction" to transfer money for the customer when customer asks to transfer money
        And a guideline to choose the more specific coin when customer asks to transfer money
        And the tool "transfer_shekels"
        And the tool "transfer_money"
        And an association between "do_transaction" and "transfer_shekels"
        And an association between "do_transaction" and "transfer_money"
        And a tool relationship whereby "transfer_shekels" overlaps with "transfer_money"
        And a customer message, "Hey, can transfer to my friend Alisse 200 shekels and to my friend Bob $300 and also 100 shekels to Bob? my name is Fredric"
        When processing is triggered
        Then the tool calls event contains 3 tool call(s)
        And the tool calls event contains a call to "transfer_shekels" with 200 from Fredric to Alisse a call to "transfer_shekels" with 100 from Fredric to Bob and a call to "transfer_money" with 300 from Fredric to Bob and no call to "transfer_money" with 200

    Scenario: Tool caller user the more suitable tool for searching when two overlap (1)
        Given a guideline "do_search" to retrieve relevant products that match the asked attributes when customer is interested in products with specific attributes
        And the tool "search_products"
        And the tool "search_electronic_products"
        And an association between "do_search" and "search_products"
        And an association between "do_search" and "search_electronic_products"
        And a tool relationship whereby "search_electronic_products" overlaps with "search_products"
        And a customer message, "Hey, Do you have trousers that costs no more than $5 for men?"
        When processing is triggered
        Then a single tool calls event is emitted
        And the tool calls event contains 1 tool call(s)
        And the tool calls event contains a call to "search_products" with trousers as keyword and max price of 5

    Scenario: Tool caller user the more suitable tool for searching when two overlap (2)
        Given a guideline "do_search" to retrieve relevant products that match the asked attributes when customer is interested in products with specific attributes
        And the tool "search_products"
        And the tool "search_electronic_products"
        And an association between "do_search" and "search_products"
        And an association between "do_search" and "search_electronic_products"
        And a tool relationship whereby "search_electronic_products" overlaps with "search_products"
        And a customer message, "Hey, Do you have laptops that costs no more than $5?"
        When processing is triggered
        Then a single tool calls event is emitted
        And the tool calls event contains 1 tool call(s)
        And the tool calls event contains a call to "search_electronic_products" with laptop as keyword and max price of 5

    Scenario: The agent correctly chooses to call the right overlapping tool based on glossary
        Given an agent whose job is to sell groceries
        And that the agent uses the canned_fluid message composition mode
        And the term "carrot" defined as a kind of fruit
        And a guideline "check_prices" to reply with the price of the item when a customer asks about an items price
        And the tool "check_fruit_price"
        And the tool "check_vegetable_price"
        And an association between "check_prices" and "check_fruit_price"
        And an association between "check_prices" and "check_vegetable_price"
        And a tool relationship whereby "check_fruit_price" overlaps with "check_vegetable_price"
        And a customer message, "What's the price of 1 kg of carrots?"
        When processing is triggered
        Then a single tool calls event is emitted
        And the tool calls event contains 1 tool call(s)
        And the tool calls event contains a call with tool_id of "local:check_fruit_price"
        And a single message event is emitted
        And the message contains that the price of 1 kg of carrots is 10 dollars

    Scenario: Tool caller calls a tool with enum list parameter
        Given a guideline "get_available_products_by_category" to get all products by a specific category when a customer asks for the availability of products from a certain category
        And the tool "available_products_by_categories" from "ksp"
        And an association between "get_available_products_by_category" and "available_products_by_categories" from "ksp"
        And a customer message, "What available keyboards and laptops do you have?"
        When processing is triggered
        Then a single tool calls event is emitted
        And no tool error has occurred
        And the tool calls event contains 1 tool call(s)
        And a single message event is emitted
        And the message mentions peripherals and laptops

    Scenario: Tool caller calls a tool with different types of parameters with no type errors (1)
        Given a guideline "set_a_meating" to set a bbq-integrated meeting with some friends when a customer wants to set a meeting with friends
        And the tool "set_a_bbq_appointment"
        And an association between "set_a_meating" and "set_a_bbq_appointment"
        And a customer message, "Hi, please set a bbq appointment in the kitchen with my friends Johnny, Jack and Glen for 2 hours on 2025-01-17 at 12:00. Describe it as 'meating' and buy 2.3kg of meat for the appointment."
        When processing is triggered
        Then a single tool calls event is emitted
        And no tool error has occurred
        And the tool calls event contains 1 tool call(s)

    # Fails when the tool is not consequential because of bad conversion from float - should be fixable
    Scenario: Tool caller calls a tool with different types of parameters with no type errors (2)
        Given a guideline "set_a_meating" to set a bbq-integrated meeting with some friends when a customer wants to set a meeting with friends
        And the tool "set_a_bbq_appointment"
        And an association between "set_a_meating" and "set_a_bbq_appointment"
        And a customer message, "Please set a bbq appointment with my friends Johnny & Jack on 2025-01-17 at 12:00, description: 'meating' , and buy 1.9kg of meat for the appointment. Note that Jack is vegetarian and the ratings are 4.5 and 8.2. Alternative location is the kitchen."
        When processing is triggered
        Then a single tool calls event is emitted
        And no tool error has occurred
        And the tool calls event contains 1 tool call(s)

    Scenario: Tool caller calls a tool with different types of parameters with no type errors (3)
        Given a guideline "find_a_meating" to find a bbq-integrated meeting when a customer wants to find a meating to attend
        And the tool "find_bbq_appointments"
        And an association between "find_a_meating" and "find_bbq_appointments"
        And a customer message, "Please find me a bbq appointment with Johnny & Jack on May 4th in the kitchen"
        When processing is triggered
        Then a single tool calls event is emitted
        And no tool error has occurred
        And the tool calls event contains 1 tool call(s)

    Scenario: Tool caller calls a tool with list of booleans and optional boolean
        Given a guideline "give_boolean" to get a list of booleans from user when a customer wants to give some booleans
        And the tool "give_boolean_types"
        And an association between "give_boolean" and "give_boolean_types"
        And a customer message, "I want to give you the list of booleans  :true, false, true and false. Also, I want to give you the boolean true."
        When processing is triggered
        Then a single tool calls event is emitted
        And no tool error has occurred
        And the tool calls event contains 1 tool call(s)

    Scenario: Tool returns a result with transient lifespan and its event is not emitted
        Given a guideline "current_time" to get the current time when a customer wants to know the time
        And the tool "check_current_time"
        And an association between "current_time" and "check_current_time"
        And a customer message, "Hey, I have a meeting in 10:00, but I lost my watch. Am I late for the meeting ?"
        When processing is triggered
        Then the message contains the text "you are late"
        And a single event is staged
        And no tool calls event is emitted
        And the staged tool calls event contains "18:03"

    Scenario: Tool returns a result with explicit long-term lifespan and its event is emitted
        Given a guideline "current_time" to get the current time when a customer wants to know the time
        And the tool "check_current_time_emit"
        And an association between "current_time" and "check_current_time_emit"
        And a customer message, "Hey, I have a meeting in 10:00, but I lost my watch. Am I late for the meeting ?"
        When processing is triggered
        Then a single tool calls event is emitted
        And the message contains the text "9:59"

    Scenario: Guidelines with reevaluation relationship to a tool are activated by the tool result
        Given a guideline "offer_luxury_room" to offer the luxury room when a luxury room is available
        And a guideline "offer_the_dungeon" to offer the dungeon when a luxury room is not available
        And a guideline "check_rooms_availability" to check for availability when the customer want to book a room
        And the tool "availability_check"
        And an association between "check_rooms_availability" and "availability_check"
        And a reevaluation relationship between the guideline "offer_luxury_room" and the "availability_check" tool
        And a reevaluation relationship between the guideline "offer_the_dungeon" and the "availability_check" tool
        And a customer message, "want to book a room."
        When processing is triggered
        Then a single tool calls event is emitted
        And the message contains an offer for the dungeon
        And the message doesn't contains an offer for the luxury room

    Scenario: Non consequential tool using datetime formatted arguments is used correctly
        Given a guideline "set_appointment" to set an appointment at the agreed time when the customer chooses an appointment time between the available options
        And the tool "schedule_appointment_2"
        And an association between "set_appointment" and "schedule_appointment_2"
        And a context variable "current_date" set to "September 12th 2025"
        And a customer message, "I'm sick, I need to see a doctor ASAP"
        And an agent message, "I'm sorry to hear that."
        And an agent message, "If it's an emergency, please call a human representitive at our number. We have appointment slots for tomorrow at 2 PM, or on the 15.10 at 11 AM"
        And a customer message, "tomorrow at 2 PM is good"
        When processing is triggered
        Then a single tool calls event is emitted
        And the staged tool calls event contains an appointment was set to 2025-09-13
        And the message contains that an appointment was successfuly set for either tomorrow or the 2025-09-13