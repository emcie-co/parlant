Feature: Tools
    Background:
        Given the engine
        And an agent
        And an empty session

    # Moved from the stable tier: the model intermittently refuses to use the
    # generic transfer_money tool for an unlisted currency ("I am unable to
    # transfer Dinars") when the currency-specific overlap set is only
    # partially declared. The direct-overlap variant remains stable.
    Scenario: Tool caller use the more suitable tool for transfer when three overlap indirectly
        Given a policy "do_transaction" to transfer money for the customer when customer asks to transfer money
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

    Scenario: The engine understands that a Q&A tool needs to be called multiple times to answer different questions
        Given a policy "answer_questions" to look up the answer and, if found, when the customer has a question related to the bank's services
        And the tool "find_answer"
        And an association between "answer_questions" and "find_answer"
        And a customer message, "How do I pay my credit card bill?"
        And an agent message, "You can just tell me the last 4 digits of the desired card and I'll help you with that."
        And a customer message, "Thank you! And I imagine this applies also if my card is currently lost, right?"
        And that the "answer_questions" policy was matched in the previous iteration
        When detection and processing are triggered
        Then a single tool calls event is emitted
        And the tool calls event contains 1 tool call(s)
        And the tool calls event contains a call to "find_answer" with an inquiry about a situation in which a card is lost

    Scenario: Relevant policies are refreshed based on tool results
        Given a policy "retrieve_account_information" to retrieve account information when customers inquire about account-related information
        And the tool "get_account_balance"
        And an association between "retrieve_account_information" and "get_account_balance"
        And a customer message, "What is the balance of Scooby Doo's account?"
        And a policy "apologize_for_missing_data" to apologize for missing data when the account balance has the value of -555
        When processing is triggered
        Then a single message event is emitted
        And the message contains an apology for missing data

    Scenario: No tool call emitted when data is ambiguous (transfer_coins)
        Given a policy "make_transfer" to make a transfer when asked to transfer money from one account to another
        And the tool "transfer_coins"
        And an association between "make_transfer" and "transfer_coins"
        And a customer message, "My name is Mark Corrigan and I want to transfer about 200-300 dollars from my account to Sophie Chapman account. My pincode is 1234"
        When processing is triggered
        Then no tool calls event is emitted

    Scenario: Tool caller correctly infers arguments values with optional (3)
        Given a policy "filter_electronic_products" to retrieve relevant products that match the asked attributes when customer is interested in electronic products with specific attributes
        And the tool "search_electronic_products"
        And an association between "filter_electronic_products" and "search_electronic_products"
        And a customer message, "Hey, how much does a SSD of Samsung cost?"
        When processing is triggered
        Then a single tool calls event is emitted
        And the tool calls event contains 1 tool call(s)
        And the tool calls event contains SSD as keyword and Samsung as Vendor

    Scenario: Tool caller chooses the right tool when two are activated 
        Given a customer named "Harry"
        And an empty session with "Harry"
        And a policy "to_schedule_meeting" to schedule a meeting when customer asks to schedule a meeting
        And a policy "to_schedule_appointment" to schedule an appointment with a doctor when user asks to make an appointment
        And a policy "to_send_email" to send an email to them when customer asks to reach out with someone
        And the tool "send_email"
        And an association between "to_send_email" and "send_email"
        And the tool "schedule_meeting"
        And an association between "to_schedule_meeting" and "schedule_meeting"
        And the tool "schedule_appointment"
        And an association between "to_schedule_appointment" and "schedule_appointment"
        And a tool relationship whereby "schedule_meeting" overlaps with "schedule_appointment"
        # And a tool relationship whereby "schedule_meeting" overlaps with "send_email"
        And a context variable "Current Date" set to "April 9th, 2025" for "Harry"
        # send_email is consequential, so the agent drafts and confirms before
        # sending - the scenario supplies the confirmation. Its subject stays
        # the choice of tool: email, not meeting/appointment scheduling.
        And a customer message, "Can you reach out to Morgan and see if she’s free to meet tomorrow at 10:30 about the hiring freeze?"
        And an agent message, "Would you like me to send an email to Morgan asking if she is available to meet tomorrow at 10:30 regarding the hiring freeze?"
        And a customer message, "Yes, please go ahead"
        When processing is triggered
        Then a single tool calls event is emitted
        And the tool calls event contains 1 tool call(s)
        And the tool calls event contains a call to "local:send_email" to morgan with subject of a meeting tomorrow and doesn't contains a call to "local:schedule_meeting"

    Scenario: Overlapped tool that has both missing and invalid parameters, some hidden and some have display names, communicate the problems correctly
        Given an empty session
        And a policy "calculate your salary" to calculate the salary of a person when the customer wants to know their salary
        And the tool "calculate_salary"
        And an association between "calculate your salary" and "calculate_salary"
        And the tool "calculate_expected_salary"
        And an association between "calculate your salary" and "calculate_expected_salary"
        And a tool relationship whereby "calculate_salary" overlaps with "calculate_expected_salary"
        And a customer message, "Hi, My name is Chris Pikrim, I work in Mike Andike's team. My mistress KittyKat and my friend Shuki asked me for my salary, so I would like you to calculate my salary based on those people I mentioned. Please provide me with all details regarding missing or invalid data, including what would be valid options for each choice, if you know that."
        When processing is triggered
        Then no tool calls event is emitted
        And a single message event is emitted
        # Adapted from alpha: alpha counted missing/invalid via tool_insights;
        # compass communicates problems conversationally. The distinctive
        # contract: display names are used, internal names stay hidden.
        And the message mentions that "The robot" is missing or needs to be specified
        And the message doesn't contains the word "cleaner"
        And the message doesn't contains any mention of a manager or a director
