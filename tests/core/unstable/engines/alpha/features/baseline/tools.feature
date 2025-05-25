Feature: Tools
    Background:
        Given the alpha engine
        And an agent
        And an empty session


    Scenario: Guideline matcher and tool caller understand that a Q&A tool needs to be called multiple times to answer different questions
        Given a guideline "answer_questions" to look up the answer and, if found, when the customer has a question related to the bank's services
        And the tool "find_answer"
        And an association between "answer_questions" and "find_answer"
        And a customer message, "How do I pay my credit card bill?"
        And an agent message, "You can just tell me the last 4 digits of the desired card and I'll help you with that."
        And a customer message, "Thank you! And I imagine this applies also if my card is currently lost, right?"
        And that the "answer_questions" guideline was matched in the previous iteration
        When detection and processing are triggered
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
    
    Scenario: No tool call emitted when data is ambiguous (transfer_coins)
        Given a guideline "make_transfer" to make a transfer when asked to transfer money from one account to another
        And the tool "transfer_coins"
        And an association between "make_transfer" and "transfer_coins"
        And a customer message, "My name is Mark Corrigan and I want to transfer about 200-300 dollars from my account to Sophie Chapman account. My pincode is 1234"
        When processing is triggered
        Then no tool calls event is emitted

    Scenario: Tool caller correctly infers arguments values with optional (3)
        Given a guideline "filter_electronic_products" to retrieve relevant products that match the asked attributes when customer is interested in electronic products with specific attributes
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
        And a guideline "to_schedule_meeting" to schedule a meeting when customer asks to schedule a meeting
        And a guideline "to_schedule_appointment" to schedule an appointment with a doctor when user asks to make an appointment
        And a guideline "to_send_email" to send an email to them when customer asks to reach out with someone
        And the tool "send_email"
        And an association between "to_send_email" and "send_email"
        And the tool "schedule_meeting"
        And an association between "to_schedule_meeting" and "schedule_meeting"
        And the tool "schedule_appointment"
        And an association between "to_schedule_appointment" and "schedule_appointment"
        And a tool relationship whereby "schedule_meeting" overlaps with "schedule_appointment"
        # And a tool relationship whereby "schedule_meeting" overlaps with "send_email"
        And a context variable "Current Date" set to "April 9th, 2025" for "Harry"
        And a customer message, "Can you reach out to Morgan and see if she’s free to meet tomorrow at 10:30 about the hiring freeze?"
        When processing is triggered
        Then a single tool calls event is emitted
        And the tool calls event contains 1 tool call(s)
        And the tool calls event contains a call to "local:send_email" to morgan with subject of a meeting tomorrow and doesn't contains a call to "local:schedule_meeting"
