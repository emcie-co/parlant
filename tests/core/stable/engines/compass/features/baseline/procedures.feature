Feature: Procedures
    Background:
        Given the engine
        And an agent
        And an empty session

    Scenario: Multistep procedure is partially followed 1
        Given the procedure called "Reset Password Journey"
        And a customer message, "I want to reset my password"
        When processing is triggered
        Then no tool calls event is emitted
        And a single message event is emitted
        And the message contains asking the customer for their username, but not for their email or phone number

    Scenario: Irrelevant procedure is ignored
        Given the procedure called "Reset Password Journey"
        And a customer message, "What are some tips I could use to come up with a strong password?"
        When processing is triggered
        Then no tool calls event is emitted
        And a single message event is emitted
        # Adapted from alpha: the agent may mention that password reset is
        # available as a fallback offer; what it must not do is START the
        # procedure.
        And the message doesn't contains a request for the customer's username or any other step of the password reset process

    Scenario: Multistep procedure is partially followed 2
        Given the procedure called "Reset Password Journey"
        And a customer message, "I want to reset my password"
        And an agent message, "I can help you do just that. What's your username?"
        And a customer message, "it's leonardo_barbosa_1982"
        When processing is triggered
        Then no tool calls event is emitted
        And a single message event is emitted
        And the message contains asking the customer for their mobile number or email address
        And the message contains nothing about wishing the customer a good day

    Scenario: Multistep procedure is aborted when the procedure requires so
        Given the procedure called "Reset Password Journey"
        And a customer message, "I want to reset my password"
        And an agent message, "I can help you do just that. What's your username?"
        And a customer message, "it's leonardo_barbosa_1982"
        And an agent message, "Great! And what's the account's associated email address or phone number?"
        And a customer message, "the email is leonardobarbosa@gmail.br"
        And an agent message, "Got it. Before proceeding to reset your password, I wanted to wish you a good day"
        And a customer message, "What that does have to do with anything?"
        When processing is triggered
        Then no tool calls event is emitted
        And a single message event is emitted
        And the message contains an answer indicating that the password cannot be reset at this time, or has otherwise failed to reset

    Scenario: Policy and procedure are used in unison
        Given the procedure called "Book Flight"
        And a policy "skip steps" to skip steps that are inapplicable due to other contextual reasons when applying a book flight procedure
        And a dependency relationship between the policy "skip steps" and the "Book Flight" procedure
        And a policy "Business Adult Only" to know that travelers under the age of 21 are illegible for business class, and may only use economy when a flight is being booked
        And a customer message, "Hi, I'd like to book a flight for myself. I'm 19 if that effects anything."
        And an agent message, "Great! From and to where would are you looking to fly?"
        And a customer message, "From LAX to JFK"
        And an agent message, "Got it. And when are you looking to travel?"
        And a customer message, "Next Monday until Friday"
        When processing is triggered
        Then a single message event is emitted
        And the message contains either asking for the name of the person traveling, or informing them that they are only eligible for economy class

    Scenario: Procedure returns to earlier step when the conversation justifies doing so (1)
        Given the procedure called "Book Taxi Ride"
        And a customer message, "Hi, I'd like to book a taxi for myself"
        And an agent message, "Great! What's your pickup location?"
        And a customer message, "Main street 1234"
        And an agent message, "Got it. What's your drop-off location?"
        And a customer message, "3rd Avenue by the river"
        And an agent message, "Got it. What time would you like to pick up?"
        And a customer message, "Oh hold up, my plans have changed. I'm actually going to need a cab for my son, he'll be waiting at JFK airport, at the taxi stand."
        When processing is triggered
        Then a single message event is emitted
        And the message contains asking the customer for the drop-off location

    Scenario: Procedure returns to earlier step when the conversation justifies doing so (2)
        Given the procedure called "Place Food Order"
        And a customer message, "Hey, I'd like to make an order"
        And an agent message, "Great! What would you like to order? We have either a salad or a sandwich."
        And a customer message, "I'd like a sandwich"
        And an agent message, "Got it. What kind of bread would you like?"
        And a customer message, "I'd like a baguette"
        And an agent message, "Got it. What main filling would you like? We have either peanut butter, jam or pesto."
        And a customer message, "If that's your only options, can I get a salad instead?"
        When processing is triggered
        Then a single message event is emitted
        And the message contains asking asking what green base the customer wants for their salad

    Scenario: Dependent policies on a procedure are matched when the procedure is active
        Given the procedure called "Book Flight"
        And a policy "under 21" to inform the customer that only economy class is available when a customer wants to book a flight and the traveler is under 21
        And a policy "21 or older" to tell te customer they may choose between economy and business class when a customer wants to book a flight and the traveler is 21 or older
        And a dependency relationship between the policy "under 21" and the "Book Flight" procedure
        And a dependency relationship between the policy "21 or older" and the "Book Flight" procedure
        And a customer message, "Hi, my name is John Smith and I'd like to book a flight for myself from Ben Gurion airport to JFK. We flight in the 12.10 and return in the 17.10. I'm 19 if that affects anything."
        When processing is triggered
        Then a single message event is emitted
        And the message contains informing the customer that only economy class is available

    Scenario: Multistep procedure invokes tool calls correctly
        Given the procedure called "Reset Password Journey"
        And a customer message, "I want to reset my password"
        And an agent message, "I can help you do just that. What's your username?"
        And a customer message, "it's leonardo_barbosa_1982"
        And an agent message, "Great! And what's the account's associated email address or phone number?"
        And a customer message, "the email is leonardobarbosa@gmail.br"
        And an agent message, "Got it. Before proceeding to reset your password, I wanted to wish you a good day"
        And a customer message, "Thank you! Have a great day as well!"
        When processing is triggered
        Then a single tool calls event is emitted
        And the tool calls event contains 1 tool call(s)
        And the tool calls event contains the tool reset password with username leonardo_barbosa_1982 and email leonardobarbosa@gmail.br
        And a single message event is emitted
        And the message contains that the password was reset and an email with instructions was sent to the customer

    Scenario: Procedure reconfirms previously provided details
        Given the procedure called "Request Loan Journey"
        And a customer message, "Hi there, i need a loan for 15k for a pizza restaurant and my account is 1234"
        And an agent message, "Got it!"
        And an agent message, "Could you let me know what type of loan you're interested in? For example, are you looking for a business loan?"
        And a customer message, "personal/business one"
        And an agent message, "Alright."
        And an agent message, "Since you're considering both personal and business options, let's clarify which one suits your needs better. For a business loan, we can proceed with the application for your pizza restaurant. If you're leaning towards a personal loan, we can discuss that too. Let me know how you'd like to proceed!"
        And a customer message, "personal"
        When processing is triggered
        Then a single message event is emitted
        And the message contains either confirming that the loan is for 15k, or asking for the purpose of the loan, or asking for the account number
        And the message contains no mention of any questions other than potentially these three - 1. confirming that the loan is for 15k 2. asking for the purpose of the loan 3. asking for the account number

    Scenario: Multiple step advancement of a procedure stopped by lacking info
        Given the procedure called "Book Flight"
        And a customer message, "Hi, my name is John Smith and I'd like to book a flight for myself from Ben Gurion airport. Our flight is on the 12.10 and we wish to return on the 17.10."
        When processing is triggered
        Then a single message event is emitted
        And the message contains asking what is the destination

    Scenario: Previously answered procedure steps are skipped
        Given the procedure called "Book Flight"
        And a customer message, "Hi, my name is John Smith and I'd like to book a flight for myself from Ben Gurion airport. We flight in the 12.10 and return in the 17.10."
        And an agent message, "Hi John, thanks for reaching out! I see you're planning to fly from Ben Gurion airport. Could you please let me know your destination airport?"
        And a customer message, "Suvarnabhumi Airport, please"
        When processing is triggered
        Then a single message event is emitted
        And the message contains asking whether they want economy or business class

    Scenario: Two consecutive procedure steps with tools are running one after the other
        Given the procedure called "Change Credit Limits"
        And a customer message, "Hi I see that my credit limit is low. Can I change it?"
        And an agent message, "Sure, I can help with that. Can you please provide your account name?"
        And a customer message, "Yes, it's Alice"
        And an agent message, "Thanks, Alice. What would you like your new credit limit to be?"
        And a customer message, "$20,000"
        And an agent message, "Got it. You'd like to change your credit limit to $20,000. Please confirm the request so I can proceed."
        And a customer message, "Yes that's good thanks"
        When processing is triggered
        Then the tool calls event contains 2 tool call(s)
        And the message contains informing that the change succeed

    Scenario: Agent starts a new procedure after finishing the previous one and receiving a new customer request
        Given the procedure called "Change Credit Limits"
        And the procedure called "Reset Password Journey"
        And a customer message, "Hi I see that my credit limit is low. Can I change it?"
        And an agent message, "Sure, I can help with that. Can you please provide your account name?"
        And a customer message, "Yes, it's Alice, account number 8492834"
        And an agent message, "Thanks, Alice. What would you like your new credit limit to be?"
        And a customer message, "$20,000"
        And an agent message, "Got it. You'd like to change your credit limit to $20,000. Please confirm the request so I can proceed."
        And a customer message, "Yes that's good thanks"
        And an agent message, "Your credit limit has been successfully updated to $20,000."
        And an agent message, "Is there anything else I can help you with today?"
        And a customer message, "Actually, I see that I can't access the website. I think I need to reset my password."
        When processing is triggered
        Then no tool calls event is emitted
        And a single message event is emitted
        And the message contains asking for the username, the email address, or the phone number

    Scenario: The procedure advances correctly when pruning is needed
        Given an agent named "Digital Assistant" whose job is to assist with bank customers
        And a customer named "Guest"
        And an empty session
        And the procedure called "Lock Card Journey"
        And a customer message, "what do i do if i lost my card."
        And an agent message, "Got it."
        And an agent message, "There's a number of ways I can help you with that. Would you like to dispute a transaction, lock your card, or replace it?"
        And a customer message, "lock it"
        And an agent message, "On it."
        And a tool event with data, {"tool_calls": [{"tool_id": "built-in:list_user_cards", "arguments": {}, "result": {"data": [{"card_id": 1, "card_name": "Freedom", "card_number": "**** **** **** 1234", "card_type": "credit"}, {"card_id": 2, "card_name": "Sapphire", "card_number": "**** **** **** 5678", "card_type": "credit"}]}}]}
        And an agent message, "Here are your cards: \n- Freedom, **** **** **** 1234\n- Sapphire, **** **** **** 5678."
        And an agent message, "Which one would you like to lock?"
        And a customer message, "1234"
        And an agent message, "Got it."
        And an agent message, "Could you please provide the reason for locking the card?"
        And a customer message, "i am traveling"
        When processing is triggered
        Then a single message event is emitted
        And the message contains that the card was locked

    Scenario: Agent confirms previously provided information when procedure fast forward stops too early
        Given the procedure called "Place Food Order"
        And a customer message, "Hi! I’d like to order a sandwich with pesto in baguette bread. No extras, please. I’m keeping it simple"
        When processing is triggered
        Then a single message event is emitted
        And the message contains a confirmation that the customer don't want any extras

    Scenario: Agent executes tool on first step of procedure
        Given an agent named "Digital Assistant" whose job is to assist customer get information from our clinic
        And an empty session
        And the procedure called "Simple Lab Journey"
        And a customer message, "Can you help me get my lab results? My name is Beth Harmon"
        When processing is triggered
        Then a single message event is emitted
        And the message contains that beth harmon is healthy

    Scenario: Agent returns to a tool-requiring first step on procedure reactivation
        Given an agent named "Digital Assistant" whose job is to assist customer get information from our clinic
        And an empty session
        And the procedure called "Simple Lab Journey"
        And a customer message, "Can you help me get my lab results? My name is Beth Harmon, I'm here with my friend Bob Buckland"
        And an agent message, "Your results are back! Please call your personal doctor to receive them."
        And a customer message, "Can you just email them to me?"
        And an agent message, "Unfortunately I cannot as they contain sensitive information. Please contact Dr. Spaceman for further details"
        And a customer message, "What's his number?"
        And an agent message, "943-123-4147"
        And a customer message, "Thanks! This is Bob Buckland now. How about my results? Can you get them please?"
        When processing is triggered
        Then a single message event is emitted
        And the message contains that the customer (also known as the patient or Bob Buckland) is healthy

    Scenario: Agent chooses correct branch for procedure when some branches require tools 1
        Given an agent named "Digital Assistant" whose job is to assist customer get information from our clinic
        And an empty session
        And the procedure called "Complex Lab Journey"
        And a customer message, "Can you help me get my blood results? My name is Beth Harmon"
        When processing is triggered
        Then a single message event is emitted
        And the message contains that the customer (also known as the patient or Beth Harmon) is healthy

    Scenario: Agent chooses correct branch for procedure when some branches require tools 2
        Given an agent named "Digital Assistant" whose job is to assist customer get information from our clinic
        And an empty session
        And the procedure called "Complex Lab Journey"
        And a customer message, "Can you help me get the results to my brain scan? My name is Beth Harmon"
        When processing is triggered
        Then a single message event is emitted
        And the message contains that the requested results are not available yet
