Feature: Journeys
    Background:
        Given the alpha engine
        And an agent
        And that the agent uses the canned_fluid message composition mode
        And an empty session

    Scenario: Multistep journey is partially followed 1
        Given the journey called "Reset Password Journey"
        And a customer message, "I want to reset my password"
        When processing is triggered
        Then no tool calls event is emitted
        And a single message event is emitted
        And the message contains asking the customer for their username, but not for their email or phone number

    Scenario: Irrelevant journey is ignored
        Given the journey called "Reset Password Journey"
        And a customer message, "What are some tips I could use to come up with a strong password?"
        When processing is triggered
        Then no tool calls event is emitted
        And a single message event is emitted
        And the message contains nothing about resetting your password

    Scenario: Multistep journey is partially followed 2
        Given the journey called "Reset Password Journey"
        And a customer message, "I want to reset my password"
        And an agent message, "I can help you do just that. What's your username?"
        And a customer message, "it's leonardo_barbosa_1982"
        And a journey path "[2]" for the journey "Reset Password Journey"
        When processing is triggered
        Then no tool calls event is emitted
        And a single message event is emitted
        And the message contains asking the customer for their mobile number or email address
        And the message contains nothing about wishing the customer a good day

    Scenario: Multistep journey is aborted when the journey description requires so
        Given the journey called "Reset Password Journey"
        And a journey path "[2, 3, 4]" for the journey "Reset Password Journey"
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
    
    Scenario: Guideline and journey are used in unison
        Given the journey called "Book Flight"
        And a guideline "skip steps" to skip steps that are inapplicable due to other contextual reasons when applying a book flight journey
        And a dependency relationship between the guideline "skip steps" and the "Book Flight" journey
        And a guideline "Business Adult Only" to know that travelers under the age of 21 are illegible for business class, and may only use economy when a flight is being booked
        And a customer message, "Hi, I'd like to book a flight for myself. I'm 19 if that effects anything."
        And an agent message, "Great! From and to where would are you looking to fly?"
        And a customer message, "From LAX to JFK"
        And an agent message, "Got it. And when are you looking to travel?"
        And a customer message, "Next Monday until Friday"
        And a journey path "[2, 3]" for the journey "Book Flight"
        When processing is triggered
        Then a single message event is emitted
        And the message contains either asking for the name of the person traveling, or informing them that they are only eligible for economy class

    Scenario: Journey returns to earlier step when the conversation justifies doing so (1)
        Given the journey called "Book Taxi Ride"
        And a journey path "[2, 3, 4]" for the journey "Book Taxi Ride"
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

    Scenario: Journey returns to earlier step when the conversation justifies doing so (2)
        Given the journey called "Place Food Order"
        And a journey path "[2, 3, 5]" for the journey "Place Food Order"
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

    Scenario: Dependent guidelines on journey are getting matched when journey is activated
        Given the journey called "Book Flight"
        And a guideline "under 21" to inform the customer that only economy class is available when a customer wants to book a flight and the traveler is under 21
        And a guideline "21 or older" to tell te customer they may choose between economy and business class when a customer wants to book a flight and the traveler is 21 or older
        And a dependency relationship between the guideline "under 21" and the "Book Flight" journey
        And a dependency relationship between the guideline "21 or older" and the "Book Flight" journey
        And a customer message, "Hi, my name is John Smith and I'd like to book a flight for myself from Ben Gurion airport to JFK. We flight in the 12.10 and return in the 17.10. I'm 19 if that affects anything."
        When processing is triggered
        Then a single message event is emitted
        And the message contains informing the customer that only economy class is available  

    Scenario: Multistep journey invokes tool calls correctly 2
        Given the journey called "Reset Password Journey"
        And a journey path "[2, 3, 4]" for the journey "Reset Password Journey"
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

    Scenario: Journey reconfirms previously provided details
        Given the journey called "Request Loan Journey"
        And a journey path "[2]" for the journey "Request Loan Journey"
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


    Scenario: Dependent guidelines on journey are getting matched when journey is activated
        Given the journey called "Book Flight"
        And a guideline "under 21" to inform the customer that only economy class is available when a customer wants to book a flight and the traveler is under 21
        And a guideline "21 or older" to tell te customer they may choose between economy and business class when a customer wants to book a flight and the traveler is 21 or older
        And a dependency relationship between the guideline "under 21" and the "Book Flight" journey
        And a dependency relationship between the guideline "21 or older" and the "Book Flight" journey
        And a customer message, "Hi, my name is John Smith and I'd like to book a flight for myself from Ben Gurion airport to JFK. We flight in the 12.10 and return in the 17.10. I'm 19 if that affects anything."
        When processing is triggered
        Then a single message event is emitted
        And the message contains informing the customer that only economy class is available  

    Scenario: Multiple step advancement of a journey stopped by lacking info
        Given the journey called "Book Flight"
        And a customer message, "Hi, my name is John Smith and I'd like to book a flight for myself from Ben Gurion airport. Our flight is on the 12.10 and we wish to return on the 17.10."  
        When processing is triggered
        Then a single message event is emitted
        And the message contains asking what is the destination 

    Scenario: Previously answered journey steps are skipped 
        Given the journey called "Book Flight"
        And a customer message, "Hi, my name is John Smith and I'd like to book a flight for myself from Ben Gurion airport. We flight in the 12.10 and return in the 17.10."
        And an agent message, "Hi John, thanks for reaching out! I see you're planning to fly from Ben Gurion airport. Could you please let me know your destination airport?"
        And a customer message, "Suvarnabhumi Airport, please"
        And a journey path "[2]" for the journey "Book Flight"
        When processing is triggered
        Then a single message event is emitted
        And the message contains asking whether they want economy or business class

    Scenario: Two consecutive journey steps with tools are running one after the other
        Given an agent with max iteration of 3
        And that the agent uses the canned_fluid message composition mode
        And the journey called "Change Credit Limits"
        And a customer message, "Hi I see that my credit limit is low. Can I change it?"
        And an agent message, "Sure, I can help with that. Can you please provide your account name?"
        And a customer message, "Yes, it's Alice"
        And an agent message, "Thanks, Alice. What would you like your new credit limit to be?"
        And a customer message, "$20,000"
        And an agent message, "Got it. You'd like to change your credit limit to $20,000. Please confirm the request so I can proceed."
        And a customer message, "Yes that's good thanks"
        And a journey path "[2, 3, 4]" for the journey "Change Credit Limits"
        When processing is triggered
        Then the tool calls event contains 2 tool call(s)
        And the message contains informing that the change succeed

    Scenario: Agent starts a new journey after finishing the previous one and receiving a new customer request.
        Given the journey called "Change Credit Limits"
        And the journey called "Reset Password Journey"
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
        And a journey path "[2, 3, 4, 5, 6, 7]" for the journey "Change Credit Limits"
        When processing is triggered
        Then no tool calls event is emitted
        And a single message event is emitted
        And the message contains asking for the email address or the phone number

    Scenario: reset password journey is created with new steps
        Given a journey "reset_password"
        And the journey "reset_password" is triggered when the customer wants to reset their password
        And the journey "reset_password" is triggered when the customer can't remember their password
        And a node "account_name" to ask for their account name in "reset_password" journey
        And the node "account_name" requires customer input
        And a transition from the root to "account_name" in "reset_password" journey
        And a node "email_phone" to ask for their email address or phone number in "reset_password" journey
        And the node "email_phone" requires customer input
        And a transition from "email_phone" to end when the customer said they don't have a phone or mail in "reset_password" journey
        And a transition from "account_name" to "email_phone" in "reset_password" journey
        And a node "good_day" to wish them a good day in "reset_password" journey
        And a transition from "email_phone" to "good_day" when the customer provided their email address or phone number in "reset_password" journey
        And a node "do_reset" to use the reset_password tool with the provided information in "reset_password" journey
        And the node "do_reset" uses the tool "reset_password"
        And the node "do_reset" is tool running only
        And a transition from "good_day" to "do_reset" when the customer wished you a good day in return in "reset_password" journey
        And a node "cant_reset" to apologize to the customer and report that the password cannot be reset at this time in "reset_password" journey
        And a transition from "good_day" to "cant_reset" when the customer did not immediately wish you a good day in return in "reset_password" journey
        And a transition from "cant_reset" to end in "reset_password" journey
        And a node "reset_succeed" to report the result to the customer in "reset_password" journey
        And a transition from "do_reset" to "reset_succeed" when reset_password tool returned that the password was successfully reset in "reset_password" journey
        And a transition from "do_reset" to "cant_reset" when reset_password tool returned that the password was not successfully reset, or otherwise failed in "reset_password" journey
        And a customer message, "I want to reset my password"
        When processing is triggered
        Then no tool calls event is emitted
        And a single message event is emitted
        And the message contains asking the customer for their username, but not for their email or phone number

    Scenario: Disambiguation between two journeys causes neither to immediately activate
        Given a journey "Dispute a transaction"
        And an observational guideline "fraud" when the customer suspects fraudulent activity
        And an observational guideline "issue" when the customer has an issue with some transaction
        And the journey "Dispute a transaction" is triggered by the condition "fraud"
        And the journey "Dispute a transaction" is triggered by the condition "issue"
        And a node "mother" to first ask them to verify the name of their mother in "Dispute a transaction" journey
        And a transition from the root to "mother" in "Dispute a transaction" journey
        And a journey "Lock card"
        And an observational guideline "recognize" when the customer does not recognize activity on their card
        And an observational guideline "fraud2" when the customer suspects fraudulent activity
        And the journey "Lock card" is triggered by the condition "recognize"
        And the journey "Lock card" is triggered by the condition "fraud2"
        And a node "father" to first ask them to verify the name of their father in "Lock card" journey
        And a transition from the root to "father" in "Lock card" journey
        And a disambiguation group head "issue_with_transaction" to activate when the customer has an issue with a transaction but it's not clear what they action they want to take in its regard
        And a guideline "fraud" is grouped under "issue_with_transaction"
        And a guideline "issue" is grouped under "issue_with_transaction"
        And a guideline "recognize" is grouped under "issue_with_transaction"
        And a guideline "fraud2" is grouped under "issue_with_transaction"
        And a customer message, "Hi, there's a transaction I don't recognize. I need help."
        When processing is triggered
        Then a single message event is emitted
        And the message contains no mention of mother or father

    Scenario: Disambiguation between two journeys causes neither to immediately activate 2
        Given a journey "Dispute a transaction"
        And an observational guideline "suspect" when the customer suspects a transaction
        And an observational guideline "dispute" when the customer asked to dispute a transaction
        And the journey "Dispute a transaction" is triggered by the condition "suspect"
        And the journey "Dispute a transaction" is triggered by the condition "dispute"
        And a node "mother" to first ask them to verify the name of their mother in "Dispute a transaction" journey
        And a transition from the root to "mother" in "Dispute a transaction" journey
        And a journey "Lock card"
        And an observational guideline "lost" when the customer suspects the card was stolen or lost
        And an observational guideline "lock" when the customer asks to lock their card
        And the journey "Lock card" is triggered by the condition "lost"
        And the journey "Lock card" is triggered by the condition "lock"
        And a node "father" to first ask them to verify the name of their father in "Lock card" journey
        And a transition from the root to "father" in "Lock card" journey
        And a disambiguation group head "issue_with_transaction" to activate when the customer has an issue with a transaction but it's not clear what they action they want to take in its regard
        And a guideline "suspect" is grouped under "issue_with_transaction"
        And a guideline "dispute" is grouped under "issue_with_transaction"
        And a guideline "lost" is grouped under "issue_with_transaction"
        And a guideline "lock" is grouped under "issue_with_transaction"
        And a customer message, "Hi, there's a transaction I don't recognize. I need help."
        When processing is triggered
        Then a single message event is emitted
        And the message contains no mention of mother or father

    Scenario: Disambiguation between a journey and a guideline causes neither to immediately activate 
        Given a journey "Dispute a transaction"
        And a guideline "dispute" to tell them to reach our website when the customer wants to dispute a transaction
        And a journey "Lock card"
        And an observational guideline "lost" when the customer suspects the card was stolen or lost
        And an observational guideline "lock" when the customer asks to lock their card
        And the journey "Lock card" is triggered by the condition "lost"
        And the journey "Lock card" is triggered by the condition "lock"
        And a node "name" to first ask them to their name in "Lock card" journey
        And a disambiguation group head "issue_with_transaction" to activate when the customer has an issue with a transaction but it's not clear what they action they want to take in its regard
        And a guideline "dispute" is grouped under "issue_with_transaction"
        And a guideline "lost" is grouped under "issue_with_transaction"
        And a guideline "lock" is grouped under "issue_with_transaction"
        And a customer message, "Hi, there's a transaction I don't recognize. I need help."
        When processing is triggered
        Then a single message event is emitted
        And the message contains no mention of asking their name or tell them to reach the website 

    Scenario: The journey advances correctly when pruning is needed
        Given an agent named "Digital Assistant" Whose job is to assist with bank customers
        And that the agent uses the canned_fluid message composition mode
        And a customer named "Guest"
        And an empty session
        And the journey called "Lock Card Journey"
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
        And a journey path "[2, 3, 4]" for the journey "Lock Card Journey"
        When processing is triggered
        Then a single message event is emitted
        And the message contains that the card was locked

    Scenario: Agent confirms previously provided information when journey fast forward stops too early
        Given the journey called "Place Food Order"
        And a customer message, "Hi! I’d like to order a sandwich with pesto in baguette bread. No extras, please. I’m keeping it simple"
        When processing is triggered
        Then a single message event is emitted
        And the message contains a confirmation that the customer don't want any extras

    Scenario: Agent executes tool on first step of journey
        Given an agent named "Digital Assistant" Whose job is to assist customer get information from our clinic
        And that the agent uses the canned_fluid message composition mode
        And an empty session
        And the journey called "Simple Lab Journey"
        And a customer message, "Can you help me get my lab results? My name is Beth Harmon"
        When processing is triggered
        Then a single message event is emitted
        And the message contains that beth harmon is healthy

    Scenario: Agent returns to root that requires tool calls on journeys reactivation
        Given an agent named "Digital Assistant" Whose job is to assist customer get information from our clinic
        And that the agent uses the canned_fluid message composition mode
        And an empty session
        And the journey called "Simple Lab Journey"
        And a customer message, "Can you help me get my lab results? My name is Beth Harmon, I'm here with my friend Bob Buckland"
        And an agent message, "Your results are back! Please call your personal doctor to receive them."
        And a customer message, "Can you just email them to me?"
        And an agent message, "Unfortunately I cannot as they contain sensitive information. Please contact Dr. Spaceman for further details"
        And a customer message, "What's his number?"
        And an agent message, "943-123-4147"
        And a customer message, "Thanks! This is Bob Buckland now. How about my results? Can you get them please?"
        And a journey path "[2, 3, None]" for the journey "Simple Lab Journey"
        When processing is triggered
        Then a single message event is emitted
        And the message contains that the customer (also known as the patient or Bob Buckland) is healthy

    Scenario: Agent chooses correct root for journey when some roots require tools 1
        Given an agent named "Digital Assistant" Whose job is to assist customer get information from our clinic
        And that the agent uses the canned_fluid message composition mode
        And an empty session
        And the journey called "Complex Lab Journey"
        And a customer message, "Can you help me get my blood results? My name is Beth Harmon"
        When processing is triggered
        Then a single message event is emitted
        And the message contains that the customer (also known as the patient or Beth Harmon) is healthy

    Scenario: Agent chooses correct root for journey when some roots require tools 2
        Given an agent named "Digital Assistant" Whose job is to assist customer get information from our clinic
        And that the agent uses the canned_fluid message composition mode
        And an empty session
        And the journey called "Complex Lab Journey"
        And a customer message, "Can you help me get the results to my brain scan? My name is Beth Harmon"
        When processing is triggered
        Then a single message event is emitted
        And the message contains that the requested results are not available yet
    