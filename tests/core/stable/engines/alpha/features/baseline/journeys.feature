Feature: Journeys
    Background:
        Given the alpha engine
        And an empty session

    Scenario: Multistep journey is partially followed 1
        Given an agent
        And the journey called "Reset Password Journey"
        And a customer message, "I want to reset my password"
        When processing is triggered
        Then no tool calls event is emitted
        And a single message event is emitted
        And the message contains asking the customer for their username, but not for their email or phone number

    Scenario: Irrelevant journey is ignored
        Given an agent
        And the journey called "Reset Password Journey"
        And a customer message, "What are some tips I could use to come up with a strong password?"
        When processing is triggered
        Then no tool calls event is emitted
        And a single message event is emitted
        And the message contains nothing about resetting your password

    Scenario: Multistep journey is partially followed 2
        Given an agent
        And the journey called "Reset Password Journey"
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
        Given an agent
        And the journey called "Reset Password Journey"
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
    
    Scenario: Two journeys are used in unison
        Given an agent
        And the journey called "Book Flight"
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
        Given an agent
        And the journey called "Book Taxi Ride"
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
        Given an agent
        And the journey called "Place Food Order"
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
        Given an agent
        And the journey called "Book Flight"
        And a guideline "under 21" to inform the customer that only economy class is available when a customer wants to book a flight and the traveler is under 21
        And a guideline "21 or older" to tell te customer they may choose between economy and business class when a customer wants to book a flight and the traveler is 21 or older
        And a dependency relationship between the guideline "under 21" and the "Book Flight" journey
        And a dependency relationship between the guideline "21 or older" and the "Book Flight" journey
        And a customer message, "Hi, my name is John Smith and I'd like to book a flight for myself from Ben Gurion airport to JFK. We flight in the 12.10 and return in the 17.10. I'm 19 if that affects anything."
        When processing is triggered
        Then a single message event is emitted
        And the message contains informing the customer that only economy class is available  

    Scenario: Multistep journey invokes tool calls correctly
        Given an agent 
        And the journey called "Reset Password Journey"
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
        Given an agent
        And the journey called "Request Loan Journey"
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
        Given an agent
        And the journey called "Book Flight"
        And a guideline "under 21" to inform the customer that only economy class is available when a customer wants to book a flight and the traveler is under 21
        And a guideline "21 or older" to tell te customer they may choose between economy and business class when a customer wants to book a flight and the traveler is 21 or older
        And a dependency relationship between the guideline "under 21" and the "Book Flight" journey
        And a dependency relationship between the guideline "21 or older" and the "Book Flight" journey
        And a customer message, "Hi, my name is John Smith and I'd like to book a flight for myself from Ben Gurion airport to JFK. We flight in the 12.10 and return in the 17.10. I'm 19 if that affects anything."
        When processing is triggered
        Then a single message event is emitted
        And the message contains informing the customer that only economy class is available  

    Scenario: Multiple step advancement of a journey stopped by lacking info
        Given an agent
        And the journey called "Book Flight"
        And a customer message, "Hi, my name is John Smith and I'd like to book a flight for myself from Ben Gurion airport. Our flight is on the 12.10 and we wish to return on the 17.10."  
        When processing is triggered
        Then a single message event is emitted
        And the message contains asking what is the destination 

    Scenario: Previously answered journey steps are skipped 
        Given an agent
        And the journey called "Book Flight"
        And a customer message, "Hi, my name is John Smith and I'd like to book a flight for myself from Ben Gurion airport. We flight in the 12.10 and return in the 17.10."
        And an agent message, "Hi John, thanks for reaching out! I see you're planning to fly from Ben Gurion airport. Could you please let me know your destination airport?"
        And a customer message, "Suvarnabhumi Airport, please"
        And a journey path "[2]" for the journey "Book Flight"
        When processing is triggered
        Then a single message event is emitted
        And the message contains asking whether they want economy or business class

    Scenario: Two consecutive journey steps with tools are running one after the other
        Given an agent with max iteration of 3
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
        Given an agent
        And the journey called "Change Credit Limits"
        And the journey called "Reset Password Journey"
        And a customer message, "Hi I see that my credit limit is low. Can I change it?"
        And an agent message, "Sure, I can help with that. Can you please provide your account name?"
        And a customer message, "Yes, it's Alice"
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
        And the message contains asking for account number

        