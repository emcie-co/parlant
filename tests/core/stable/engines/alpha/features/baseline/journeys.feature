Feature: Journeys
    Background:
        Given the alpha engine
        And an agent
        And an empty session

    Scenario: Multistep journey is partially followed 1
        Given a journey titled "Reset Password Journey" to follow these steps to reset a customers password: 1. ask for their account name 2. ask for their email or phone number 3. Wish them a good day and only proceed if they wish one back to you. Otherwise abort. 3. use the tool reset_password with the provided information 4. report the result to the customer when the customer wants to reset their password
        And the tool "reset_password"
        And a customer message, "I want to reset my password"
        When processing is triggered
        Then no tool calls event is emitted
        And a single message event is emitted
        And the message contains asking the customer for their username, but not for their email or phone number

    Scenario: Irrelevant journey is ignored
        Given a journey titled "Reset Password Journey" to follow these steps to reset a customers password: 1. ask for their account name 2. ask for their email or phone number 3. Wish them a good day and only proceed if they wish one back to you. Otherwise abort. 3. use the tool reset_password with the provided information 4. report the result to the customer when always
        And the tool "reset_password"
        And a customer message, "What are some tips I could use to come up with a strong password?"
        When processing is triggered
        Then no tool calls event is emitted
        And a single message event is emitted
        And the message contains nothing about resetting your password

    Scenario: Multistep journey is partially followed 2
        Given a journey titled "Reset Password Journey" to follow these steps to reset a customers password: 1. ask for their account name 2. ask for their email or phone number 3. Wish them a good day and only proceed if they wish one back to you. Otherwise abort. 3. use the tool reset_password with the provided information 4. report the result to the customer when the customer wants to reset their password
        And the tool "reset_password"
        And a customer message, "I want to reset my password"
        And an agent message, "I can help you do just that. What's your username?"
        And a customer message, "it's leonardo_barbosa_1982"
        When processing is triggered
        Then no tool calls event is emitted
        And a single message event is emitted
        And the message contains asking the customer for their mobile number or email address
        And the message contains nothing about wishing the customer a good day

    Scenario: Multistep journey is aborted when the journey description requires so
        Given a journey titled "Reset Password Journey" to follow these steps to reset a customers password: 1. ask for their account name 2. ask for their email or phone number 3. Wish them a good day and only proceed if they wish one back to you. Otherwise refuse to continue with resetting their password, without explaining the reason for the refusal. 4. use the tool reset_password with the provided information 5. report the result to the customer when the customer wants to reset their password
        And the tool "reset_password"
        And a customer message, "I want to reset my password"
        And an agent message, "I can help you do just that. What's your username?"
        And a customer message, "it's leonardo_barbosa_1982"
        And an agent message, "Great! And what's the account's associated email address or phone number?"
        And a customer message, "the email is leonardobarbosa@gmail.br"
        And an agent message, "Got it. Before proceeding to reset your password, I wanted to wish you a good day"
        And a customer message, "What? Just reset my password please"
        When processing is triggered
        Then no tool calls event is emitted
        And a single message event is emitted
        And the message contains an answer indicating that the password cannot be reset at this time, or has otherwise failed to reset


    Scenario: Critical guideline overrides journey
        Given a journey titled "Reset Password Journey" to follow these steps to reset a customers password: 1. ask for their account name 2. ask for their email or phone number 3. Wish them a good day and only proceed if they wish one back to you. Otherwise abort. 3. use the tool reset_password with the provided information 4. report the result to the customer when the customer wants to reset their password
        And the tool "reset_password"
        And a guideline to ask the customer their age, and do not continue with any other process unless it is over 21 when the customer provides a username that includes what could potentially be their year of birth
        And a customer message, "I want to reset my password"
        And an agent message, "I can help you do just that. What's your username?"
        And a customer message, "it's leonardo_barbosa_1982"
        When processing is triggered
        Then no tool calls event is emitted
        And a single message event is emitted
        And the message contains asking the customer for their age
        And the message contains no questions about the customer's email address or phone number
    
    Scenario: Journey information is followed
        Given a journey titled "Change Credit Limits" to remember that credit limits can be decreased through this chat, using the decrease_limits tool, but that to increase credit limits you must visit a physical branch when credit limits are discussed
        And a customer message, "Hey there. I want to increase the credit limit on my platinum silver gold card. I want the new limits to be twice as high, please."
        When processing is triggered
        Then a single message event is emitted
        And the message contains that you must visit a physical branch to increase credit limits

    Scenario: Journey informs tool call parameterization
        Given a guideline "reset_password_guideline" to use the reset_password tool when the customer wants to reset their password and has provided their username and email address or phone number
        And the tool "reset_password"
        And an association between "reset_password_guideline" and "reset_password"
        And a journey titled "Email Domain" to remember that all gmail addresses with local domains are saved within our systems and tools using gmail.com instead of the local domain when a gmail address with a domain other than .com is mentioned
        And a customer message, "I want to reset my password"
        And an agent message, "I can help you do just that. What's your username?"
        And a customer message, "it's leonardo_barbosa_1982"
        And an agent message, "Great! And what's the account's associated email address or phone number?"
        And a customer message, "the email is leonardobarbosa@gmail.br"
        When processing is triggered
        Then a single tool calls event is emitted
        And the tool calls event contains 1 tool call(s)
        And the tool calls event contains a call to reset password with username leonardo_barbosa_1982 and email leonardobarbosa@gmail.com (NOT leonardobarbosa@gmail.br) 

    Scenario: Two journeys are used in unison
        Given a journey titled "Book Flight" to ask for the source and destination airport first, the date second, economy or business class third, and finally to ask for the name of the traveler. You may skip steps that are inapplicable due to other contextual reasons. when a customer wants to book a flight
        And a journey titled "Business Adult Only" to know that travelers under the age of 21 are illegible for business class, and may only use economy when a flight is being booked
        And a customer message, "Hi, I'd like to book a flight for myself. I'm 19 if that effects anything."
        And an agent message, "Great! From and to where would are you looking to fly?"
        And a customer message, "From LAX to JFK"
        And an agent message, "Got it. And when are you looking to travel?"
        And a customer message, "Next Monday"
        When processing is triggered
        Then a single message event is emitted
        And the message contains either asking for the name of the person traveling, or informing them that they are only eligible for economy class

    Scenario: Simple journey is followed to inform decision
        Given a guideline "recommend_pizza" to recommend either tomato, mushrooms or pepperoni when the customer asks for topping recommendations
        And a journey titled "Vegetarian Customers" to Be aware that the customer is vegetarian. Only discuss vegetarian options with them. when the customer has a name that begins with R
        And a customer message, "Hey, there. How are you?"
        And an agent message, "I'm doing alright, thank you! What's your name?"
        And a customer message, "Rajon, have we spoken before? I want one large pie but I'm not sure which topping to get, what do you recommend?"
        When processing is triggered
        Then a single message event is emitted
        And the message contains recommendations for either mushrooms or tomatoes, but not pepperoni

    Scenario: Journey returns to earlier step when the conversation justifies doing so (1)
        Given a journey titled "Book Taxi Ride" to follow these steps to book a customer a taxi ride: 1. Ask for the pickup location. 2. Ask for the drop-off location. 3. Ask for the desired pickup time. 4. Confirm all details with the customer before booking. Each step should be handled in a separate message. when the customer wants to book a taxi
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
        Given a journey titled "Place Food Order" to follow these steps to place a customer’s order: 1. Ask if they’d like a salad or a sandwich. 2. If they choose a sandwich, ask what kind of bread they’d like. 3. If they choose a sandwich, ask what main filling they’d like from: Peanut butter, jam or pesto. 4. If they choose a sandwich, ask if they want any extras. 5. If they choose a salad, ask what base greens they want. 6. If they choose a salad, ask what toppings they’d like. 7. If they choose a salad, ask what kind of dressing they prefer. 8. Confirm the full order before placing it. Each step should be handled in a separate message, when the customer wants to order food 
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

    Scenario: Dependent guidelines on multiple journeys are getting matched when journeys are activated
        Given a journey titled "Book Flight" to ask for the source and destination airport. You may skip steps that are inapplicable due to other contextual reasons. when a customer wants to book a flight
        And a journey titled "Business Adult Only" to ensure that travelers under the age of 21 are ineligible for business class. If a traveler is under 21, they should be informed that only economy class is available. If the traveler is 21 or older, they may choose between economy and business class when a customer wants to book a flight
        And a guideline "book_flight_guideline" to collect the source and destination airport, and then use the book_flight tool when a customer wants to book a flight
        And a guideline "business_class_age_restriction" to check that the traveler is at least 21 years old before allowing business class, otherwise restrict to economy, using the class_access_validator tool when a customer wants to book a flight
        And the tool "book_flight"
        And the tool "class_access_validator"
        And an association between "book_flight_guideline" and "book_flight"
        And an association between "business_class_age_restriction" and "class_access_validator"
        And a dependency relationship between the guideline "book_flight_guideline" and the "Book Flight" journey
        And a dependency relationship between the guideline "business_class_age_restriction" and the "Business Adult Only" journey
        And a customer message, "Hi, my name is John Smith and I'd like to book a flight for myself from Ben Gurion airport to JFK. I'm 19 if that affects anything."
        When processing is triggered
        Then the tool calls event contains 2 tool call(s)
        And the tool calls event contains a call to "class_access_validator"
        And the tool calls event contains a call to "book_flight"
