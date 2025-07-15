Feature: Journeys
    Background:
        Given the alpha engine
        And an agent
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
        And a customer message, "What? Just reset my password please"
        When processing is triggered
        Then no tool calls event is emitted
        And a single message event is emitted
        And the message contains an answer indicating that the password cannot be reset at this time, or has otherwise failed to reset

    Scenario: Critical guideline overrides journey
        Given the journey called "Reset Password Journey"
        And a guideline to ask the customer their age, and do not continue with any other process unless it is over 21 when the customer provides a username that includes what could potentially be their year of birth
        And a customer message, "I want to reset my password"
        And an agent message, "I can help you do just that. What's your username?"
        And a customer message, "it's leonardo_barbosa_1982"
        And a journey path "[2]" for the journey "Reset Password Journey"
        When processing is triggered
        Then no tool calls event is emitted
        And a single message event is emitted
        And the message contains asking the customer for their age
        And the message contains no questions about the customer's email address or phone number
    
    Scenario: Two journeys are used in unison
        Given the journey called "Book Flight"
        And a guideline "skip steps" to skip steps that are inapplicable due to other contextual reasons when applying a book flight journey
        And a dependency relationship between the guideline "skip steps" and the "Book Flight" journey
        And a guideline "Business Adult Only" to know that travelers under the age of 21 are illegible for business class, and may only use economy when a flight is being booked
        And a customer message, "Hi, I'd like to book a flight for myself. I'm 19 if that effects anything."
        And an agent message, "Great! From and to where would are you looking to fly?"
        And a customer message, "From LAX to JFK"
        And an agent message, "Got it. And when are you looking to travel?"
        And a customer message, "Next Monday"
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
        And a guideline "Business Adult Only" to ensure that travelers under the age of 21 are ineligible for business class. If a traveler is under 21, they should be informed that only economy class is available. If the traveler is 21 or older, they may choose between economy and business class when a customer wants to book a flight
        And a guideline "business_class_age_restriction" to check that the traveler is at least 21 years old before allowing business class, otherwise restrict to economy, using the class_access_validator tool when a customer wants to book a flight
        And the tool "class_access_validator"
        And an association between "business_class_age_restriction" and "class_access_validator"
        And a dependency relationship between the guideline "Business Adult Only" and the "Book Flight" journey
        And a dependency relationship between the guideline "business_class_age_restriction" and the "Book Flight" journey
        And a customer message, "Hi, my name is John Smith and I'd like to book a flight for myself from Ben Gurion airport to JFK. We flight in the 12.10 and return in the 17.10. I'm 19 if that affects anything."
        When processing is triggered
        Then the tool calls event contains 2 tool call(s)
        And the tool calls event contains a call to "class_access_validator"
