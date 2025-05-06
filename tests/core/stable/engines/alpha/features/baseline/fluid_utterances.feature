Feature: Fluid Utterance
    Background:
        Given the alpha engine
        And an agent
        And that the agent uses the fluid_utterance message composition mode
        And an empty session

    Scenario: The agent greets the customer (fluid utterance)
        Given a guideline to greet with 'Howdy' when the session starts
        When processing is triggered
        Then a status event is emitted, acknowledging event -1
        And a status event is emitted, processing event -1
        And a status event is emitted, typing in response to event -1
        And a single message event is emitted
        And the message contains a 'Howdy' greeting
        And a status event is emitted, ready for further engagement after reacting to event -1

    Scenario: Responding based on data the user is providing (fluid utterance)
        Given a customer message, "I say that a banana is green, and an apple is purple. What did I say was the color of a banana?"
        And an utterance, "Sorry, I do not know"
        And an utterance, "The answer is {{generative.answer}}"
        When messages are emitted
        Then the message doesn't contain the text "I do not know"
        And the message mentions the color green

    Scenario: Reverting to fluid generation when a full utterance match isn't found (fluid utterance)
        Given a customer message, "I say that a banana is green, and an apple is purple. What did I say was the color of a banana?"
        And an utterance, "Sorry, I do not know"
        And an utterance, "I'm not sure. The answer might be {{generative.answer}}. How's that?"
        When messages are emitted
        Then the message doesn't contain the text "I do not know"
        And the message mentions the color green
        
    Scenario: Multistep journey is partially followed 1 (fluid utterance)
        Given a journey "reset_password_journey" to follow these steps to reset a customers password: 1. ask for their account name 2. ask for their email or phone number 3. Wish them a good day and only proceed if they wish one back to you. Otherwise abort. 3. use the tool reset_password with the provided information 4. report the result to the customer when the customer wants to reset their password
        And an utterance, "What is the name of your account?"
        And an utterance, "can you please provide the email address or phone number attached to this account?"
        And an utterance, "Thank you, have a good day!"
        And an utterance, "I'm sorry but I have no information about that"
        And an utterance, "Is there anything else I could help you with?"
        And an utterance, "Your password was successfully reset. An email with further instructions will be sent to your address."
        And an utterance, "An error occurred, your password could not be reset"
        And the tool "reset_password"
        And an association between "reset_password" and "reset_password_journey"
        And a customer message, "I want to reset my password"
        When processing is triggered
        Then no tool calls event is emitted
        And a single message event is emitted
        And the message contains asking the customer for their username, but not for their email or phone number

    Scenario: Irrelevant journey is ignored (fluid utterance)
        Given a journey "reset_password_journey" to follow these steps to reset a customers password: 1. ask for their account name 2. ask for their email or phone number 3. Wish them a good day and only proceed if they wish one back to you. Otherwise abort. 3. use the tool reset_password with the provided information 4. report the result to the customer when always
        And an utterance, "What is the name of your account?"
        And an utterance, "can you please provide the email address or phone number attached to this account?"
        And an utterance, "Thank you, have a good day!"
        And an utterance, "I'm sorry but I have no information about that"
        And an utterance, "Is there anything else I could help you with?"
        And an utterance, "Your password was successfully reset. An email with further instructions will be sent to your address."
        And an utterance, "An error occurred, your password could not be reset"
        And the tool "reset_password"
        And an association between "reset_password" and "reset_password_journey"
        And a customer message, "What are some tips I could use to come up with a strong password?"
        When processing is triggered
        Then no tool calls event is emitted
        And a single message event is emitted
        And the message contains nothing about resetting your password

    Scenario: Multistep journey is partially followed 2 (fluid utterance)
        Given a journey "reset_password_journey" to follow these steps to reset a customers password: 1. ask for their account name 2. ask for their email or phone number 3. Wish them a good day and only proceed if they wish one back to you. Otherwise abort. 3. use the tool reset_password with the provided information 4. report the result to the customer when the customer wants to reset their password
        And an utterance, "What is the name of your account?"
        And an utterance, "can you please provide the email address or phone number attached to this account?"
        And an utterance, "Thank you, have a good day!"
        And an utterance, "I'm sorry but I have no information about that"
        And an utterance, "Is there anything else I could help you with?"
        And an utterance, "Your password was successfully reset. An email with further instructions will be sent to your address."
        And an utterance, "An error occurred, your password could not be reset"
        And the tool "reset_password"
        And an association between "reset_password" and "reset_password_journey"
        And a customer message, "I want to reset my password"
        And a agent message, "I can help you do just that. What's your username?"
        And a customer message, "it's leonardo_barbosa_1982"
        When processing is triggered
        Then no tool calls event is emitted
        And a single message event is emitted
        And the message contains asking the customer for their mobile number or email address
        And the message contains nothing about wishing the customer a good day

    Scenario: Multistep journey invokes tool calls correctly (fluid utterance)
        Given a journey "reset_password_journey" to follow these steps to reset a customers password: 1. ask for their account name 2. ask for their email or phone number 3. Wish them a good day and only proceed if they wish one back to you. Otherwise abort. 3. use the tool reset_password with the provided information 4. report the result to the customer when the customer wants to reset their password
        And an utterance, "What is the name of your account?"
        And an utterance, "can you please provide the email address or phone number attached to this account?"
        And an utterance, "Thank you, have a good day!"
        And an utterance, "I'm sorry but I have no information about that"
        And an utterance, "Is there anything else I could help you with?"
        And an utterance, "Your password was successfully reset. An email with further instructions will be sent to your address."
        And an utterance, "An error occurred, your password could not be reset"
        And the tool "reset_password"
        And an association between "reset_password" and "reset_password_journey"
        And a customer message, "I want to reset my password"
        And a agent message, "I can help you do just that. What's your username?"
        And a customer message, "it's leonardo_barbosa_1982"
        And a agent message, "Great! And what's the account's associated email address or phone number?"
        And a customer message, "the email is leonardobarbosa@gmail.br"
        And a agent message, "Got it. Before proceeding to reset your password, I wanted to wish you a good day"
        And a customer message, "Thank you! Have a great day as well!"
        When processing is triggered
        Then a single tool calls event is emitted
        And the tool calls event contains 1 tool call(s)
        And the tool calls event contains a call to reset password with username leonardo_barbosa_1982 and email leonardobarbosa@gmail.br 
        And a single message event is emitted
        And the message contains that the password was reset and an email with instructions was sent to the customer

    Scenario: Multistep journey is aborted when the journey description requires so (fluid utterance)
        Given a journey "reset_password_journey" to follow these steps to reset a customers password: 1. ask for their account name 2. ask for their email or phone number 3. Wish them a good day and only proceed if they wish one back to you. Otherwise abort. 3. use the tool reset_password with the provided information 4. report the result to the customer when the customer wants to reset their password
        And an utterance, "What is the name of your account?"
        And an utterance, "can you please provide the email address or phone number attached to this account?"
        And an utterance, "Thank you, have a good day!"
        And an utterance, "I'm sorry but I have no information about that"
        And an utterance, "Is there anything else I could help you with?"
        And an utterance, "Your password was successfully reset. An email with further instructions will be sent to your address."
        And an utterance, "An error occurred, your password could not be reset"
        And the tool "reset_password"
        And an association between "reset_password" and "reset_password_journey"
        And a customer message, "I want to reset my password"
        And a agent message, "I can help you do just that. What's your username?"
        And a customer message, "it's leonardo_barbosa_1982"
        And a agent message, "Great! And what's the account's associated email address or phone number?"
        And a customer message, "the email is leonardobarbosa@gmail.br"
        And a agent message, "Got it. Before proceeding to reset your password, I wanted to wish you a good day"
        And a customer message, "Ok I guess..."
        When processing is triggered
        When processing is triggered
        Then no tool calls event is emitted
        And a single message event is emitted
        And the message contains that the password resetting has failed, or something completely unrelated to password resetting

    Scenario: Critical guideline overrides journey (fluid utterance)
        Given a journey "reset_password_journey" to follow these steps to reset a customers password: 1. ask for their account name 2. ask for their email or phone number 3. Wish them a good day and only proceed if they wish one back to you. Otherwise abort. 3. use the tool reset_password with the provided information 4. report the result to the customer when the customer wants to reset their password
        And an utterance, "What is the name of your account?"
        And an utterance, "can you please provide the email address or phone number attached to this account?"
        And an utterance, "Thank you, have a good day!"
        And an utterance, "I'm sorry but I have no information about that"
        And an utterance, "Is there anything else I could help you with?"
        And an utterance, "Your password was successfully reset. An email with further instructions will be sent to your address."
        And an utterance, "An error occurred, your password could not be reset"
        And an utterance, "Before proceeding, could you please state your age?"
        And the tool "reset_password"
        And an association between "reset_password" and "reset_password_journey"
        And a guideline to ask the customer their age, and do not continue with any other process unless it is over 21 when the customer provides a username that includes what could potentially be their year of birth
        And a customer message, "I want to reset my password"
        And a agent message, "I can help you do just that. What's your username?"
        And a customer message, "it's leonardo_barbosa_1982"
        When processing is triggered
        Then no tool calls event is emitted
        And a single message event is emitted
        And the message contains asking the customer for their age
        And the message contains no questions about the customer's email address or phone number

    Scenario: Observation is followed to inform decision (fluid utterance)
        Given a guideline "recommend_pizza" to recommend either tomato, mushrooms or pepperoni when the customer asks for topping recommendations
        And an utterance, "I recommend tomatoes"
        And an utterance, "I recommend tomatoes or mushrooms"
        And an utterance, "I recommend tomatoes, mushrooms or pepperoni"
        And an utterance, "I recommend tomatoes or pepperoni"
        And an utterance, "I recommend mushrooms"
        And an utterance, "I recommend mushrooms or pepperoni"
        And an utterance, "I recommend pepperoni"
        And an observation to remember that this means that the customer is vegetarian when the customer has a name that begins with R
        And a customer message, "Hey, there. How are you?"
        And an agent message, "I'm doing alright, thank you! What's your name?"
        And a customer message, "Rajon, have we spoken before? I want one large pie but I'm not sure which topping to get, what do you recommend?"
        When processing is triggered
        Then a single message event is emitted
        And the message contains recommendations for either mushrooms or tomatoes, but not pepperoni0
    
    Scenario: Journey information is followed (fluid utterance)
        Given a journey "informative_journey" to remember that credit limits can be decreased through this chat, using the decrease_limits tool, but that to increase credit limits you must visit a physical branch
        And an utterance, "To increase credit limits, you must visit a physical branch"
        And an utterance, "Sure. Let me check how that could be done"
        And a customer message, "Hey there. I want to increase the withdrawl limit on my platinum silver gold card. I want the new limits to be twice as high, please."
        When processing is triggered
        Then a single message event is emitted
        And the message contains that you must visit a physical branch to increase credit limits

    Scenario: Observation informs tool call parameterization (fluid utterance)
        Given a journey "reset_password_guideline" to use the reset_password tool the customer wants to reset their password and has provided their username and email address or phone number
        And the tool "reset_password"
        And an association between "reset_password" and "reset_password_journey"
        And an observation to remember that all gmail addresses with local domains are saved within our systems and tools using gmail.com instead of the local domain.
        And a customer message, "I want to reset my password"
        And a agent message, "I can help you do just that. What's your username?"
        And a customer message, "it's leonardo_barbosa_1982"
        And a agent message, "Great! And what's the account's associated email address or phone number?"
        And a customer message, "the email is leonardobarbosa@gmail.br"
        When processing is triggered
        Then a single tool calls event is emitted
        And the tool calls event contains 1 tool call(s)
        And the tool calls event contains a call to reset password with username leonardo_barbosa_1982 and email leonardobarbosa@gmail.com (NOT leonardobarbosa@gmail.br) 

    Scenario: Observation and journey are used in unison (fluid utterance)
        Given a journey "book_flight_journey" to ask for the source and destination airport first, the date second, economy or business class third, and finally to ask for the name of the traveler. You may skip steps that are inapplicable due to other contextual reasons.
        And an utterance, "Great. Are you interested in economy or business class?"
        And an utterance "Great. What is the name of the traveler?"
        And an utterance, "Great. Are you interested in economy or business class? Also, what is the name of the person traveling?"
        And an observation "no_economy" to remember that travelers under the age of 21 are illegible for business class, and may only use economy when a flight is being booked
        And a customer message, "Hi, I'd like to book a flight for myself. I'm 19 if that effects anything."
        And an agent message, "Great! From and to where would are you looking to fly?"
        And a customer message, "From LAX to JFK"
        And an agent message, "Got it. And when are you looking to travel?"
        And a customer message, "Next Monday"
        When processing is triggered
        Then a single message event is emitted
        And the message contains asking for the name of the person traveling
        And the message contains no question regarding choosing between economy and business class
