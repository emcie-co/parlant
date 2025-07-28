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
        Given a journey titled "Reset Password Journey" to follow these steps to reset a customers password: 1. ask for their account name 2. ask for their email or phone number 3. Wish them a good day and only proceed if they wish one back to you. Otherwise abort. 3. use the tool reset_password with the provided information 4. report the result to the customer when the customer wants to reset their password
        And an utterance, "What is the name of your account?"
        And an utterance, "can you please provide the email address or phone number attached to this account?"
        And an utterance, "Thank you, have a good day!"
        And an utterance, "I'm sorry but I have no information about that"
        And an utterance, "Is there anything else I could help you with?"
        And an utterance, "Your password was successfully reset. An email with further instructions will be sent to your address."
        And an utterance, "An error occurred, your password could not be reset"
        And the tool "reset_password"
        And a customer message, "I want to reset my password"
        When processing is triggered
        Then no tool calls event is emitted
        And a single message event is emitted
        And the message contains asking the customer for their username, but not for their email or phone number

    Scenario: Irrelevant journey is ignored (fluid utterance)
        Given a journey titled "Reset Password Journey" to follow these steps to reset a customers password: 1. ask for their account name 2. ask for their email or phone number 3. Wish them a good day and only proceed if they wish one back to you. Otherwise abort. 3. use the tool reset_password with the provided information 4. report the result to the customer when always
        And an utterance, "What is the name of your account?"
        And an utterance, "can you please provide the email address or phone number attached to this account?"
        And an utterance, "Thank you, have a good day!"
        And an utterance, "I'm sorry but I have no information about that"
        And an utterance, "Is there anything else I could help you with?"
        And an utterance, "Your password was successfully reset. An email with further instructions will be sent to your address."
        And an utterance, "An error occurred, your password could not be reset"
        And the tool "reset_password"
        And a customer message, "What are some tips I could use to come up with a strong password?"
        When processing is triggered
        Then no tool calls event is emitted
        And a single message event is emitted
        And the message contains nothing about resetting your password

    Scenario: Multistep journey is partially followed 2 (fluid utterance)
        Given a journey titled "Reset Password Journey" to follow these steps to reset a customers password: 1. ask for their account name 2. ask for their email or phone number 3. Wish them a good day and only proceed if they wish one back to you. Otherwise abort. 3. use the tool reset_password with the provided information 4. report the result to the customer when the customer wants to reset their password
        And an utterance, "What is the name of your account?"
        And an utterance, "can you please provide the email address or phone number attached to this account?"
        And an utterance, "Thank you, have a good day!"
        And an utterance, "I'm sorry but I have no information about that"
        And an utterance, "Is there anything else I could help you with?"
        And an utterance, "Your password was successfully reset. An email with further instructions will be sent to your address."
        And an utterance, "An error occurred, your password could not be reset"
        And the tool "reset_password"
        And a customer message, "I want to reset my password"
        And an agent message, "I can help you do just that. What's your username?"
        And a customer message, "it's leonardo_barbosa_1982"
        When processing is triggered
        Then no tool calls event is emitted
        And a single message event is emitted
        And the message contains asking the customer for their mobile number or email address
        And the message contains nothing about wishing the customer a good day

    Scenario: Critical guideline overrides journey (fluid utterance)
        Given a journey titled "Reset Password Journey" to follow these steps to reset a customers password: 1. ask for their account name 2. ask for their email or phone number 3. Wish them a good day and only proceed if they wish one back to you. Otherwise abort. 3. use the tool reset_password with the provided information 4. report the result to the customer when the customer wants to reset their password
        And an utterance, "What is the name of your account?"
        And an utterance, "can you please provide the email address or phone number attached to this account?"
        And an utterance, "Thank you, have a good day!"
        And an utterance, "I'm sorry but I have no information about that"
        And an utterance, "Is there anything else I could help you with?"
        And an utterance, "Your password was successfully reset. An email with further instructions will be sent to your address."
        And an utterance, "An error occurred, your password could not be reset"
        And an utterance, "Before proceeding, could you please state your age?"
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
    
    Scenario: Journey information is followed (fluid utterance)
        Given a journey titled "Change Credit Limits" to remember that credit limits can be decreased through this chat, using the decrease_limits tool, but that to increase credit limits you must visit a physical branch when credit limits are discussed
        And an utterance, "To increase credit limits, you must visit a physical branch"
        And an utterance, "Sure. Let me check how that could be done"
        And a customer message, "Hey there. I want to increase the credit limit on my platinum silver gold card. I want the new limits to be twice as high, please."
        When processing is triggered
        Then a single message event is emitted
        And the message contains that you must visit a physical branch to increase credit limits

    Scenario: Two journeys are used in unison (fluid utterance)
        Given a journey titled "Book Flight" to ask for the source and destination airport first, the date second, economy or business class third, and finally to ask for the name of the traveler. You may skip steps that are inapplicable due to other contextual reasons. when a customer wants to book a flight
        And an utterance, "Great. Are you interested in economy or business class?"
        And an utterance, "Great. Only economy class is available for this booking. What is the name of the traveler?"
        And an utterance, "Great. What is the name of the traveler?"
        And an utterance, "Great. Are you interested in economy or business class? Also, what is the name of the person traveling?"
        And a journey titled "No Economy" to remember that travelers under the age of 21 are illegible for business class, and may only use economy when a flight is being booked
        And a customer message, "Hi, I'd like to book a flight for myself. I'm 19 if that effects anything."
        And an agent message, "Great! From and to where would are you looking to fly?"
        And a customer message, "From LAX to JFK"
        And an agent message, "Got it. And when are you looking to travel?"
        And a customer message, "Next Monday"
        When processing is triggered
        Then a single message event is emitted
        And the message contains either asking for the name of the person traveling, or informing them that they are only eligible for economy class


    Scenario: The agent greets the customer 2 (fluid utterance)
        Given a guideline to greet with 'Howdy' when the session starts
        And an utterance, "Hello there! How can I help you today?"
        And an utterance, "Howdy! How can I be of service to you today?"
        And an utterance, "Thank you for your patience!"
        And an utterance, "Is there anything else I could help you with?"
        And an utterance, "I'll look into that for you right away."
        When processing is triggered
        Then a status event is emitted, acknowledging event -1
        And a status event is emitted, processing event -1
        And a status event is emitted, typing in response to event -1
        And a single message event is emitted
        And the message contains a 'Howdy' greeting

    Scenario: The agent offers a thirsty customer a drink (fluid utterance)
        Given a customer message, "I'm thirsty"
        And a guideline to offer thirsty customers a Pepsi when the customer is thirsty
        And an utterance, "Would you like a Pepsi? I can get one for you right away."
        And an utterance, "I understand you're thirsty. Can I get you something to drink?"
        And an utterance, "Is there anything specific you'd like to drink?"
        And an utterance, "Thank you for letting me know. Is there anything else I can help with?"
        And an utterance, "I'll be happy to assist you with all your beverage needs today."
        When processing is triggered
        Then a status event is emitted, acknowledging event 0
        And a status event is emitted, processing event 0
        And a status event is emitted, typing in response to event 0
        And a single message event is emitted
        And the message contains an offering of a Pepsi
        And a status event is emitted, ready for further engagement after reacting to event 0

    Scenario: The agent correctly applies greeting guidelines based on auxiliary data (fluid utterance)
        Given an agent named "Chip Bitman" whose job is to work at a tech store and help customers choose what to buy. You're clever, witty, and slightly sarcastic. At the same time you're kind and funny.
        And that the agent uses the fluid_utterance message composition mode
        And a customer named "Beef Wellington"
        And an empty session with "Beef Wellingotn"
        And the term "Bug" defined as The name of our tech retail store, specializing in gadgets, computers, and tech services.
        And the term "Bug-Free" defined as Our free warranty and service package that comes with every purchase and covers repairs, replacements, and tech support beyond the standard manufacturer warranty.
        And a tag "business"
        And a customer tagged as "business"
        And a context variable "plan" set to "Business Plan" for the tag "business"
        And a guideline to just welcome them to the store and ask how you can help when the customer greets you
        And a guideline to refer to them by their first name only, and welcome them 'back' when a customer greets you
        And a guideline to assure them you will escalate it internally and get back to them when a business-plan customer is having an issue
        And a customer message, "Hi there"
        And an utterance, "Hi Beef! Welcome back to Bug. What can I help you with today?"
        And an utterance, "Hello there! How can I assist you today?"
        And an utterance, "Welcome to Bug! Is this your first time shopping with us?"
        And an utterance, "I'll escalate this issue internally and get back to you as soon as possible."
        And an utterance, "Have you heard about our Bug-Free warranty program?"
        When processing is triggered
        Then a single message event is emitted
        And the message contains the name 'Beef'
        And the message contains a welcoming back of the customer to the store and asking how the agent could help

    Scenario: The agent follows a guideline with agent intention (fluid utterance)
        Given a guideline to do not provide any personal medical information even if you have it when you discusses a patient's medical record
        And that the agent uses the fluid_utterance message composition mode
        And a customer named "Alex Smith"
        And an empty session with "Alex Smith"
        And a context variable "medical_record" set to "Amoxicillin and Lisinopril" for "Alex Smith" 
        And a customer message, "Hi, I need to know what medications I was prescribed during my visit last month. Can you pull up my medical record?"
        And an utterance, "I'm not able to provide personal medical information from your records."
        And an utterance, "I can help you with that. You were prescribed the following medications: {{generative.medication}}"
        When processing is triggered
        Then a single message event is emitted
        And the message contains no prescription of medications 
        And the message contains explanation that can't provide personal medical information

    Scenario: The agent ignores a matched agent intention guideline when you doesn't intend to do its condition (fluid utterance)
        Given a guideline to remind that we have a special sale if they book today when you recommends on flights options
        Given a guideline to suggest only ground based travel options when the customer asks about travel options
        And that the agent uses the fluid_utterance message composition mode
        And a customer message, "Hi, I want to go to California from New york next week. What are my options?"
        And an utterance, "I recommend taking a direct flight. It's the most efficient and comfortable option."
        And an utterance, "I recommend taking a train or a long-distance bus service. It's the most efficient and comfortable option"
        And an utterance, "I recommend taking a direct flight. It's the most efficient and comfortable option. We also have a special sale if you book today!"
        And an utterance, "I recommend taking a train or a long-distance bus service. It's the most efficient and comfortable option. We also have a special sale if you book today!"
        When processing is triggered
        Then a single message event is emitted
        And the message contains a suggestion to travel with bus or train but not with a flight
        And the message contains no sale option
