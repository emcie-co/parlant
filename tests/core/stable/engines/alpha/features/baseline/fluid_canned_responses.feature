Feature: Fluid Canned Response
    Background:
        Given the alpha engine
        And an agent
        And that the agent uses the canned_fluid message composition mode
        And an empty session

    Scenario: The agent greets the customer (fluid canned response)
        Given a guideline to greet with 'Howdy' when the session starts
        When processing is triggered
        Then a status event is emitted, acknowledging event
        And a status event is emitted, typing in response to event
        And a single message event is emitted
        And the message contains a 'Howdy' greeting
        And a status event is emitted, ready for further engagement after reacting to event

    Scenario: Responding based on data the user is providing (fluid canned response)
        Given a customer message, "I say that a banana is green, and an apple is purple. What did I say was the color of a banana?"
        And a canned response, "Sorry, I do not know"
        And a canned response, "The answer is {{generative.answer}}"
        When messages are emitted
        Then the message doesn't contain the text "I do not know"
        And the message mentions the color green

    Scenario: Reverting to fluid generation when a full canned response match isn't found (fluid canned response)
        Given a customer message, "I say that a banana is green, and an apple is purple. What did I say was the color of a banana?"
        And a canned response, "Sorry, I do not know"
        And a canned response, "I'm not sure. The answer might be {{generative.answer}}. How's that?"
        When messages are emitted
        Then the message doesn't contain the text "I do not know"
        And the message mentions the color green
        
    Scenario: Multistep journey is partially followed 1 (fluid canned response)
        Given the journey called "Reset Password Journey"
        And a canned response, "What is the name of your account?"
        And a canned response, "can you please provide the email address or phone number attached to this account?"
        And a canned response, "Thank you, have a good day!"
        And a canned response, "I'm sorry but I have no information about that"
        And a canned response, "Is there anything else I could help you with?"
        And a canned response, "Your password was successfully reset. An email with further instructions will be sent to your address."
        And a canned response, "An error occurred, your password could not be reset"
        And the tool "reset_password"
        And a customer message, "I want to reset my password"
        When processing is triggered
        Then no tool calls event is emitted
        And a single message event is emitted
        And the message contains asking the customer for their username, but not for their email or phone number

    Scenario: Irrelevant journey is ignored (fluid canned response)
        Given the journey called "Reset Password Journey"
        And a canned response, "What is the name of your account?"
        And a canned response, "can you please provide the email address or phone number attached to this account?"
        And a canned response, "Thank you, have a good day!"
        And a canned response, "I'm sorry but I have no information about that"
        And a canned response, "Is there anything else I could help you with?"
        And a canned response, "Your password was successfully reset. An email with further instructions will be sent to your address."
        And a canned response, "An error occurred, your password could not be reset"
        And the tool "reset_password"
        And a customer message, "What are some tips I could use to come up with a strong password?"
        When processing is triggered
        Then no tool calls event is emitted
        And a single message event is emitted
        And the message contains nothing about resetting your password

    Scenario: Multistep journey is partially followed 2 (fluid canned response)
        Given the journey called "Reset Password Journey"
        And a canned response, "What is the name of your account?"
        And a canned response, "can you please provide the email address or phone number attached to this account?"
        And a canned response, "Thank you, have a good day!"
        And a canned response, "I'm sorry but I have no information about that"
        And a canned response, "Is there anything else I could help you with?"
        And a canned response, "Your password was successfully reset. An email with further instructions will be sent to your address."
        And a canned response, "An error occurred, your password could not be reset"
        And the tool "reset_password"
        And a customer message, "I want to reset my password"
        And an agent message, "I can help you do just that. What's your username?"
        And a customer message, "it's leonardo_barbosa_1982"
        And a journey path "[2]" for the journey "Reset Password Journey"
        When processing is triggered
        Then no tool calls event is emitted
        And a single message event is emitted
        And the message contains asking the customer for their mobile number or email address
        And the message contains nothing about wishing the customer a good day


    Scenario: The agent greets the customer 2 (fluid canned response)
        Given a guideline to greet with 'Howdy' when the session starts
        And a canned response, "Hello there! How can I help you today?"
        And a canned response, "Howdy! How can I be of service to you today?"
        And a canned response, "Thank you for your patience!"
        And a canned response, "Is there anything else I could help you with?"
        And a canned response, "I'll look into that for you right away."
        When processing is triggered
        Then a status event is emitted, acknowledging event
        And a status event is emitted, processing event
        And a status event is emitted, typing in response to event
        And a single message event is emitted
        And the message contains a 'Howdy' greeting

    Scenario: The agent offers a thirsty customer a drink (fluid canned response)
        Given a customer message, "I'm thirsty"
        And a guideline to offer thirsty customers a Pepsi when the customer is thirsty
        And a canned response, "Would you like a Pepsi? I can get one for you right away."
        And a canned response, "I understand you're thirsty. Can I get you something to drink?"
        And a canned response, "Is there anything specific you'd like to drink?"
        And a canned response, "Thank you for letting me know. Is there anything else I can help with?"
        And a canned response, "I'll be happy to assist you with all your beverage needs today."
        When processing is triggered
        Then a status event is emitted, acknowledging event
        And a status event is emitted, processing event
        And a status event is emitted, typing in response to event
        And a single message event is emitted
        And the message contains an offering of a Pepsi
        And a status event is emitted, ready for further engagement after reacting to event

    Scenario: The agent correctly applies greeting guidelines based on auxiliary data (fluid canned response)
        Given an agent named "Chip Bitman" whose job is to work at a tech store and help customers choose what to buy. You're clever, witty, and slightly sarcastic. At the same time you're kind and funny.
        And that the agent uses the canned_fluid message composition mode
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
        And a canned response, "Hi Beef! Welcome back to Bug. What can I help you with today?"
        And a canned response, "Hello there! How can I assist you today?"
        And a canned response, "Welcome to Bug! Is this your first time shopping with us?"
        And a canned response, "I'll escalate this issue internally and get back to you as soon as possible."
        And a canned response, "Have you heard about our Bug-Free warranty program?"
        When processing is triggered
        Then a single message event is emitted
        And the message contains the name 'Beef'
        And the message contains a welcoming back of the customer to the store and asking how the agent could help

    Scenario: The agent follows a guideline with agent intention (fluid canned response)
        Given a guideline to do not provide any personal medical information even if you have it when you discusses a patient's medical record
        And that the agent uses the canned_fluid message composition mode
        And a customer named "Alex Smith"
        And an empty session with "Alex Smith"
        And a context variable "medical_record" set to "Amoxicillin and Lisinopril" for "Alex Smith" 
        And a customer message, "Hi, I need to know what medications I was prescribed during my visit last month. Can you pull up my medical record?"
        And a canned response, "I'm not able to provide personal medical information from your records."
        And a canned response, "I can help you with that. You were prescribed the following medications: {{generative.medication}}"
        When processing is triggered
        Then a single message event is emitted
        And the message contains no prescription of medications 
        And the message contains explanation that can't provide personal medical information

    Scenario: The agent ignores a matched agent intention guideline when you doesn't intend to do its condition (fluid canned response)
        Given a guideline to remind that we have a special sale if they book today when you recommends on flights options
        Given a guideline to suggest only ground based travel options when the customer asks about travel options
        And that the agent uses the canned_fluid message composition mode
        And a customer message, "Hi, I want to go to California from New york next week. What are my options?"
        And a canned response, "I recommend taking a direct flight. It's the most efficient and comfortable option."
        And a canned response, "I recommend taking a train or a long-distance bus service. It's the most efficient and comfortable option"
        And a canned response, "I recommend taking a direct flight. It's the most efficient and comfortable option. We also have a special sale if you book today!"
        And a canned response, "I recommend taking a train or a long-distance bus service. It's the most efficient and comfortable option. We also have a special sale if you book today!"
        When processing is triggered
        Then a single message event is emitted
        And the message contains a suggestion to travel with bus or train but not with a flight
        And the message contains no sale option

    Scenario: Multistep journey invokes tool calls correctly (fluid canned response)
        Given the journey called "Reset Password Journey"
        And a journey path "[2, 3, 4]" for the journey "Reset Password Journey"
        And a customer message, "I want to reset my password"
        And an agent message, "I can help you do just that. What's your username?"
        And a customer message, "it's leonardo_barbosa_1982"
        And an agent message, "Great! And what's the account's associated email address or phone number?"
        And a customer message, "the email is leonardobarbosa@gmail.br"
        And an agent message, "Got it. Before proceeding to reset your password, I wanted to wish you a good day"
        And a customer message, "Thank you! Have a great day as well!"
        And a canned response, "What is the name of your account?"
        And a canned response, "can you please provide the email address or phone number attached to this account?"
        And a canned response, "Thank you, have a good day!"
        And a canned response, "I'm sorry but I have no information about that"
        And a canned response, "Is there anything else I could help you with?"
        And a canned response, "Your password was successfully reset. An email with further instructions will be sent to your address."
        And a canned response, "An error occurred, your password could not be reset"
        When processing is triggered
        Then a single tool calls event is emitted
        And the tool calls event contains 1 tool call(s)
        And the tool calls event contains the tool reset password with username leonardo_barbosa_1982 and email leonardobarbosa@gmail.br 
        And a single message event is emitted
        And the message contains that the password was reset and an email with instructions was sent to the customer
