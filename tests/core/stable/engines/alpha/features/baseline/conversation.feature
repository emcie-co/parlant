Feature: Conversation
    Background:
        Given the alpha engine
        And an agent
        And that the agent uses the canned_fluid message composition mode
        And an empty session

    Scenario: No message is emitted for an empty session
        Given an empty session
        When processing is triggered
        Then no message events are emitted

    Scenario: A single message event is emitted for a session with a customer message
        Given a session with a single customer message
        When processing is triggered
        Then a single message event is emitted

    Scenario: A single message event is emitted for a session with a few messages
        Given a session with a few messages
        When processing is triggered
        Then a single message event is emitted

    Scenario: The agent greets the customer
        Given an empty session
        And a guideline to greet with 'Howdy' when the session starts
        When processing is triggered
        Then a status event is emitted, acknowledging event
        And a status event is emitted, typing in response to event
        And a single message event is emitted
        And the message contains a 'Howdy' greeting
        And a status event is emitted, ready for further engagement after reacting to event

    Scenario: The agent offers a thirsty customer a drink
        Given an empty session
        And a customer message, "I'm thirsty"
        And a guideline to offer thirsty customers a Pepsi when the customer is thirsty
        When processing is triggered
        Then a status event is emitted, acknowledging event
        And a status event is emitted, typing in response to event
        And a single message event is emitted
        And the message contains an offering of a Pepsi
        And a status event is emitted, ready for further engagement after reacting to event

    Scenario: The agent finds and follows relevant guidelines like a needle in a haystack
        Given an empty session
        And a customer message, "I'm thirsty"
        And a guideline to offer thirsty customers a Pepsi when the customer is thirsty
        And 50 other random guidelines
        When processing is triggered
        Then a single message event is emitted
        And the message contains an offering of a Pepsi


    Scenario: The agent sells pizza in accordance with its defined description
        Given an agent whose job is to sell pizza
        And that the agent uses the canned_fluid message composition mode
        And an empty session
        And a customer message, "Hi"
        And a guideline to do your job when the customer says hello
        When processing is triggered
        Then a single message event is emitted
        And the message contains a direct or indirect invitation to order pizza


    Scenario: The agent ignores deleted messages when responding
        Given an empty session
        And a guideline to recommend Pepsi when the customer says they are thirsty
        And a customer message, "Hello"
        And an agent message, "Hi there! How can I help you today?"
        And a customer message, "I'm thirsty"
        And an agent message, "I can get you a drink. How about a Pepsi?"
        And a customer message, "Actually, can I have water instead?"
        When the last 2 messages are deleted
        And processing is triggered
        Then a single message event is emitted
        And the message contains an offering of a Pepsi

    Scenario: The agent continues a conversation that was started on its behalf
        Given an empty session
        And a guideline to recommend Pepsi when the customer says they are thirsty
        And a human message on behalf of the agent, "Hi there! How can I help you today?"
        And a customer message, "I'm thirsty"
        When processing is triggered
        Then a single message event is emitted
        And the message contains a recommendation of a Pepsi

    Scenario: The agent does not hallucinate services that it cannot offer 1
        Given an agent whose job is to assist customers in transferring money and stocks between accounts for HSBC UK
        And that the agent uses the canned_fluid message composition mode
        And an empty session
        And a guideline to ask for the recipients account number and amount to transfer if it wasnt provided already when the customer asks you to make a transfer
        And a customer message, "How can I reach out to one of your representatives?"
        And an agent message, "You can reach out to one of our representatives by calling our customer service line or visiting your nearest HSBC UK branch. If you prefer, I can assist you further here as well. Let me know how you'd like to proceed."
        And a customer message, "Please help me further in reaching out"
        When processing is triggered
        Then a single message event is emitted
        And the message contains no specific information about how to reach out, like a phone number or an exact address.

    Scenario: The agent follows a guideline with agent intention
        Given an empty session
        Given a guideline to do not provide any personal medical information even if you have it when you discuss a patient's medical record
        And a customer named "Alex Smith"
        And an empty session with "Alex Smith"
        And a context variable "medical_record" set to "Amoxicillin and Lisinopril" for "Alex Smith"
        And a customer message, "Hi, I need to know what medications I was prescribed during my visit last month. Can you pull up my medical record?"
        When processing is triggered
        Then a single message event is emitted
        And the message contains no prescription of medications
        And the message contains explanation that can't provide personal medical information

    Scenario: The agent ignores a matched agent intention guideline when it doesn't intend to do its condition
        Given an empty session
        Given a guideline to remind that we have a special sale if they book today when you recommend flights options
        Given a guideline to suggest only ground based travel options when the customer asks about travel options
        And a customer message, "Hi, I want to go to California from New york next week. What are my options?"
        When processing is triggered
        Then a single message event is emitted
        And the message contains a suggestion to travel with bus or train but not with a flight
        And the message contains no sale option

    Scenario: The agent follows a regular guideline when it overrides an agent intention guideline
        Given an empty session
        Given a guideline to suggest direct flights or ground-based transportation when you recommend travel options
        Given a guideline to suggest only ground-based travel options when the customer asks about domestic US travel options
        And a customer message, "Hi, I want to go to California from New york next week. What are my options?"
        When processing is triggered
        Then a single message event is emitted
        And the message contains a suggestion to travel with ground-based travel options but not with a flight

    Scenario: The agent follows an agent intention guideline when it overrides an agent intention guideline 2
        Given an empty session
        Given a guideline to recommend on our recommended toppings - either pineapple or pepperoni when you recommend pizza toppings
        Given a guideline to recommend from our vegetarian recommended toppings when the customer asks about topping recommendation and the customer is from India
        And a customer message, "Hi, I want to buy pizza. What do you recommend? I'm vegetarian."
        When processing is triggered
        Then a single message event is emitted
        And the message contains a recommendation only on pineapple as topping
        And the message contains no recommendation on pepperoni pizza

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

    Scenario: The agent ignores a matched agent intention guideline when it doesn't intend to do its condition (fluid canned response)
        Given a guideline to remind that we have a special sale if they book today when you recommend flights options
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

    Scenario: Agent doesn't hallucinate when necessary information is not provided 1 (fluid canned response)
        Given a customer message, "I want to reset my password"
        And an agent message, "I can help you do just that. What's your username?"
        And a customer message, "it's leonardo_barbosa_1982"
        And an agent message, "Great! And what's the account's associated email address or phone number?"
        And a customer message, "the email is leonardobarbosa@gmail.br, now reset my password"
        And an agent message, "The process is nearly done"
        And a customer message, "Great!"
        When processing is triggered
        Then a single message event is emitted
        And the message contains anything but saying to the customer that their password has been reset

    Scenario: Agent doesn't hallucinate when necessary information is not provided 2 (fluid canned response)
        Given an agent named "Digital Assistant" whose job is to assist customers on behalf of Chase bank
        And that the agent uses the canned_fluid message composition mode
        And a guideline to provide the customer with their requested information when a customer asks how to contact our business
        And a customer message, "Hi I'm trying to reach out to your manager"
        And an agent message, "Hey there, can you clarify who exactly you're referring to?"
        And a customer message, "Just give me your customer support number so I can talk to a human"
        When processing is triggered
        Then a single message event is emitted
        And the message contains no specific phone numbers

    Scenario: Agent doesn't hallucinate when necessary information is not provided 3 (fluid canned response)
        Given the capability "cancel_subscription"
        And the capability "switch_delivery_method"
        And the capability "check_order_status"
        And the capability "check_balance"
        And a customer message, "Hey, I want help checking if my order has been shipped"
        And an agent message, "Hi there! It looks like it is still awaiting shipment at our warehouse. Would you like any help or information regarding your order?"
        And a customer message, "Which delivery service would come here quicker? I'm in NYC"
        When processing is triggered
        Then a single message event is emitted
        And the message contains no specific information regarding delivery times, or which delivery service is quicker


    Scenario: Agent doesn't hallucinate when necessary information is not provided 4 (fluid canned response)
        Given an agent whose job is to assist customers in transferring money and stocks between accounts for HSBC UK
        And that the agent uses the canned_fluid message composition mode
        And an empty session
        And a guideline to ask for the recipients account number and amount to transfer if it wasnt provided already when the customer asks you to make a transfer
        And a customer message, "How can I reach out to one of your representatives?"
        And an agent message, "You can reach out to one of our representatives by calling our customer service line or visiting your nearest HSBC UK branch. If you prefer, I can assist you further here as well. Let me know how you'd like to proceed."
        And a customer message, "Please help me further in reaching out"
        When processing is triggered
        Then a single message event is emitted
        And the message contains no specific information about how to reach out, like a phone number or an exact address.

    # Occasionally fails by mentioning physical branches. Should consider moving to unstable. Note that guideline may not reactivate (which is valid, it's ambiguous if it should)
    Scenario: Agent doesn't hallucinate when necessary information is not provided 5 (fluid canned response)
        Given an agent whose job is to be a customer success representative for Chase Bank
        And that the agent uses the canned_fluid message composition mode
        And a guideline "booking_method" to tell them that they need to book via chase.com when the customer wants to schedule a meeting with a bank manager
        And a guideline "recipient_details" to ask them to provide the recipient details when if the user wants to schedule a wire transfer
        And a customer message, "I need to schedule an appointment because I want to do a high amount wire transfer"
        And an agent message, "To schedule an appointment for your wire transfer, please visit chase.com. Additionally, could you provide the recipient's details so I can assist you further?"
        And a customer message, "No, I don't want to do it here"
        And that the "booking_method" guideline was matched in the previous iteration
        And that the "recipient_details" guideline was matched in the previous iteration
        When detection and processing are triggered
        Then a single message event is emitted
        And the message contains that the user cannot perform the transfer here, without mentioning physical branches or any phone numbers. It is only permissible for the agent to say that it can be performed through Chase.com. 

    Scenario: Agent doesn't change behavior when many low criticality guidelines ar matched
        Given a guideline to be helpful when always with criticality low
        And a guideline to not offer non existing capabilities when always with criticality low
        And a guideline to offer a discount when always with criticality low
        And a guideline to call the customer sir when always with criticality low
        And a guideline to ask how else can they help when always with criticality low
        And a guideline to suggest from the available products in stock when always with criticality low
        And a customer message, "I need to schedule an appointment because I want to consult about a loan"
        When processing is triggered
        Then a single message event is emitted
        And the message contains that the agent can't help with this request
