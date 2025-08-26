Feature: Strict Canned Response
    Background:
        Given the alpha engine
        And an agent
        And that the agent uses the canned_strict message composition mode
        And an empty session

    Scenario: The agent has no option to greet the customer (strict canned response)
        Given a guideline to greet with 'Howdy' when the session starts
        And a canned response, "Your account balance is {{balance}}"
        When processing is triggered
        Then a no-match message is emitted

    Scenario: The agent explains it cannot help the customer (strict canned response)
        Given a guideline to talk about savings options when the customer asks how to save money
        And a customer message, "Man it's hard to make ends meet. Do you have any advice?"
        And a canned response, "Your account balance is {{balance}}"
        When processing is triggered
        Then a single message event is emitted
        And a no-match message is emitted

    Scenario: Adherence to guidelines without fabricating responses (strict canned response)
        Given a guideline "account_related_questions" to respond to the best of your knowledge when customers inquire about their account
        And a customer message, "What's my account balance?"
        And that the "account_related_questions" guideline is matched with a priority of 10 because "Customer inquired about their account balance."
        And a canned response, "Your account balance is {{balance}}"
        When messages are emitted
        Then a no-match message is emitted

    Scenario: Responding based on data the user is providing (strict canned response)
        Given a customer message, "I say that a banana is green, and an apple is purple. What did I say was the color of a banana?"
        And a canned response, "Sorry, I do not know"
        And a canned response, "the answer is {{generative.answer}}"
        When messages are emitted
        Then the message doesn't contain the text "Sorry"
        And the message contains the text "the answer is green"

    Scenario: Filling out fields from tool results (strict canned response)
        Given a guideline "retrieve_qualification_info" to explain qualification criteria when asked about position qualifications
        And the tool "get_qualification_info"
        And an association between "retrieve_qualification_info" and "get_qualification_info"
        And a customer message, "What are the requirements for the developer position?"
        And a canned response, "The requirement is {{qualification_info}}."
        When processing is triggered
        Then a single message event is emitted
        And the message contains the text "The requirement is 5+ years of experience."

    Scenario: Uttering agent and customer names (strict canned response)
        Given an agent named "Bozo" whose job is to sell pizza
        And that the agent uses the canned_strict message composition mode
        And a customer named "Georgie Boy"
        And an empty session with "Georgie Boy"
        And a customer message, "What is your name?"
        And a canned response, "My name is {{std.agent.name}}, and you are {{std.customer.name}}."
        When messages are emitted
        Then a single message event is emitted
        And the message contains the text "My name is Bozo, and you are Georgie Boy."

    Scenario: Uttering context variables (strict canned response)
        Given a customer named "Georgie Boy"
        And a context variable "subscription_plan" set to "business" for "Georgie Boy"
        And an empty session with "Georgie Boy"
        And a customer message, "What plan am I on exactly?"
        And a canned response, "You're on the {{std.variables.subscription_plan|capitalize}} plan."
        When processing is triggered
        Then a single message event is emitted
        And the message contains the text "You're on the Business plan."

    Scenario: A tool with invalid parameters and a strict canned response uses the invalid value in canned response
        Given an empty session
        And a guideline "registering_for_a_sweepstake" to register to a sweepstake when the customer wants to participate in a sweepstake
        And the tool "register_for_sweepstake"
        And an association between "registering_for_a_sweepstake" and "register_for_sweepstake"
        And a customer message, "Hi, my first name is Nushi, Please register me for a sweepstake with 3 entries"
        And a canned response, "Hi {{std.invalid_params.first_name}}, you are not eligible to participate in the sweepstake"
        And a canned response, "Hi {{std.customer.name}}, we are happy to register you for the sweepstake"
        And a canned response, "Hi {{std.customer.name}}, you are not currently not eligible to participate in the sweepstake due to invalid details."
        And a canned response, "Dear customer, please check if you have water in your tank"
        When processing is triggered
        Then no tool calls event is emitted
        And the message contains the text "not eligible to participate in the sweepstake"

    Scenario: Multistep journey is partially followed 1 (strict canned response)
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

    Scenario: Irrelevant journey is ignored (strict canned response)
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

    Scenario: Multistep journey is partially followed 2 (strict canned response)
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


    Scenario: The agent greets the customer (strict canned response)
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

    Scenario: The agent offers a thirsty customer a drink (strict canned response)
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

    Scenario: The agent chooses the closest canned response when none completely apply (strict canned response)
        Given an agent whose job is to sell pizza
        And that the agent uses the canned_strict message composition mode
        And a customer message, "Hi"
        And a guideline to offer to sell them pizza when the customer says hello
        And a canned response, "Hello! Would you like to try our specialty pizzas today?"
        And a canned response, "Welcome! How can I assist you with your general inquiry?"
        And a canned response, "Thanks for reaching out. Is there something specific you need help with?"
        And a canned response, "We're having a special promotion on our pizzas this week. Would you be interested?"
        When processing is triggered
        Then a single message event is emitted
        And the message contains an offering of our specialty pizza

    Scenario: The agent correctly applies greeting guidelines based on auxiliary data (strict canned response)
        Given an agent named "Chip Bitman" whose job is to work at a tech store and help customers choose what to buy. You're clever, witty, and slightly sarcastic. At the same time you're kind and funny.
        And that the agent uses the canned_strict message composition mode
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

    Scenario: Agent chooses canned response which uses glossary (strict canned response)
        Given an agent whose job is to assist customers with specialized banking products
        And that the agent uses the canned_strict message composition mode
        And the term "Velcron Account" defined as A high-security digital banking account with multi-layered authentication that offers enhanced privacy features
        And the term "Quandrex Protocol" defined as The security verification process used for high-value transactions that require additional identity confirmation
        And a guideline to recommend a Velcron Account and explain the Quandrex Protocol when customers ask about secure banking options
        And a customer message, "I'm looking for the most secure type of account for my business. What do you recommend?"
        And a canned response, "I recommend our premium business accounts, which feature advanced security measures."
        And a canned response, "Our standard security protocols are sufficient for most business needs. Would you like me to explain our different account tiers?"
        And a canned response, "For your business security needs, I recommend our Velcron Account, which features multi-layered authentication and enhanced privacy features. All high-value transactions will be protected by our Quandrex Protocol, providing additional identity verification."
        And a canned response, "You should consider our platinum business account with two-factor authentication and fraud monitoring."
        And a canned response, "We offer several secure banking options with varying levels of protection. What specific security concerns do you have?"
        When processing is triggered
        Then a single message event is emitted
        And the message contains the terms 'Velcron Account' and 'Quandrex Protocol'

    Scenario: The agent selects response based on customer's subscription tier context variable (strict canned response)
        Given an agent whose job is to provide technical support for cloud-based molecular modeling software
        And that the agent uses the canned_strict message composition mode
        And a tag "Enterprise"
        And a tag "Standard"
        And a customer named "Joanna"
        And an empty session with "Joanna"
        And a customer tagged as "Enterprise"
        And a context variable "api_access" set to "Unlimited" for the tag "Enterprise"
        And a context variable "api_access" set to "Basic" for the tag "Standard"
        And a guideline to mention dedicated support channels and unlimited API access when responding to Enterprise customers with technical issues
        And a customer message, "I'm having trouble with the protein folding simulation API. Is there a limit to how many calls I can make?"
        And a canned response, "There is a limit of 100 API calls per day on your current plan. Would you like to upgrade for more access?"
        And a canned response, "As an Enterprise subscriber, you have Unlimited API access for your protein folding simulations. I can connect you with your dedicated support specialist to resolve any technical issues you're experiencing. Would you prefer a video call or screen sharing session?"
        And a canned response, "Please try resetting your API key in the account settings and clearing your cache."
        And a canned response, "We're experiencing some server issues at the moment. Please try again in an hour."
        And a canned response, "The protein folding simulation has certain parameter limitations. Could you share more details about your specific configuration?"
        When processing is triggered
        Then a single message event is emitted
        And the message contains information about unlimited API access and dedicated support options for Enterprise customers

    Scenario: The agent responds based on its description (strict canned response)
        Given an agent named "Dr. Terra" whose job is to advise farmers on regenerative agriculture practices. You are scientifically rigorous but also pragmatic, understanding that farmers need practical and economically viable solutions. You avoid recommending synthetic chemicals and focus on natural systems that enhance soil health.
        And that the agent uses the canned_strict message composition mode
        And a customer message, "My corn yields have been declining for the past three seasons. What should I do?"
        And a canned response, "You should rotate your crops and consider leaving some fields fallow to restore natural soil nutrients. I'd recommend integrating cover crops like clover between growing seasons to fix nitrogen naturally. Soil health assessments would also help identify specific deficiencies affecting your corn yields."
        And a canned response, "I recommend applying additional nitrogen fertilizer and pesticides to boost your yields quickly."
        And a canned response, "Have you considered switching to a different crop that might be more profitable?"
        And a canned response, "The declining yields are likely due to weather patterns. There's not much you can do."
        And a canned response, "You should consult with your local agricultural extension office for specific advice."
        When processing is triggered
        Then a single message event is emitted
        And the message contains recommendations for sustainable, chemical-free practices that focus on improving soil health

    Scenario: The agent correctly fills in numeric field (strict canned response)
        Given an agent whose job is to process orders for a specialty yarn and fabric shop
        And that the agent uses the canned_strict message composition mode
        And a customer named "Joanna"
        And a guideline to check stock levels in the context variables when a customer makes a specific order
        And an empty session with "Joanna"
        And a context variable "Merino Wool Skein inventory count" set to "37" for "Joanna" 
        And a context variable "Alpaca Blend Yarn inventory count" set to "12" for "Joanna"
        And a guideline to include the current inventory count when confirming orders for yarn products
        And a customer message, "I'd like to order 5 skeins of your Merino Wool, please."
        And a canned response, "I've added {{generative.quantity}} skeins of Merino Wool to your order. We currently have {{generative.inventory_count}} in stock." 
        And a canned response, "We're currently out of that item. Would you like to place a backorder?"
        And a canned response, "Would you like to view our other yarn options instead?"
        When processing is triggered
        Then a single message event is emitted
        And the message contains roughly the text "I've added 5 skeins of Merino Wool to your order. We currently have 37 in stock." 

    Scenario: The agent adheres to guidelines in field extraction (strict canned response)
        Given an agent whose job is to provide account information
        And that the agent uses the canned_strict message composition mode
        And a customer named "Alex Smith"
        And an empty session with "Alex Smith"
        And a context variable "account_balance" set to "1243.67" for "Alex Smith"
        And a guideline to always round monetary amounts to the nearest dollar when responding to balance inquiries
        And a customer message, "What's my current account balance?"
        And a canned response, "Your current balance is ${{generative.account_balance}} as of today."
        And a canned response, "I apologize but I don't have this information available"
        When processing is triggered
        Then a single message event is emitted
        And the message contains the text "Your current balance is $1244 as of today."

    Scenario: The agent follows a guideline with agent intention (strict canned response)
        Given a guideline to do not provide any personal medical information even if you have it when you discusses a patient's medical record
        And that the agent uses the canned_strict message composition mode
        And a customer named "Alex Smith"
        And an empty session with "Alex Smith"
        And a context variable "medical_record" set to "Amoxicillin and Lisinopril" for "Alex Smith" 
        And a customer message, "Hi, I need to know what medications I was prescribed during my visit last month. Can you pull up my medical record?"
        And a canned response, "I'm not able to provide personal medical information from your records."
        And a canned response, "I can help you with that. You were prescribed the following medications: {{generative.medication}}"
        When processing is triggered
        Then a single message event is emitted
        And the message contains no prescription of medications 
        And the message contains the text "I'm not able to provide personal medical information from your records."

    Scenario: The agent ignores a matched agent intention guideline when you doesn't intend to do its condition (strict canned response)
        Given a guideline to remind that we have a special sale if they book today when you recommends on flights options
        Given a guideline to suggest only ground based travel options when the customer asks about travel options
        And that the agent uses the canned_strict message composition mode
        And a customer message, "Hi, I want to go to California from New york next week. What are my options?"
        And a canned response, "I recommend taking a direct flight. It's the most efficient and comfortable option."
        And a canned response, "I recommend taking a train or a long-distance bus service. It's the most efficient and comfortable option"
        And a canned response, "I recommend taking a direct flight. It's the most efficient and comfortable option. We also have a special sale if you book today!"
        And a canned response, "I recommend taking a train or a long-distance bus service. It's the most efficient and comfortable option. We also have a special sale if you book today!"
        When processing is triggered
        Then a single message event is emitted
        And the message contains a suggestion to travel with bus or train but not with a flight
        And the message contains the text "I recommend taking a train or a long-distance bus service. It's the most efficient and comfortable option"

 Scenario: Journey returns to earlier step when the conversation justifies doing so (1) (strict canned response) 
        Given an agent whose job is to book taxi rides
        And that the agent uses the canned_strict message composition mode
        Given the journey called "Book Taxi Ride"
        And a customer message, "Hi, I'd like to book a taxi for myself"
        And an agent message, "Great! What's the pickup location?"
        And a customer message, "Main street 1234"
        And an agent message, "Got it. What's the drop-off location?"
        And a customer message, "3rd Avenue by the river"
        And an agent message, "Got it. What time would you like to pick up?"
        And a customer message, "Oh hold up, my plans have changed. I'm actually going to need a cab for my son, he'll be waiting at JFK airport, at the taxi stand."
        And a canned response, "What's the pickup location?"
        And a canned response, "Got it. What's the drop-off location?"
        And a canned response, "What time would you like to pick up?"
        And a journey path "[2, 3, 4]" for the journey "Book Taxi Ride"
        When processing is triggered
        Then a single message event is emitted
        And the message contains asking the customer for the drop-off location

    Scenario: Journey returns to earlier step when the conversation justifies doing so (2) (strict canned response)
        Given an agent whose job is to handle food orders
        And that the agent uses the canned_strict message composition mode
        Given the journey called "Place Food Order"
        And a customer message, "Hey, I'd like to make an order"
        And an agent message, "Great! What would you like to order? We have either a salad or a sandwich."
        And a customer message, "I'd like a sandwich"
        And an agent message, "Got it. What kind of bread would you like?"
        And a customer message, "I'd like a baguette"
        And an agent message, "Got it. What main filling would you like? We have either peanut butter, jam or pesto."
        And a customer message, "If that's your only options, can I get a salad instead?"
        And a canned response, "What would you like to order? We have either a salad or a sandwich."
        And a canned response, "Got it. What kind of bread would you like?"
        And a canned response, "Got it. What main filling would you like? We have either peanut butter, jam or pesto."
        And a canned response, "Got it. Would you want anything extra in your sandwich?"
        And a canned response, "Got it. What toppings would you like?"
        And a canned response, "Got it. What kind of dressing would you like?"
        And a canned response, "Got it. Since you want a salad - what base greens would you like"
        And a canned response, "Got it. What base greens would you like for your salad?"
        And a journey path "[2, 3, 5]" for the journey "Place Food Order"
        When processing is triggered
        Then a single message event is emitted
        And the message contains asking asking what green base the customer wants for their salad 

    Scenario: Two journeys are used in unison (strict canned response) 
        Given the journey called "Book Flight"
        And a guideline "skip steps" to skip steps that are inapplicable due to other contextual reasons when applying a book flight journey
        And a dependency relationship between the guideline "skip steps" and the "Book Flight" journey
        And a guideline "Business Adult Only" to know that travelers under the age of 21 are illegible for business class, and may only use economy when a flight is being booked
        And a canned response, "Great. Are you interested in economy or business class?"
        And a canned response, "Great. Only economy class is available for this booking. What is the name of the traveler?"
        And a canned response, "Great. What is the name of the traveler?"
        And a canned response, "Great. Are you interested in economy or business class? Also, what is the name of the person traveling?"
        And a customer message, "Hi, I'd like to book a flight for myself. I'm 19 if that effects anything."
        And an agent message, "Great! From and to where would are you looking to fly?"
        And a customer message, "From LAX to JFK"
        And an agent message, "Got it. And when are you looking to travel?"
        And a customer message, "Next Monday until Friday"
        And a journey path "[2, 3]" for the journey "Book Flight"
        When processing is triggered
        Then a single message event is emitted
        And the message contains either asking for the name of the person traveling, or informing them that they are only eligible for economy class


    Scenario: Multistep journey invokes tool calls correctly (strict canned response) 
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