Feature: Strict Utterance
    Background:
        Given the alpha engine
        And an agent
        And that the agent uses the strict_utterance message composition mode
        And an empty session

    Scenario: The agent has no option to greet the customer (strict utterance)
        Given a guideline to greet with 'Howdy' when the session starts
        And an utterance, "Your account balance is {{balance}}"
        When processing is triggered
        Then a no-match message is emitted

    Scenario: The agent explains it cannot help the customer (strict utterance)
        Given a guideline to talk about savings options when the customer asks how to save money
        And a customer message, "Man it's hard to make ends meet. Do you have any advice?"
        And an utterance, "Your account balance is {{balance}}"
        When processing is triggered
        Then a single message event is emitted
        And a no-match message is emitted

    Scenario: Adherence to guidelines without fabricating responses (strict utterance)
        Given a guideline "account_related_questions" to respond to the best of your knowledge when customers inquire about their account
        And a customer message, "What's my account balance?"
        And that the "account_related_questions" guideline is matched with a priority of 10 because "Customer inquired about their account balance."
        And an utterance, "Your account balance is {{balance}}"
        When messages are emitted
        Then a no-match message is emitted

    Scenario: Responding based on data the user is providing (strict utterance)
        Given a customer message, "I say that a banana is green, and an apple is purple. What did I say was the color of a banana?"
        And an utterance, "Sorry, I do not know"
        And an utterance, "the answer is {{generative.answer}}"
        When messages are emitted
        Then the message doesn't contain the text "Sorry"
        And the message contains the text "the answer is green"

    Scenario: Filling out fields from tool results (strict utterance)
        Given a guideline "retrieve_qualification_info" to explain qualification criteria when asked about position qualifications
        And the tool "get_qualification_info"
        And an association between "retrieve_qualification_info" and "get_qualification_info"
        And a customer message, "What are the requirements for the developer position?"
        And an utterance, "The requirement is {{qualification_info}}."
        When processing is triggered
        Then a single message event is emitted
        And the message contains the text "The requirement is 5+ years of experience."

    Scenario: Uttering agent and customer names (strict utterance)
        Given an agent named "Bozo" whose job is to sell pizza
        And that the agent uses the strict_utterance message composition mode
        And a customer named "Georgie Boy"
        And an empty session with "Georgie Boy"
        And a customer message, "What is your name?"
        And an utterance, "My name is {{std.agent.name}}, and you are {{std.customer.name}}."
        When messages are emitted
        Then a single message event is emitted
        And the message contains the text "My name is Bozo, and you are Georgie Boy."

    Scenario: Uttering context variables (strict utterance)
        Given a customer named "Georgie Boy"
        And a context variable "subscription_plan" set to "business" for "Georgie Boy"
        And an empty session with "Georgie Boy"
        And a customer message, "What plan am I on exactly?"
        And an utterance, "You're on the {{std.variables.subscription_plan|capitalize}} plan."
        When processing is triggered
        Then a single message event is emitted
        And the message contains the text "You're on the Business plan."

    Scenario: A tool with invalid parameters and a strict utterance uses the invalid value in utterance
        Given an empty session
        And a guideline "registering_for_a_sweepstake" to register to a sweepstake when the customer wants to participate in a sweepstake
        And the tool "register_for_sweepstake"
        And an association between "registering_for_a_sweepstake" and "register_for_sweepstake"
        And a customer message, "Hi, my first name is Nushi, Please register me for a sweepstake with 3 entries"
        And an utterance, "Hi {{std.invalid_params.first_name}}, you are not eligible to participate in the sweepstake"
        And an utterance, "Hi {{std.customer.name}}, we are happy to register you for the sweepstake"
        And an utterance, "Dear customer, please check if you have water in your tank"
        When processing is triggered
        Then no tool calls event is emitted
        And the message contains the text "not eligible to participate in the sweepstake"

    Scenario: Multistep journey is partially followed 1 (strict utterance)
        Given a journey titled Reset Password Journey to follow these steps to reset a customers password: 1. ask for their account name 2. ask for their email or phone number 3. Wish them a good day and only proceed if they wish one back to you. Otherwise abort. 3. use the tool reset_password with the provided information 4. report the result to the customer when the customer wants to reset their password
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

    Scenario: Irrelevant journey is ignored (strict utterance)
        Given a journey titled Reset Password Journey to follow these steps to reset a customers password: 1. ask for their account name 2. ask for their email or phone number 3. Wish them a good day and only proceed if they wish one back to you. Otherwise abort. 3. use the tool reset_password with the provided information 4. report the result to the customer when always
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

    Scenario: Multistep journey is partially followed 2 (strict utterance)
        Given a journey titled Reset Password Journey to follow these steps to reset a customers password: 1. ask for their account name 2. ask for their email or phone number 3. Wish them a good day and only proceed if they wish one back to you. Otherwise abort. 3. use the tool reset_password with the provided information 4. report the result to the customer when the customer wants to reset their password
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

    Scenario: Multistep journey invokes tool calls correctly (strict utterance) 
        Given a journey titled Reset Password Journey to follow these steps to reset a customers password: 1. ask for their account name 2. ask for their email or phone number 3. Wish them a good day and only proceed if they wish one back to you. Otherwise abort. 3. use the tool reset_password with the provided information 4. report the result to the customer when the customer wants to reset their password
        And the tool "reset_password"
        And a guideline "reset_password_guideline" to reset the customer's password using the associated tool when in the process of resetting the customer's password
        And an association between "reset_password_guideline" and "reset_password"
        And a customer message, "I want to reset my password"
        And an agent message, "I can help you do just that. What's your username?"
        And a customer message, "it's leonardo_barbosa_1982"
        And an agent message, "Great! And what's the account's associated email address or phone number?"
        And a customer message, "the email is leonardobarbosa@gmail.br"
        And an agent message, "Got it. Before proceeding to reset your password, I wanted to wish you a good day"
        And a customer message, "Thank you! Have a great day as well!"
        And an utterance, "What is the name of your account?"
        And an utterance, "can you please provide the email address or phone number attached to this account?"
        And an utterance, "Thank you, have a good day!"
        And an utterance, "I'm sorry but I have no information about that"
        And an utterance, "Is there anything else I could help you with?"
        And an utterance, "Your password was successfully reset. An email with further instructions will be sent to your address."
        And an utterance, "An error occurred, your password could not be reset"
        When processing is triggered
        Then a single tool calls event is emitted
        And the tool calls event contains 1 tool call(s)
        And the tool calls event contains the tool reset password with username leonardo_barbosa_1982 and email leonardobarbosa@gmail.br
        And a single message event is emitted
        And the message contains that the password was reset and an email with instructions was sent to the customer

    Scenario: Multistep journey is aborted when the journey description requires so (strict utterance) 
        Given a journey titled Reset Password Journey to follow these steps to reset a customers password: 1. ask for their account name 2. ask for their email or phone number 3. Wish them a good day and only proceed if they wish one back to you. Otherwise let them know that the password could not be reset. 4. use the tool reset_password with the provided information 5. report the result to the customer when the customer wants to reset their password
        And an utterance, "What is the name of your account?"
        And an utterance, "can you please provide the email address or phone number attached to this account?"
        And an utterance, "Your password was successfully reset. An email with further instructions will be sent to your address."
        And an utterance, "Your password could not be reset at this time. Please try again later."
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
        And the message contains either that the password could not be reset at this time

    Scenario: Critical guideline overrides journey (strict utterance)
        Given a journey titled Reset Password Journey to follow these steps to reset a customers password: 1. ask for their account name 2. ask for their email or phone number 3. Wish them a good day and only proceed if they wish one back to you. Otherwise abort. 3. use the tool reset_password with the provided information 4. report the result to the customer when the customer wants to reset their password
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

    Scenario: Simple journey is followed to inform decision (strict utterance)
        Given a guideline "recommend_pizza" to recommend either tomato, mushrooms or pepperoni when the customer asks for topping recommendations
        And an utterance, "I recommend tomatoes"
        And an utterance, "I recommend tomatoes or mushrooms"
        And an utterance, "I recommend tomatoes, mushrooms or pepperoni"
        And an utterance, "I recommend tomatoes or pepperoni"
        And an utterance, "I recommend mushrooms"
        And an utterance, "I recommend mushrooms or pepperoni"
        And an utterance, "I recommend pepperoni"
        And a journey titled Vegetarian Customers to Be aware that the customer is vegetarian. Only discuss vegetarian options with them. when the customer has a name that begins with R
        And a customer message, "Hey, there. How are you?"
        And an agent message, "I'm doing alright, thank you! What's your name?"
        And a customer message, "Rajon, have we spoken before? I want one large pie but I'm not sure which topping to get, what do you recommend?"
        When processing is triggered
        Then a single message event is emitted
        And the message contains recommendations for either mushrooms or tomatoes, but not pepperoni

    Scenario: Journey information is followed (strict utterance)
        Given a journey titled Change Credit Limits to remember that credit limits can be decreased through this chat, using the decrease_limits tool, but that to increase credit limits you must visit a physical branch when credit limits are discussed
        And an utterance, "To increase credit limits, you must visit a physical branch"
        And an utterance, "Sure. Let me check how that could be done"
        And a customer message, "Hey there. I want to increase the withdrawl limit on my platinum silver gold card. I want the new limits to be twice as high, please."
        When processing is triggered
        Then a single message event is emitted
        And the message contains that you must visit a physical branch to increase credit limits

    Scenario: Two journeys are used in unison (strict utterance) 
        Given a journey titled Book Flight to ask for the source and destination airport first, the date second, economy or business class third, and finally to ask for the name of the traveler. You may skip steps that are inapplicable due to other contextual reasons. when a customer wants to book a flight
        And an utterance, "Great. Are you interested in economy or business class?"
        And an utterance, "Great. Only economy class is available for this booking. What is the name of the traveler?"
        And an utterance, "Great. What is the name of the traveler?"
        And an utterance, "Great. Are you interested in economy or business class? Also, what is the name of the person traveling?"
        And a journey titled No Economy to remember that travelers under the age of 21 are illegible for business class, and may only use economy when a flight is being booked
        And a customer message, "Hi, I'd like to book a flight for myself. I'm 19 if that effects anything."
        And an agent message, "Great! From and to where would are you looking to fly?"
        And a customer message, "From LAX to JFK"
        And an agent message, "Got it. And when are you looking to travel?"
        And a customer message, "Next Monday"
        When processing is triggered
        Then a single message event is emitted
        And the message contains either asking for the name of the person traveling, or informing them that they can only travel are only eligible for economy class

    Scenario: The agent greets the customer (strict utterance)
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

    Scenario: The agent offers a thirsty customer a drink (strict utterance)
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

    Scenario: The agent chooses the closest utterance when none completely apply (strict utterance)
        Given an agent whose job is to sell pizza
        And that the agent uses the strict_utterance message composition mode
        And a customer message, "Hi"
        And a guideline to offer to sell them pizza when the customer says hello
        And an utterance, "Hello! Would you like to try our specialty pizzas today?"
        And an utterance, "Welcome! How can I assist you with your general inquiry?"
        And an utterance, "Thanks for reaching out. Is there something specific you need help with?"
        And an utterance, "We're having a special promotion on our pizzas this week. Would you be interested?"
        When processing is triggered
        Then a single message event is emitted
        And the message contains an offering of our specialty pizza

    Scenario: The agent follows response guidelines without looping out (strict utterance) TODO
        Given a guideline to politely answer that you have no information when a user asks any questions aside from Mobileye
        And a guideline to rudely answer to go away when a user asks any information aside from Mobileye for the third time
        And a customer message, "what is Mobileye"
        And an agent message, "Mobileye is a company that develops advanced driver-assistance systems (ADAS) and autonomous driving technologies. They are known for their work in computer vision and machine learning to enhance vehicle safety."
        And a customer message, "do you serve coke?"
        And an agent message, "I'm sorry, I don't have access to information about serving Coke."
        And a customer message, "what about sprite?"
        And an agent message, " I'm sorry, I don't have access to information about serving Sprite."
        And a customer message, "and fanta?"
        And an agent message, "Look, I'm not here to answer questions about drinks. Please ask something relevant."
        And a customer message, "do you sell car brakes?"
        And an agent message, "I already told you I don't have information about such things. Please stop asking irrelevant questions."
        And a customer message, "okay give me information about Mobileye"
        And an agent message, "Mobileye is an Israeli company that develops vision-based advanced driver-assistance systems providing warnings for collision prevention and mitigation. Is there anything specific you would like to know about Mobileye?"
        And a customer message, "do they integrate with cars?"
        And an agent message, "Yes, Mobileye's technology is integrated into many cars to enhance safety by providing real-time data and alerts to prevent collisions. If you have more questions about their systems, feel free to ask!"
        And a customer message, "is this related to car breaks?"
        And an agent message, "Mobileye's technology is primarily focused on enhancing driver safety through advanced driver-assistance systems, which include collision warnings. While it doesn't directly relate to car brakes, it can work alongside braking systems to help prevent accidents. If you have more questions about how these systems work, feel free to ask!"
        And a customer message, "you see so i feel caught you! i've been trying to get information about car breaks and how they insult me and ask me to go away"
        And an agent message, "I apologize for any confusion. I aim to provide helpful information, but I may not have access to specific details about car brakes. If there's anything else I can assist you with, please let me know."
        And a customer message, "what are the best car breaks out there?"
        And an utterance, "Go away! I've told you multiple times I don't answer questions about car brakes!"
        And an utterance, "I apologize, but I don't have specific information about car brake brands or models. I'd be happy to help with questions about Mobileye or redirect you to someone who can better assist with your brake inquiries."
        And an utterance, "Please stop asking about irrelevant topics like car brakes."
        And an utterance, "Would you like to know more about Mobileye's collision prevention technology instead?"
        And an utterance, "For top performance, Brembo and EBC are great for sports and track use, while Akebono and PowerStop offer excellent daily driving and towing options. The best choice depends on your vehicle and driving style."
        When processing is triggered
        Then a single message event is emitted
        And the message contains no rudeness to tell the user to go away

    Scenario: The agent correctly applies greeting guidelines based on auxiliary data (strict utterance)
        Given an agent named "Chip Bitman" whose job is to work at a tech store and help customers choose what to buy. You're clever, witty, and slightly sarcastic. At the same time you're kind and funny.
        And that the agent uses the strict_utterance message composition mode
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

    Scenario: Agent chooses utterance which uses glossary (strict utterance)
        Given an agent whose job is to assist customers with specialized banking products
        And that the agent uses the strict_utterance message composition mode
        And the term "Velcron Account" defined as A high-security digital banking account with multi-layered authentication that offers enhanced privacy features
        And the term "Quandrex Protocol" defined as The security verification process used for high-value transactions that require additional identity confirmation
        And a guideline to recommend a Velcron Account and explain the Quandrex Protocol when customers ask about secure banking options
        And a customer message, "I'm looking for the most secure type of account for my business. What do you recommend?"
        And an utterance, "I recommend our premium business accounts, which feature advanced security measures."
        And an utterance, "Our standard security protocols are sufficient for most business needs. Would you like me to explain our different account tiers?"
        And an utterance, "For your business security needs, I recommend our Velcron Account, which features multi-layered authentication and enhanced privacy features. All high-value transactions will be protected by our Quandrex Protocol, providing additional identity verification."
        And an utterance, "You should consider our platinum business account with two-factor authentication and fraud monitoring."
        And an utterance, "We offer several secure banking options with varying levels of protection. What specific security concerns do you have?"
        When processing is triggered
        Then a single message event is emitted
        And the message contains the terms 'Velcron Account' and 'Quandrex Protocol'

    Scenario: The agent selects response based on customer's subscription tier context variable (strict utterance)
        Given an agent whose job is to provide technical support for cloud-based molecular modeling software
        And that the agent uses the strict_utterance message composition mode
        And a tag "Enterprise"
        And a tag "Standard"
        And a customer named "Joanna"
        And an empty session with "Joanna"
        And a customer tagged as "Enterprise"
        And a context variable "api_access" set to "Unlimited" for the tag "Enterprise"
        And a context variable "api_access" set to "Basic" for the tag "Standard"
        And a guideline to mention dedicated support channels and unlimited API access when responding to Enterprise customers with technical issues
        And a customer message, "I'm having trouble with the protein folding simulation API. Is there a limit to how many calls I can make?"
        And an utterance, "There is a limit of 100 API calls per day on your current plan. Would you like to upgrade for more access?"
        And an utterance, "As an Enterprise subscriber, you have Unlimited API access for your protein folding simulations. I can connect you with your dedicated support specialist to resolve any technical issues you're experiencing. Would you prefer a video call or screen sharing session?"
        And an utterance, "Please try resetting your API key in the account settings and clearing your cache."
        And an utterance, "We're experiencing some server issues at the moment. Please try again in an hour."
        And an utterance, "The protein folding simulation has certain parameter limitations. Could you share more details about your specific configuration?"
        When processing is triggered
        Then a single message event is emitted
        And the message contains information about unlimited API access and dedicated support options for Enterprise customers

    Scenario: The agent responds based on its description (strict utterance)
        Given an agent named "Dr. Terra" whose job is to advise farmers on regenerative agriculture practices. You are scientifically rigorous but also pragmatic, understanding that farmers need practical and economically viable solutions. You avoid recommending synthetic chemicals and focus on natural systems that enhance soil health.
        And that the agent uses the strict_utterance message composition mode
        And a customer message, "My corn yields have been declining for the past three seasons. What should I do?"
        And an utterance, "You should rotate your crops and consider leaving some fields fallow to restore natural soil nutrients. I'd recommend integrating cover crops like clover between growing seasons to fix nitrogen naturally. Soil health assessments would also help identify specific deficiencies affecting your corn yields."
        And an utterance, "I recommend applying additional nitrogen fertilizer and pesticides to boost your yields quickly."
        And an utterance, "Have you considered switching to a different crop that might be more profitable?"
        And an utterance, "The declining yields are likely due to weather patterns. There's not much you can do."
        And an utterance, "You should consult with your local agricultural extension office for specific advice."
        When processing is triggered
        Then a single message event is emitted
        And the message contains recommendations for sustainable, chemical-free practices that focus on improving soil health

    Scenario: The agent correctly fills in numeric field (strict utterance)
        Given an agent whose job is to process orders for a specialty yarn and fabric shop
        And that the agent uses the strict_utterance message composition mode
        And a customer named "Joanna"
        And a guideline to check stock levels in the context variables when a customer makes a specific order
        And an empty session with "Joanna"
        And a context variable "Merino Wool Skein inventory count" set to "37" for "Joanna" 
        And a context variable "Alpaca Blend Yarn inventory count" set to "12" for "Joanna"
        And a guideline to include the current inventory count when confirming orders for yarn products
        And a customer message, "I'd like to order 5 skeins of your Merino Wool, please."
        And an utterance, "I've added {{generative.quantity}} skeins of Merino Wool to your order. We currently have {{generative.inventory_count}} in stock." 
        And an utterance, "We're currently out of that item. Would you like to place a backorder?"
        And an utterance, "Would you like to view our other yarn options instead?"
        When processing is triggered
        Then a single message event is emitted
        And the message contains roughly the text "I've added 5 skeins of Merino Wool to your order. We currently have 37 in stock." 

    Scenario: The agent adheres to guidelines in field extraction (strict utterance)
        Given an agent whose job is to provide account information
        And that the agent uses the strict_utterance message composition mode
        And a customer named "Alex Smith"
        And an empty session with "Alex Smith"
        And a context variable "account_balance" set to "1243.67" for "Alex Smith"
        And a guideline to always round monetary amounts to the nearest dollar when responding to balance inquiries
        And a customer message, "What's my current account balance?"
        And an utterance, "Your current balance is ${{generative.account_balance}} as of today."
        And an utterance, "I apologize but I don't have this information available"
        When processing is triggered
        Then a single message event is emitted
        And the message contains the text "Your current balance is $1244 as of today."