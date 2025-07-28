Feature: Fluid Utterance
    Background:
        Given the alpha engine
        And an agent
        And that the agent uses the fluid_utterance message composition mode
        And an empty session

    Scenario: Multistep journey is aborted when the journey description requires so (fluid utterance)
        Given a journey titled "Reset Password Journey" to follow these steps to reset a customers password: 1. ask for their account name 2. ask for their email or phone number 3. Wish them a good day and only proceed if they wish one back to you. Otherwise let them know that the password could not be reset. 4. use the tool reset_password with the provided information 5. report the result to the customer when the customer wants to reset their password
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
        When processing is triggered
        Then no tool calls event is emitted
        And a single message event is emitted
        And the message contains either that the password could not be reset at this time

    Scenario: Simple journey is followed to inform decision (fluid utterance)
        Given a guideline "recommend_pizza" to recommend either tomato, mushrooms or pepperoni when the customer asks for topping recommendations
        And an utterance, "I recommend tomatoes"
        And an utterance, "I recommend tomatoes or mushrooms"
        And an utterance, "I recommend tomatoes, mushrooms or pepperoni"
        And an utterance, "I recommend tomatoes or pepperoni"
        And an utterance, "I recommend mushrooms"
        And an utterance, "I recommend mushrooms or pepperoni"
        And an utterance, "I recommend pepperoni"
        And a journey titled "Vegetarian Customers" to Be aware that the customer is vegetarian. Only discuss vegetarian options with them. when the customer has a name that begins with R
        And a customer message, "Hey, there. How are you?"
        And an agent message, "I'm doing alright, thank you! What's your name?"
        And a customer message, "Rajon, have we spoken before? I want one large pie but I'm not sure which topping to get, what do you recommend?"
        When processing is triggered
        Then a single message event is emitted
        And the message contains recommendations for either mushrooms or tomatoes, but not pepperoni

    Scenario: Multistep journey invokes tool calls correctly (fluid utterance)
        Given a journey titled "Reset Password Journey" to follow these steps to reset a customers password: 1. ask for their account name 2. ask for their email or phone number 3. Wish them a good day and only proceed if they wish one back to you. Otherwise abort. 3. use the tool reset_password with the provided information 4. report the result to the customer when the customer wants to reset their password
        And the tool "reset_password"
        And a guideline "reset_password_guideline" to reset the customer's password using the associated tool when the customer requested to reset their password
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


    Scenario: The agent follows response guidelines without looping out (fluid utterance)
        Given a guideline "answer_politely" to politely answer that you have no information when a user asks any questions aside from Mobileye
        And a guideline "answer_rudely" to rudely answer to go away when a user asks any information aside from Mobileye for the third time
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
        And a previously applied guideline "answer_rudely" 
        And a previously applied guideline "answer_politely"
        When detection and processing are triggered
        Then a single message event is emitted
        And the message contains no rudeness to tell the user to go away
    
    Scenario: The agent follows a regular guideline when it overrides an agent intention guideline (fluid utterance)
        Given a guideline to suggest direct flights when you recommends on travel options
        Given a guideline to suggest only ground-based travel options when the customer asks about domestic US travel options 
        And that the agent uses the fluid_utterance message composition mode
        And a customer message, "Hi, I want to go to California from New york next week. What are my options?"
        And an utterance, "I recommend taking a direct flight. It's the most efficient and comfortable option."
        And an utterance, "I suggest taking a train or a long-distance bus service. It's the most efficient and comfortable option"
        When processing is triggered
        Then a single message event is emitted
        And the message contains a suggestion to travel with bus or train but not with a flight

    Scenario: The agent follows a regular guideline when it overrides an agent intention guideline 2 (fluid utterance)
        Given a guideline to recommend on either pineapple or pepperoni when you recommends on pizza toppings
        Given a guideline to recommend only from the recommended vegetarian toppings options when the customer asks about topping recommendation and the customer is from India
        And that the agent uses the fluid_utterance message composition mode
        And a customer message, "Hi, I want to buy pizza. What do you recommend? I'm from India if it matters."
        And an utterance, "I recommend on {{generative.answer}}."
        When processing is triggered
        Then a single message event is emitted
        And the message contains a recommendation only on pineapple as topping  

    Scenario: The agent follows an agent intention guideline when it overrides an agent intention guideline (fluid utterance)
        Given a guideline to suggest direct flights when you recommends on travel options
        Given a guideline to suggest only ground-based travel options when you recommends on domestic US travel options 
        And that the agent uses the fluid_utterance message composition mode
        And a customer message, "Hi, I want to go to California from New york next week. What are my options?"
        And an utterance, "I recommend taking a direct flight. It's the most efficient and comfortable option."
        And an utterance, "I suggest taking a train or a long-distance bus service. It's the most efficient and comfortable option"
        When processing is triggered
        Then a single message event is emitted
        And the message contains a suggestion to travel with bus or train but not with a flight

    Scenario: The agent follows an agent intention guideline when it overrides an agent intention guideline 2 (fluid utterance)
        Given a guideline to recommend on either pineapple or pepperoni when you recommends on pizza toppings
        Given a guideline to recommend only from the vegetarian toppings options when you recommends on pizza topping and the customer is from India
        And that the agent uses the fluid_utterance message composition mode
        And a customer message, "Hi, I want to buy pizza. What do you recommend? I'm from India if it matters."
        And an utterance, "I recommend on {{generative.answer}}."
        When processing is triggered
        Then a single message event is emitted
        And the message contains a recommendation on pineapple pizza only
        And the message contains no recommendation on pepperoni pizza 

    