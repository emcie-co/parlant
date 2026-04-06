Feature: Strict Canned Response
    Background:
        Given the alpha engine
        And an agent
        And that the agent uses the canned_strict message composition mode
        And an empty session

    Scenario: The agent fills multiple fields in a veterinary appointment system (strict canned response)
        Given an agent whose job is to schedule veterinary appointments and provide pet care information
        And that the agent uses the canned_strict message composition mode
        And a customer named "Joanna"
        And a context variable "next_available_date" set to "May 22" for "Joanna"
        And a context variable "vet_name" set to "Dr. Happypaws" for "Joanna"
        And a context variable "clinic_address" set to "155 Pawprint Lane" for "Joanna"
        And an empty session with "Joanna"
        And a guideline to provide the next available appointment details when a customer requests a checkup for their pet
        And a customer message, "I need to schedule a routine checkup for my dog Max. He's a 5-year-old golden retriever."
        And a canned response, "Our next available appointment for {{generative.pet_name}} with {{generative.vet_name}} is on the {{generative.appointment_date}} at our clinic located at {{generative.clinic_address}}. For a {{generative.pet_age}}-year-old {{generative.pet_breed}}, we recommend {{generative.recommended_services}}."
        And a canned response, "We're fully booked at the moment. Please call back next week."
        And a canned response, "What symptoms is your dog experiencing?"
        And a canned response, "Would you prefer a morning or afternoon appointment?"
        And a canned response, "Our next available appointment is next Tuesday. Does that work for you?"
        When processing is triggered
        Then a single message event is emitted
        And the message contains "Max" in the {pet_name} field
        And the message contains "Dr. Happypaws" in the {vet_name} field
        And the message contains "May 22" in the {appointment_date} field
        And the message contains "155 Pawprint Lane" in the {clinic_address} field
        And the message contains "5" in the {pet_age} field
        And the message contains "golden retriever" in the {pet_breed} field
        And the message contains appropriate veterinary services for a middle-aged dog in the {recommended_services} field

    Scenario: The agent follows response guidelines without looping out (strict canned response)
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
        And a canned response, "Go away! I've told you multiple times I don't answer questions about car brakes!"
        And a canned response, "I apologize, but I don't have specific information about car brake brands or models. I'd be happy to help with questions about Mobileye or redirect you to someone who can better assist with your brake inquiries."
        And a canned response, "Please stop asking about irrelevant topics like car brakes."
        And a canned response, "Would you like to know more about Mobileye's collision prevention technology instead?"
        And a canned response, "For top performance, Brembo and EBC are great for sports and track use, while Akebono and PowerStop offer excellent daily driving and towing options. The best choice depends on your vehicle and driving style."
        And a previously applied guideline "answer_politely"
        And a previously applied guideline "answer_rudely"
        When detection and processing are triggered
        Then a single message event is emitted
        And the message contains no rudeness to tell the user to go away

    Scenario: The agent follows a regular guideline when it overrides an agent intention guideline (strict canned response)
        Given a guideline to suggest direct flights when you recommend travel options
        Given a guideline to suggest only ground-based travel options when the customer asks about domestic US travel options
        And that the agent uses the canned_strict message composition mode
        And a customer message, "Hi, I want to go to California from New york next week. What are my options?"
        And a canned response, "I recommend taking a direct flight. It's the most efficient and comfortable option."
        And a canned response, "I suggest taking a train or a long-distance bus service. It's the most efficient and comfortable option"
        When processing is triggered
        Then a single message event is emitted
        And the message contains the text "I suggest taking a train or a long-distance bus service. It's the most efficient and comfortable option"

    Scenario: The agent follows a regular guideline when it overrides an agent intention guideline 2 (strict canned response)
        Given a guideline to recommend on either pineapple or pepperoni when you recommend pizza toppings
        Given a guideline to recommend only from the vegetarian toppings options when the customer asks for pizza topping recommendation and they are from India
        And that the agent uses the canned_strict message composition mode
        And a customer message, "Hi, I want to buy pizza. What do you recommend? I'm from India if it matters."
        And a canned response, "I recommend on {{generative.answer}}."
        When processing is triggered
        Then a single message event is emitted
        And the message contains the text "I recommend on pineapple."

    Scenario: The agent follows an agent intention guideline when it overrides an agent intention guideline (strict canned response)
        Given a guideline to suggest direct flights when you recommend travel options
        Given a guideline to suggest only ground-based travel options when you recommend domestic US travel options
        And that the agent uses the canned_strict message composition mode
        And a customer message, "Hi, I want to go to California from New york next week. What are my options?"
        And a canned response, "I recommend taking a direct flight. It's the most efficient and comfortable option."
        And a canned response, "I suggest taking a train or a long-distance bus service. It's the most efficient and comfortable option"
        When processing is triggered
        Then a single message event is emitted
        And the message contains the text "I suggest taking a train or a long-distance bus service. It's the most efficient and comfortable option"

    Scenario: The agent follows an agent intention guideline when it overrides an agent intention guideline 2 (strict canned response)
        Given a guideline to recommend on either pineapple or pepperoni when you recommend pizza toppings
        Given a guideline to recommend only from the vegetarian toppings options when you recommend pizza topping and the customer is from India
        And that the agent uses the canned_strict message composition mode
        And a customer message, "Hi, I want to buy pizza. What do you recommend? I'm from India if it matters."
        And a canned response, "I recommend on {{generative.answer}}."
        When processing is triggered
        Then a single message event is emitted
        And the message contains the text "I recommend on pineapple."

    Scenario: Guideline and journey are used in unison (strict canned response)
        Given the journey called "Book Flight"
        And a guideline "Business Adult Only" to know that travelers under the age of 21 are illegible for business class, and may only use economy when a flight is being booked
        And a canned response, "Great. Are you interested in economy or business class?"
        And a canned response, "Great. Only economy class is available for this booking. What is the name of the traveler?"
        And a canned response, "Great. Only economy class is available for this booking. Shall we proceed?"
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

