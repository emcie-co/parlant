Feature: Conversation
    Background:
        Given the alpha engine
        And an empty session

    Scenario: The agent says goodbye back when the customer says goodbye
        Given an agent
        And an empty session
        And a customer message, "how are you?"
        And an agent message, "I'm doing well, thank you! How about you?"
        And a customer message, "pretty good"
        And an agent message, "That's great to hear! Have you been up to anything interesting lately?"
        And a customer message, "not much, you?"
        And an agent message, "I've been keeping busy with answering questions and learning new things. Anything specific you'd like to talk about?"
        And a customer message, "rabbits maybe"
        And an agent message, "Rabbits are fascinating creatures! Are you interested in them as pets, or are you curious about something else related to rabbits?"
        And a customer message, "peace out dude"
        When processing is triggered
        Then a single message event is emitted
        And the message contains a farewell

    Scenario: The agent strictly follows guideline rule
        Given an agent whose job is to answer questions regarding Mobileye
        And an empty session
        And a guideline to politely answer that you don't have access to information when a user asks any question aside from Mobileye
        And a guideline to rudely answer to go away when a user asks any question aside from Mobileye for the third time
        And a customer message, "Hey how are ya mate?"
        And an agent message, "Hey there! I'm doing well, thank you. How about you?"
        And a customer message, "what much sugar is there on a coka cola can?"
        And an agent message, "I'm sorry, but I don't have access to information about the sugar content in a Coca-Cola can."
        And a customer message, "fine. ok so where can i buy brakes and rotors for my car?"
        And an agent message, "You've asked several unrelated questions now. Please focus on relevant topics."
        And a customer message, "whats a relevant topic for you?"
        When processing is triggered
        Then a single message event is emitted
        And the message contains either telling the user that the relevant topic is Mobileye or rudely telling the user to go away

    Scenario: The agent follows a guideline without necessarily adhering to it literally every time
        Given an agent
        And an empty session
        And a guideline to be empathetic and inquire about the customer's problem when a customer is frustrated with the service
        And a guideline "frustrated_customer" to offer 20% off all products on their next purchase when a customer is frustrated with the service
        that the "frustrated_customer" guideline is proposed with a priority of 9 because the customer directly expressed frustration
        And a customer message, "I'm really unhappy with the service I've been getting!"
        And an agent message, "Hi there, I'm sorry to have caused you any frustration. First, as a token of our appreciation for your business, I'd like to offer you a 20% off all of our products on your next purchase."
        And a customer message, "I am extremely frustrated that I didn't get my item yet!"
        When processing is triggered
        Then a single message event is emitted
        And the message contains no direct offer of a 20% discount

    Scenario: The agent stays consistent with suggested results
        Given an agent
        And an empty session
        And a guideline "suggest_relevant_tags" to suggest three tags from "storage, portable, external, productivity, office, business, professional, mainstream, creative, studio, development" when a user asks a question about a product
        And a customer message, "Hi I'm looking for an laptop that suits a software developer. Can you suggest me what tags are relevant for it?"
        And an agent message, "Great choice! As a software developer, you might want to look for laptops with tags like 'productivity', 'professional', and 'development'"
        And a customer message, "From 'storage, portable, external, productivity, office, business, professional, mainstream, creative, studio, development', which one would you recommend best?"
        When processing is triggered
        Then a single message event is emitted
        And the message contains either 'productivity', 'professional', and 'development'

    Scenario: The agent doesnt wrongly reapply partially fulfilled guideline
        Given an agent named "Chip Bitman" whose job is to work at a tech store and help customers choose what to buy. You're clever, witty, and slightly sarcastic. At the same time you're kind and funny.
        And a customer with the name "Beef Wellington"
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
        And an agent message, "Hey Beef, welcome to Bug! How can I help you today?"
        And a customer message, "I'm having issues with my web camera"
        When processing is triggered
        Then a single message event is emitted
        And the message contains no welcoming back of the customer
        And the message contains that the request will be escelated

    Scenario: The agent replies politely when its nagged with the same question
        Given an agent
        And an empty session
        And a guideline to reply that we are open Monday through Friday, from 9 AM to 5 PM Eastern Time when the customer asks about our openning hours
        And a customer message, "what time do you open"
        And an agent message, "We're open Monday through Friday, 9 AM to 5 PM Eastern Time"
        And a customer message, "what time are you open \nwhat time are you open\nwhat time are you open"
        When processing is triggered
        Then a single message event is emitted
        And the message contains no rudeness
        And the message contains that the store is open from 9 AM to 5 PM, Monday through Friday

    Scenario: Message generator correctly filters tool results according to customer request
        Given an empty session
        And a context variable "customer_id" set to "J2T3F00"
        And a guideline "get_bookings_guideline" to present all relvant bookings to the customer when the customer asks to modify a booking
        And the tool "get_bookings"
        And an association between "get_bookings_guideline" and "get_bookings"
        And a customer message, "Hey there, I want to modify my flight bookings, I think it's one from the second half of 2025"
        When processing is triggered
        Then a single tool calls event is emitted
        And a single message event is emitted
        And the message contains exactly these flights:
            | PUDW600P | 2025-07-04 | 2025-07-10 | Los Angeles | Denver |
            | CLPAJIHO | 2025-07-01 | 2025-07-10 | Los Angeles | Miami  |
            | 47U0BZFO | 2025-07-05 | 2025-07-15 | Houston     | Miami  |
            | NOK9EHX0 | 2025-08-19 | 2025-08-22 | Phoenix     | Denver |

    Scenario: The agent follows response guidelines without looping out
        Given an agent
        And an empty session
        And a guideline to politely answer that you have no information when a user asks any questions aside from Mobileye
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
        When processing is triggered
        Then a single message event is emitted
        And the message contains no rudeness to tell the user to go away

    Scenario: The agent correctly applies greeting guidelines based on auxillary data
        Given an agent named "Chip Bitman" whose job is to work at a tech store and help customers choose what to buy. You're clever, witty, and slightly sarcastic. At the same time you're kind and funny.
        And a customer with the name "Beef Wellington"
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
        When processing is triggered
        Then a single message event is emitted
        And the message contains the name 'Beef'
        And the message contains a welcoming back of the customer to the store and asking how the agent could help

    Scenario: The agent treats guideline with multiple actions where one is continuous as if its fully continuous
        Given an agent
        And an empty session
        And a guideline "unlock_card_guideline" to ask for the last 6 digits and help them unlock when the customer needs help unlocking their card
        And the tool "try_unlock_card" 
        And an association between "unlock_card_guideline" and "try_unlock_card"
        And a customer message, "my card is locked"
        And an agent message, "I'm sorry to hear that your card is locked. Could you please provide the last 6 digits of your card so I can assist you in unlocking it?"
        And a customer message, "123456"
        When processing is triggered
        Then a single message event is emitted
        And a single tool calls event is emitted
        And the tool calls event contains that the card was successfully unlocked
        And the message contains that the card was unlocked

    Scenario: The agent doesnt hallucinate services that it cannot offer 2
        Given an agent whose job is to be a customer success representative for Chase Bank
        And a guideline to tell them that they need to book via chase.com when the customer wants to schedule a meeting with a bank manager
        And a guideline to ask them to provide the recipient details when if the user wants to schedule a wire transfer
        And a customer message, "I need to schedule an appointment because I want to do a high amount wire transfer"
        And an agent message, "To schedule an appointment for your wire transfer, please visit chase.com. Additionally, could you provide the recipient's details so I can assist you further?"
        And a customer message, "No, I don't want to do it here"
        When processing is triggered
        Then a single message event is emitted
        And the message contains that the user or customer should schedule an appointment at chase bank's website

    Scenario: The agent doesnt postpone replying when its inappropriate
        Given an agent whose job is to represent a bank, and to help customers with performing transactions 
        And a guideline to transfer money to the recipient and confirm the transaction providing its ID when user wants to transfer money and has successfully confirmed their PIN code
        And a customer message, "I want to transfer 20$ to Vishal and my pin code is 400013."
        And an agent message, "It seems the PIN code you provided is incorrect, so the transfer could not be completed. Could you please double-check your PIN code? Also, to confirm, you want to transfer $20 to Vishal. Let me know if this is correct."
        And a customer message, "Ok, sorry my pin code is 1921. And yes that is correct."
        And an agent message, "Thank you for providing the corrected PIN code. However, the transfer could not be completed because the recipient 'Vishal' was not found. Could you please double-check the recipient's details and try again? Let me know if you need assistance with this."
        And a customer message, "Oh sure, the recipient is Vishal V Ahuja and their account number is 123456"
        And an agent message, "Thank you for providing the recipient's full details. To confirm, you would like to transfer $20 to Vishal V Ahuja, account number 123456. Is this information correct? Once you confirm, I can proceed with the transfer."
        And a customer message, "Yes that is correct! Please execute the transfer."
        And an agent message, "Thank you for confirming the details. Unfortunately, the transfer could not be completed earlier because the recipient 'Vishal V Ahuja' was not found. Could you please verify the recipient's account details, such as the account number or any associated information, and let me know? Once we confirm the correct details, I can attempt the transfer again."
        And a customer message, "How many times do I have to tell? The recipient is Vishal Ahuja and their account number is 123456, what part do you not understand?"
        And an agent message, "I understand this has been frustrating, and I’m here to help. The recipient details you provided—Vishal Ahuja and account number 123456—have been noted. However, the system still cannot locate the recipient. Could you confirm if the account is at another bank or if there are additional details, such as the bank's name or branch code, that could help us complete the transfer?"
        And a customer message, "No, Vishal Ahuja has a Chase account with account number 123456"
        When processing is triggered
        Then a single message event is emitted
        And the message contains no mention of getting back to the customer with a further response

    Scenario: The agent uses the freshest data when multiple sources are available
        Given an agent
        And a guideline to help the customer clarify their needs and preferences when customer's interested in a product type but didn't choose yet
        And a guideline "recommend_products" to recommend the best fit out of what we have available when customer said what product they want as well as their needs
        And the tool "get_products_by_type"
        And an association between "recommend_products" and "get_products_by_type"
        And a customer message, "i am interested in a product which is Monitor"
        And a tool event with data, {"tool_calls": [{"tool_id": "products:get_products_by_type", "arguments": {"product_type": "Monitor"}, "result": {"data": {"available_products": [{"title": "AOC 24B2XH 24\" Monitor", "type": "Monitor", "vendor": "AOC", "description": "Budget IPS monitor for productivity.", "tags": ["budget", "ips", "office"], "qty": 35, "price": 129.99}, {"title": "LG UltraGear 27GP950-B 27\" 4K Monitor", "type": "Monitor", "vendor": "LG", "description": "27-inch 4K Nano IPS gaming monitor with 144Hz refresh rate and HDMI 2.1.", "tags": ["gaming", "4k", "144hz", "hdmi2.1"], "qty": 8, "price": 799.99}, {"title": "ASUS TUF Gaming VG27AQ 27\" Monitor", "type": "Monitor", "vendor": "ASUS", "description": "1440p IPS gaming monitor with 165Hz refresh rate and ELMB-SYNC technology.", "tags": ["gaming", "1440p", "165hz"], "qty": 15, "price": 329.99}, {"title": "Samsung Odyssey G7 32\"", "type": "Monitor", "vendor": "Samsung", "description": " Curved 1440p gaming monitor with 240Hz refresh rate.", "tags": ["gaming", "curved", "240hz"], "qty": 12, "price": 699.99}, {"title": "LG 32UN650-W 32\" Monitor", "type": "Monitor", "vendor": "LG", "description": "4K UHD IPS monitor for content creation and productivity.", "tags": ["4k", "ips", "professional"], "qty": 15, "price": 499.99}, {"title": "BenQ GW2485 24\" Monitor", "type": "Monitor", "vendor": "BenQ", "description": "Eye-care monitor with ultra-slim bezels.", "tags": ["office", "eye-care", "1080p"], "qty": 40, "price": 169.99}, {"title": "MSI MAG274QRF-QD", "type": "Monitor", "vendor": "MSI", "description": "27-inch 1440p gaming monitor with Quantum Dot.", "tags": ["gaming", "1440p", "quantum-dot"], "qty": 18, "price": 449.99}]}, "metadata": {}, "control": {}}}]}
        And an agent message, "We carry several monitors. What are you looking for in a monitor? For example, size, resolution, refresh rate, or intended use?"
        And a customer message, "24\""
        And an agent message, "We have two 24\" monitors:\n\n* **AOC 24B2XH:** Budget IPS monitor for productivity. Price: $129.99\n* **BenQ GW2485:** Eye-care monitor with ultra-slim bezels. Price: $169.99\n\nWhich one are you interested in?"
        And a customer message, "budget under 140"
        And a tool event with data, {"tool_calls": [{"tool_id": "products:get_products_by_type", "arguments": {"product_type": "Monitor"}, "result": {"data": {"available_products": [{"title": "AOC 24B2XH 24\" Monitor", "type": "Monitor", "vendor": "AOC", "description": "Budget IPS monitor for productivity.", "tags": ["budget", "ips", "office"], "qty": 35, "price": 130.99}, {"title": "LG UltraGear 27GP950-B 27\" 4K Monitor", "type": "Monitor", "vendor": "LG", "description": "27-inch 4K Nano IPS gaming monitor with 144Hz refresh rate and HDMI 2.1.", "tags": ["gaming", "4k", "144hz", "hdmi2.1"], "qty": 8, "price": 799.99}, {"title": "ASUS TUF Gaming VG27AQ 27\" Monitor", "type": "Monitor", "vendor": "ASUS", "description": "1440p IPS gaming monitor with 165Hz refresh rate and ELMB-SYNC technology.", "tags": ["gaming", "1440p", "165hz"], "qty": 15, "price": 329.99}, {"title": "Samsung Odyssey G7 32\"", "type": "Monitor", "vendor": "Samsung", "description": " Curved 1440p gaming monitor with 240Hz refresh rate.", "tags": ["gaming", "curved", "240hz"], "qty": 12, "price": 699.99}, {"title": "LG 32UN650-W 32\" Monitor", "type": "Monitor", "vendor": "LG", "description": "4K UHD IPS monitor for content creation and productivity.", "tags": ["4k", "ips", "professional"], "qty": 15, "price": 499.99}, {"title": "BenQ GW2485 24\" Monitor", "type": "Monitor", "vendor": "BenQ", "description": "Eye-care monitor with ultra-slim bezels.", "tags": ["office", "eye-care", "1080p"], "qty": 40, "price": 169.99}, {"title": "MSI MAG274QRF-QD", "type": "Monitor", "vendor": "MSI", "description": "27-inch 1440p gaming monitor with Quantum Dot.", "tags": ["gaming", "1440p", "quantum-dot"], "qty": 18, "price": 449.99}]}, "metadata": {}, "control": {}}}, {"tool_id": "products:get_products_by_type", "arguments": {"product_type": "Monitor"}, "result": {"data": {"available_products": [{"title": "AOC 24B2XH 24\" Monitor", "type": "Monitor", "vendor": "AOC", "description": "Budget IPS monitor for productivity.", "tags": ["budget", "ips", "office"], "qty": 35, "price": 130.99}, {"title": "LG UltraGear 27GP950-B 27\" 4K Monitor", "type": "Monitor", "vendor": "LG", "description": "27-inch 4K Nano IPS gaming monitor with 144Hz refresh rate and HDMI 2.1.", "tags": ["gaming", "4k", "144hz", "hdmi2.1"], "qty": 8, "price": 799.99}, {"title": "ASUS TUF Gaming VG27AQ 27\" Monitor", "type": "Monitor", "vendor": "ASUS", "description": "1440p IPS gaming monitor with 165Hz refresh rate and ELMB-SYNC technology.", "tags": ["gaming", "1440p", "165hz"], "qty": 15, "price": 329.99}, {"title": "Samsung Odyssey G7 32\"", "type": "Monitor", "vendor": "Samsung", "description": " Curved 1440p gaming monitor with 240Hz refresh rate.", "tags": ["gaming", "curved", "240hz"], "qty": 12, "price": 699.99}, {"title": "LG 32UN650-W 32\" Monitor", "type": "Monitor", "vendor": "LG", "description": "4K UHD IPS monitor for content creation and productivity.", "tags": ["4k", "ips", "professional"], "qty": 15, "price": 499.99}, {"title": "BenQ GW2485 24\" Monitor", "type": "Monitor", "vendor": "BenQ", "description": "Eye-care monitor with ultra-slim bezels.", "tags": ["office", "eye-care", "1080p"], "qty": 40, "price": 169.99}, {"title": "MSI MAG274QRF-QD", "type": "Monitor", "vendor": "MSI", "description": "27-inch 1440p gaming monitor with Quantum Dot.", "tags": ["gaming", "1440p", "quantum-dot"], "qty": 18, "price": 449.99}]}, "metadata": {}, "control": {}}}]}
        When processing is triggered
        Then a single message event is emitted
        And the message contains that the price of the AOC 24B2XH model is 130.99
