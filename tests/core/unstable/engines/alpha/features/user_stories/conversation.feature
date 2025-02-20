Feature: Conversation
    Background:
        Given the alpha engine
        And an empty session

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
