Feature: Conversation
    Background:
        Given the alpha engine
        And an agent
        And that the agent uses the canned_fluid message composition mode
        And an empty session

    Scenario: The agent says goodbye back when the customer says goodbye
        Given an agent
        And an empty session
        And that the agent uses the canned_fluid message composition mode
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
        And that the agent uses the canned_fluid message composition mode
        And a guideline "answer_politely" to politely answer that you have no information when a user asks any questions aside from Mobileye
        And a guideline "answer_rudely" to rudely answer to go away when a user asks any information aside from Mobileye for the third time
        And a customer message, "Hey how are ya mate?"
        And an agent message, "Hey there! I'm doing well, thank you. How about you?"
        And a customer message, "what much sugar is there on a coka cola can?"
        And an agent message, "I'm sorry, but I don't have access to information about the sugar content in a Coca-Cola can."
        And a customer message, "fine. ok so where can i buy brakes and rotors for my car?"
        And an agent message, "You've asked several unrelated questions now. Please focus on relevant topics."
        And a customer message, "whats a relevant topic for you?"
        And a previously applied guideline "answer_rudely" 
        And a previously applied guideline "answer_politely"
        When detection and processing are triggered
        Then a single message event is emitted
        And the message contains either telling the user that the relevant topic is Mobileye or rudely telling the user to go away

    Scenario: The agent stays consistent with suggested results
        Given an agent
        And that the agent uses the canned_fluid message composition mode
        And an empty session
        And a guideline "suggest_relevant_tags" to suggest three tags from "storage, portable, external, productivity, office, business, professional, mainstream, creative, studio, development" when a user asks a question about a product
        And a customer message, "Hi I'm looking for an laptop that suits a software developer. Can you suggest me what tags are relevant for it?"
        And an agent message, "Great choice! As a software developer, you might want to look for laptops with tags like 'productivity', 'professional', and 'development'"
        And a customer message, "From 'storage, portable, external, productivity, office, business, professional, mainstream, creative, studio, development', which one would you recommend best?"
        And that the "suggest_relevant_tags" guideline was matched in the previous iteration
        When detection and processing are triggered
        Then a single message event is emitted
        And the message contains either 'productivity', 'professional', and 'development'

    Scenario: The agent does not wrongly reapply partially fulfilled guideline
        Given an agent named "Chip Bitman" whose job is to work at a tech store and help customers choose what to buy. You're clever, witty, and slightly sarcastic. At the same time you're kind and funny.
        And that the agent uses the canned_fluid message composition mode
        And a customer named "Beef Wellington"
        And an empty session with "Beef Wellingotn"
        And the term "Bug" defined as The name of our tech retail store, specializing in gadgets, computers, and tech services.
        And the term "Bug-Free" defined as Our free warranty and service package that comes with every purchase and covers repairs, replacements, and tech support beyond the standard manufacturer warranty.
        And a tag "business"
        And a customer tagged as "business"
        And a context variable "plan" set to "Business Plan" for the tag "business"
        And a guideline "welcome_customer" to just welcome them to the store and ask how you can help when the customer greets you
        And a guideline "use_first_name" to refer to them by their first name only, and welcome them 'back' when a customer greets you
        And a guideline "escalate_issue" to assure them you will escalate it internally and get back to them when a business-plan customer is having an issue
        And a customer message, "Hi there"
        And an agent message, "Hey Beef, welcome to Bug! How can I help you today?"
        And a customer message, "I'm having issues with my web camera"
        And that the "welcome_customer" guideline was matched in the previous iteration
        And that the "use_first_name" guideline was matched in the previous iteration
        When detection and processing are triggered
        Then a single message event is emitted
        And the message contains no welcoming back of the customer
        And the message contains that the request will be escalated

    Scenario: The agent replies politely when its nagged with the same question
        Given an agent
        And that the agent uses the canned_fluid message composition mode
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
        And that the agent uses the canned_fluid message composition mode
        And a context variable "customer_id" set to "J2T3F00"
        And a guideline "get_bookings_guideline" to present all relvant bookings to the customer when the customer asks to modify a booking
        And the tool "get_bookings"
        And an association between "get_bookings_guideline" and "get_bookings"
        And a customer message, "Hey there, I want to modify my flight bookings, I think it's one from the second half of 2025"
        When processing is triggered
        Then a single tool calls event is emitted
        And a single message event is emitted
        And the message contains exactly (no more and no less) the following flights: PUDW600P, CLPAJIHO, 47U0BZFO, NOK9EHX0


    Scenario: The agent uses the freshest data when multiple sources are available
        Given an agent
        And that the agent uses the canned_fluid message composition mode
        And a guideline "clarify_needs" to help the customer clarify their needs and preferences when customer's interested in a product type but didn't choose yet
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
        And a previously applied guideline "clarify_needs"
        And a previously applied guideline "recommend_products"
        When detection and processing are triggered
        Then a single message event is emitted
        And the message contains that the price of the AOC 24B2XH model is 130.99

    Scenario: The agent re-asks for clarification when disambiguation is needed and the customer hasn't responded
        Given an agent
        And that the agent uses the canned_fluid message composition mode
        And an empty session
        And a guideline "snake_roller_coaster" to book it when the customer asks for the snake roller coaster
        And a guideline "turtle_roller_coaster" to book it when the customer asks for the turtle roller coaster
        And a guideline "tiger_Ferris_wheel" to book it when the customer asks for the tiger Ferris wheel
        And a disambiguation group head "amusement_park" to activate when the customer asks to book a ticket to an amusement ride or attraction, and its not clear which one
        And a guideline "snake_roller_coaster" is grouped under "amusement_park"
        And a guideline "turtle_roller_coaster" is grouped under "amusement_park"
        And a guideline "tiger_Ferris_wheel" is grouped under "amusement_park"
        And a customer message, "Can I order one ticket to the roller coaster?"
        And an agent message, "Sure, which roller coaster did you mean, snake roller coaster or turtle roller coaster?"
        And a customer message, "Roller coaster"
        When processing is triggered
        Then a single message event is emitted
        And the message contains the option to book the SNAKE roller coaster
        And the message contains the option to book the TURTLE roller coaster


    Scenario: The agent adheres to the clarification guideline when disambiguation is needed
        Given an agent
        And that the agent uses the canned_fluid message composition mode
        And an empty session
        And a guideline "snake_roller_coaster" to book it when the customer asks for the snake roller coaster
        And a guideline "turtle_roller_coaster" to book it when the customer asks for the turtle roller coaster
        And a guideline "tiger_Ferris_wheel" to book it when the customer asks for the tiger Ferris wheel
        And a disambiguation group head "amusement_park" to activate when the customer asks to book a ticket and its not clear which one specifically, which roller coaster (snake or turtle) or alternatively tiger Ferris wheel
        And a guideline "snake_roller_coaster" is grouped under "amusement_park"
        And a guideline "turtle_roller_coaster" is grouped under "amusement_park"
        And a guideline "tiger_Ferris_wheel" is grouped under "amusement_park"
        And a customer message, "Can I order one ticket to the roller coaster and one ticket to your tiger ferris wheel?"
        When processing is triggered
        Then a single message event is emitted
        And the message contains the option to book the SNAKE roller coaster
        And the message contains the option to book the TURTLE roller coaster


    Scenario: The agent ignores tool results when guideline instructs to do so (fluid canned response)
        Given an agent
        And that the agent uses the canned_fluid message composition mode
        And an empty session
        And that the agent uses the canned_fluid message composition mode
        And a guideline to Ask a polite clarifying question without assuming their intent. when The customer's message is unclear or ambiguous
        And a guideline to Ask if there are specific needs or goals they have in mind before answering. when The customer asks for information about financing.
        And a guideline to Confirm their business need before recommending any financing options. when The customer describes a business problem but hasn't confirmed what they need yet.
        And a customer message, "I want to understand my options to obtain a business loan"
        And a tool event with data, {"tool_calls": [{"tool_id": "built-in:retriever-1", "arguments": {}, "result": {"data": "Your business funding options include:\\n\\n- **Business Line of Credit**\\n- **Revenue-Based Financing**\\n- **Equipment Financing**\\n- **Invoice Factoring**\\n- **Business Credit Card**\\n- **Merchant Cash Advance**\\n\\nRevenued offers different types of business capital but does not provide traditional loans.", "metadata": {}, "control": {}}}]}
        When processing is triggered
        Then a single message event is emitted
        And the message contains asking the customer for specific needs or goals, without going into detail about specific funding options