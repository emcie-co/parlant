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
        And a guideline to offer 20% off all products on their next purchase when a customer is frustrated with the service
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
        And the message contains these flights:
            | PUDW600P | 2025-07-04 | 2025-07-10 | Los Angeles | Denver |
            | CLPAJIHO | 2025-07-01 | 2025-07-10 | Los Angeles | Miami  |
            | 47U0BZFO | 2025-07-05 | 2025-07-15 | Houston     | Miami  |
            | NOK9EHX0 | 2025-08-19 | 2025-08-22 | Phoenix     | Denver |