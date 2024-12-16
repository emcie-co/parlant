Feature: Conversation
    Background:
        Given the alpha engine
        And an empty session

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