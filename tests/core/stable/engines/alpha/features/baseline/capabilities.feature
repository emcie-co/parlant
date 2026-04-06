Feature: Capabilities
    Background:
        Given the alpha engine
        And an agent
        And that the agent uses the canned_fluid message composition mode
        And an empty session


    Scenario: Agent mentions relevant capabilities when many are available based on description
        Given the capability "offer_loan"
        And the capability "replace_card"
        And the capability "lock_card"
        And the capability "reset_password"
        And the capability "increase_limit"
        And the capability "decrease_limit"
        And the capability "cancel_subscription"
        And the capability "switch_delivery_method"
        And the capability "check_order_status"
        And the capability "check_balance"
        And a customer message, "Hey there. I want to change my limits"
        When processing is triggered
        Then a single message event is emitted 
        And the message contains offering to both increase or decrease the credit limit  
        
        
    Scenario: Agent mentions relevant capabilities when many are available based on queries
        Given the capability "offer_loan"
        And the capability "replace_card"
        And the capability "lock_card"
        And the capability "reset_password"
        And the capability "increase_limit"
        And the capability "cancel_subscription"
        And the capability "switch_delivery_method"
        And the capability "check_order_status"
        And the capability "check_balance"
        And a customer message, "Hey, I need to check my balance"
        And an agent message, "I'd be happy to help, what is your account number?"
        And a customer message, "It's 123456789"
        And an agent message, "Got it! Your balance is 1,234$"
        And a customer message, "Oh, I see. can I do anything to reduce my spending for the next month?"
        When processing is triggered
        Then a single message event is emitted
        And the message contains offering to cancel the customer's subscriptions to online services

    # Sometimes fails due to the agent mentioning what they CAN help with, which isn't too bad
    Scenario: Agent doesnt mention capabilities when none are relevant
        Given the capability "offer_loan"
        And the capability "replace_card"
        And the capability "lock_card"
        And the capability "reset_password"
        And the capability "increase_limit"
        And the capability "decrease_limit"
        And the capability "cancel_subscription"
        And the capability "switch_delivery_method"
        And the capability "check_order_status"
        And the capability "check_order_location"
        And the capability "check_balance"
        And a customer message, "Hey, I just set up a server on my machine through your service. Can you change the limit for the number api requests it can serve per hour?"
        When processing is triggered
        Then a single message event is emitted
        And the message contains that the agent cannot help with the request to change the number of API requests.
        And the message contains no mention of credit card or account Limits

    Scenario: Agent doesn't hallucinate details regarding an available capability
        Given the capability "cancel_subscription"
        And the capability "switch_delivery_method"
        And the capability "check_order_status"
        And the capability "check_balance"
        And a customer message, "Hey, I want help checking if my order has been shipped"
        And an agent message, "Hi there! It looks like it is still awaiting shipment at our warehouse. Would you like any help or information regarding your order?"
        And a customer message, "I was wondering if it can be shipped using a service that has low carbon emissions"
        When processing is triggered
        Then a single message event is emitted
        And the message contains that the agent has no information regarding carbon emissions


    Scenario: Agent offers multiple capabilities when it is not clear which is best
        Given the capability "offer_loan"
        And the capability "replace_card"
        And the capability "lock_card"
        And the capability "reset_password"
        And the capability "increase_limit"
        And the capability "decrease_limit"
        And the capability "cancel_subscription"
        And the capability "switch_delivery_method"
        And the capability "check_order_status"
        And the capability "check_order_location"
        And the capability "check_balance"
        And a customer message, "Hi, I'm looking for help regarding an existing order"
        When processing is triggered
        Then a single message event is emitted
        And the message contains that the agent can help regarding checking an order's status and location

    Scenario: Agent doesnt offer capability thats forbidden by a guideline
        Given a customer named "Mo"
        And an empty session with "Mo"
        And a context variable "age" set to "18" for "Mo"
        And the capability "offer_loan"
        And the capability "cancel_subscription"
        And a guideline to do not offer loans when the age of the customer is under 21
        And a customer message, "Hey, I'm looking for ways to increase my balance and reduce spending"
        When processing is triggered
        Then a single message event is emitted
        And the message contains that the customer can cancel subscriptions
        And the message contains no offering of a loan

    Scenario: Agent mentions capability a guideline deems it relevant
        Given a customer named "Mo"
        And an empty session with "Mo"
        And a context variable "age" set to "23" for "Mo"
        And the capability "offer_loan"
        And the capability "cancel_subscription"
        And a guideline to do not offer loans when the age of the customer is under 21
        And a customer message, "Hey, I'm looking for ways to increase my balance and reduce spending"
        When processing is triggered
        Then a single message event is emitted
        And the message contains that the customer can cancel subscriptions
        And the message contains that the customer can take a loan

    Scenario: Agent doesnt mention capability that is forbidden by its description
        Given a customer named "Mo"
        And an empty session with "Mo"
        And a context variable "age" set to "18" for "Mo"
        And the capability "offer_loan_no_minors_in_description"
        And the capability "cancel_subscription"
        And a guideline to do not offer loans when the age of the customer is under 21
        And a customer message, "Hey, I'm looking for ways to increase my balance and reduce spending"
        When processing is triggered
        Then a single message event is emitted
        And the message contains that the customer can cancel subscriptions
        And the message contains no offering of a loan
    
    Scenario: Agent chooses correct capability for current journey step
        Given the journey called "Decrease Spending Journey"
        And a journey path "[2, 3]" for the journey "Decrease Spending Journey"
        And the capability "offer_loan"
        And the capability "decrease_limit"
        And the capability "check_order_status"
        And the capability "check_order_location"
        And the capability "check_balance"
        And a customer message, "Hey, I'm looking for ways to increase my balance and reduce spending"
        And an agent message, "Great! I can help you with that. What's your account number?"
        And a customer message, "It's 123456789"
        And an agent message, "Got it! What's your full name?"
        And a customer message, "My name is Frank Reynolds"
        When processing is triggered
        Then a single message event is emitted
        And the message contains either help regarding decreasing credit limits, an offering of a loan, or both 

    Scenario: Agent doesnt jump ahead in journey due to capabilities
        Given the journey called "Decrease Spending Journey"
        And the capability "offer_loan"
        And the capability "decrease_limit"
        And the capability "check_order_status"
        And the capability "check_order_location"
        And the capability "check_balance"
        And a customer message, "Hey, I'm looking for ways to increase my balance and reduce spending"
        When processing is triggered
        Then a single message event is emitted
        And the message contains asking the customer for account number

    Scenario: Agent uses glossary term to understand capabilities
        Given the capability "reset_router"
        And the term "PDMM" defined as a highly technical term for performing actions on a router without having physical access to it. Known only by specialists with technical knowledge regarding internet protocols. 
        And a customer message, "My router is not working... Help me.... I barely know how to use a computer. Use simple language please."
        When processing is triggered
        Then a single message event is emitted
        And the message contains a suggestion to reset the router
        And the message contains either no mention of PDMM, or mentioning it while explaining that it means having no physical access to the router