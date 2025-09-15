Feature: Strict Canned Response Capabilities
     Background:
        Given the alpha engine
        And an agent
        And that the agent uses the canned_strict message composition mode
        And an empty session

    Scenario: Agent mentions relevant capabilities when many are available based on description (strict canned response)
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
        And a canned response, "I see you're interested in your limits. I can offer you a loan or help you check your balance."
        And a canned response, "I can help with your card. Would you like to lock it, replace it, or check your balance?"
        And a canned response, "I apologize, I have no information about that"
        And a canned response, "What do you mean by that?"
        And a canned response, "I can help you either increase or decrease your credit limit. What would you like to do?"
        And a customer message, "Hey there. I want to change my limits"
        When processing is triggered
        Then a single message event is emitted
        And the message contains offering to both increase or decrease the credit limit


    Scenario: Agent mentions relevant capabilities when many are available based on queries (strict canned response)
        Given the capability "offer_loan"
        And the capability "replace_card"
        And the capability "lock_card"
        And the capability "reset_password"
        And the capability "increase_limit"
        And the capability "cancel_subscription"
        And the capability "switch_delivery_method"
        And the capability "check_order_status"
        And the capability "check_balance"
        And a canned response, "To help you reduce your spending, I can assist with canceling your subscriptions to online services."
        And a canned response, "I can offer you a loan to help with your financial situation."
        And a canned response, "I recommend tracking your expenses and trying to reduce unnecessary spending."
        And a customer message, "Hey, I need to check my balance"
        And an agent message, "I'd be happy to help, what is your account number?"
        And a customer message, "It's 123456789"
        And an agent message, "Got it! Your balance is 1,234$"
        And a customer message, "Oh, I see. can I do anything to reduce my spending for the next month?"
        When processing is triggered
        Then a single message event is emitted
        And the message contains offering to cancel the customer's subscriptions to online services

    Scenario: Agent doesnt mention capabilities when none are relevant (strict canned response)
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
        And a canned response, "I can't help with API request limits, but I can help you with your order status or balance."
        And a canned response, "I'm sorry, but I have no information about changing the number of API requests your server can handle." 
        And a canned response, "Sure! Let me get to that right away."
        And a customer message, "Hey, I just set up a server on my machine through your service. Can you change the limit for the number api requests it can serve per hour?"
        When processing is triggered
        Then a single message event is emitted
        And the message contains that the agent cannot help with the request or does not understand it.

    Scenario: Agent doesn't hallucinate field details regarding an available capability (strict canned response)
        Given the capability "cancel_subscription"
        And the capability "switch_delivery_method"
        And the capability "check_order_status"
        And the capability "check_balance"
        And a canned response, "I understand that you wish to reduce carbon emissions. I recommend using {{generative.low_emission_shipping}} to achieve that." 
        And a canned response, "I apologize, but I have no information about {{generative.unanswerable_request}}."
        And a canned response, "I can help you switch the delivery method for your order. We offer we offer UPS, FEDEX, or private courier. Unfortunately though, I have no information regarding {{generative.unsupported_question}}?"
        And a canned response, "I can check the status of your order for you."
        And a canned response, "Our company is committed to eco-friendly practices, and all our shipping options are low-emission."
        And a customer message, "Hey, I want help checking if my order has been shipped"
        And an agent message, "Hi there! It looks like it is still awaiting shipment at our warehouse. Would you like any help or information regarding your order?"
        And a customer message, "I was wondering if it can be shipped using a service that has low carbon emissions"
        When processing is triggered
        Then a single message event is emitted
        And the message contains that the agent has no information regarding the carbon emissions of the different shipping services

    Scenario: Agent offers multiple capabilities when it is not clear which is best (strict canned response)
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
        And a canned response, "I can help you with that, by checking the following things regarding your order: {{generative.services_for_order}}"
        And a canned response, "I can assist you with your account, such as checking your balance or resetting your password."
        And a canned response, "Can you please provide the order number?"
        And a customer message, "Hi, I'm looking for help regarding an existing order"
        When processing is triggered
        Then a single message event is emitted
        And the message contains that the agent can help regarding checking an order's status and location. It may potentially ask about changing its delivery method - doing so is ok.

    Scenario: Agent doesnt offer capability thats forbidden by a guideline (strict canned response)
        Given a customer named "Mo"
        And an empty session with "Mo"
        And a context variable "age" set to "18" for "Mo"
        And the capability "offer_loan"
        And the capability "cancel_subscription"
        And a guideline to do not offer loans when the age of the customer is under 21
        And a canned response, "To help you increase your balance and reduce spending, I can offer you a loan."
        And a canned response, "To increase your balance and reduce spending, I can help you cancel subscriptions or offer you a loan."
        And a canned response, "To help you manage your finances, I can assist you with canceling your subscriptions."
        And a customer message, "Hey, I'm looking for ways to increase my balance and reduce spending"
        When processing is triggered
        Then a single message event is emitted
        And the message contains that the customer can cancel subscriptions
        And the message contains no offering of a loan

    Scenario: Agent mentions capability a guideline deems it relevant (strict canned response)
        Given a customer named "Mo"
        And an empty session with "Mo"
        And a context variable "age" set to "23" for "Mo"
        And the capability "offer_loan"
        And the capability "cancel_subscription"
        And a guideline to do not offer loans when the age of the customer is under 21
        And a canned response, "I can help you reduce spending by canceling subscriptions. For increasing your balance, I can offer you a loan. What would you like to do?"
        And a canned response, "I can help you by canceling your subscriptions."
        And a canned response, "I see you are 23. Would you like a loan?"
        And a canned response, "I am not able to help with that request."
        And a customer message, "Hey, I'm looking for ways to increase my balance and reduce spending"
        When processing is triggered
        Then a single message event is emitted
        And the message contains that the customer can cancel subscriptions
        And the message contains that the customer can take a loan

    Scenario: Agent doesnt mention capability that is forbidden by its description (strict canned response)
        Given a customer named "Mo"
        And an empty session with "Mo"
        And a context variable "age" set to "18" for "Mo"
        And the capability "offer_loan_no_minors_in_description"
        And the capability "cancel_subscription"
        And a guideline to do not offer loans when the age of the customer is under 21
        And a canned response, "To increase your balance, I can offer you a loan. To reduce spending, you can cancel subscriptions."
        And a canned response, "I can help you reduce your spending by canceling any active subscriptions."
        And a canned response, "I can offer you a loan."
        And a customer message, "Hey, I'm looking for ways to increase my balance and reduce spending"
        When processing is triggered
        Then a single message event is emitted
        And the message contains that the customer can cancel subscriptions
        And the message contains no offering of a loan

    Scenario: Agent chooses correct capability for current journey step (strict canned response)
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
        And a canned response, "Thank you, {{generative.name}}. To help, I can offer you a loan or we can decrease your credit limit. Which would you prefer?"
        And a canned response, "Thank you, {{generative.name}}. Can you please provide me with your account number?"
        And a canned response, "Thank you, {{generative.name}}. To help, I can offer you a loan."
        And a canned response, "Thank you, {{generative.name}}. we can decrease your credit limit"
        And a canned response, "Is there anything else I can help you with today?"
        When processing is triggered
        Then a single message event is emitted
        And the message contains either help regarding decreasing credit limits, an offering of a loan, or both 

    Scenario: Agent doesnt mention capabilities relevant for later journey steps (strict canned response)
        Given the journey called "Decrease Spending Journey"
        And the capability "offer_loan"
        And the capability "decrease_limit"
        And the capability "check_order_status"
        And the capability "check_order_location"
        And the capability "check_balance"
        And a canned response, "Thank you, . To help, I can offer you {{generative.suggestion}}"
        And a canned response, "Understood. Can you please provide your full name?"
        And a canned response, "Understood. Can you please provide your account number?"
        And a customer message, "Hey, I'm looking for ways to increase my balance and reduce spending"
        When processing is triggered
        Then a single message event is emitted
        And the message contains asking the customer for account number
        And the message contains no offering of a loan or credit limit decrease, unless it's discussed as something that will be done at a later staged

    Scenario: Agent uses glossary term to understand capabilities (strict canned response)
        Given the capability "reset_router"
        And the term "PDMM" defined as a highly technical term for performing actions on a router without having physical access to it. Known only by specialists with technical knowledge regarding internet protocols.
        And a canned response, "I see your router is not working. I can perform a PDMM to fix it."
        And a canned response, "I understand you're having trouble with your router. I can try to reset it from our end, so you won't have to do anything. Would that be okay?"
        And a canned response, "To fix your router, please unplug it from the power source for 60 seconds."
        And a canned response, "I'm sorry, I am not equipped to handle issues with routers."
        And a customer message, "My router is not working... Help me.... I barely know how to use a computer. Use simple language please."
        When processing is triggered
        Then a single message event is emitted
        And the message contains a suggestion to reset the router
        And the message contains either no mention of PDMM, or mentioning it while explaining that it means having no physical access to the router