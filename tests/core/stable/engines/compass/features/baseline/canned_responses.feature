Feature: Canned responses
    Background:
        Given the engine
        And an agent
        And an empty session

    Scenario: Filling out fields from tool results
        Given a policy "retrieve_qualification_info" to explain qualification criteria when asked about position qualifications
        And the tool "get_qualification_info"
        And an association between "retrieve_qualification_info" and "get_qualification_info"
        And a customer message, "What are the requirements for the developer position?"
        And a canned response, "In terms of years of experience, the requirement is {{qualification_info}}."
        When processing is triggered
        Then a single message event is emitted
        # Adapted: canned responses guide wording rather than dictate it; the
        # substituted field value and the substance must come through.
        And the message contains the text "5+ years"
        And the message mentions the experience requirement for the position

    Scenario: Uttering agent and customer names
        Given an agent named "Bozo" whose job is to sell pizza
        And a customer named "Georgie Boy"
        And an empty session with "Georgie Boy"
        And a customer message, "What is your name?"
        And a canned response, "My name is {{std.agent.name}}, and you are {{std.customer.name}}."
        When messages are emitted
        Then a single message event is emitted
        # Adapted: assert the substituted names rather than the verbatim sentence.
        And the message contains the text "Bozo"
        And the message contains the text "Georgie Boy"

    Scenario: Uttering context variables
        Given a customer named "Georgie Boy"
        And a context variable "subscription_plan" set to "business" for "Georgie Boy"
        And an empty session with "Georgie Boy"
        And a customer message, "What plan am I on exactly?"
        And a canned response, "You're on the {{std.variables.subscription_plan|capitalize}} plan."
        When processing is triggered
        Then a single message event is emitted
        # Adapted: assert the substituted plan name rather than the verbatim sentence.
        And the message contains the text "Business"
        And the message mentions which plan the customer is on

    Scenario: Agent chooses canned response which uses glossary
        Given an agent whose job is to assist customers with specialized banking products
        And the term "Velcron Account" defined as A high-security digital banking account with multi-layered authentication that offers enhanced privacy features
        And the term "Quandrex Protocol" defined as The security verification process used for high-value transactions that require additional identity confirmation
        And a policy to recommend a Velcron Account and explain the Quandrex Protocol when customers ask about secure banking options
        And a customer message, "I'm looking for the most secure type of account for my business. What do you recommend?"
        And a canned response, "I recommend our premium business accounts, which feature advanced security measures."
        And a canned response, "Our standard security protocols are sufficient for most business needs. Would you like me to explain our different account tiers?"
        And a canned response, "For your business security needs, I recommend our Velcron Account, which features multi-layered authentication and enhanced privacy features. All high-value transactions will be protected by our Quandrex Protocol, providing additional identity verification."
        And a canned response, "You should consider our platinum business account with two-factor authentication and fraud monitoring."
        And a canned response, "We offer several secure banking options with varying levels of protection. What specific security concerns do you have?"
        When processing is triggered
        Then a single message event is emitted
        And the message contains the terms 'Velcron Account' and 'Quandrex Protocol'

    Scenario: The agent selects response based on customer's subscription tier context variable
        Given an agent whose job is to provide technical support for cloud-based molecular modeling software
        And a group "Enterprise"
        And a group "Standard"
        And a customer named "Joanna"
        And an empty session with "Joanna"
        And a customer grouped as "Enterprise"
        And a context variable "api_access" set to "Unlimited" for the group "Enterprise"
        And a context variable "api_access" set to "Basic" for the group "Standard"
        And a policy to mention dedicated support channels and unlimited API access when responding to Enterprise customers with technical issues
        And a customer message, "I'm having trouble with the protein folding simulation API. Is there a limit to how many calls I can make?"
        And a canned response, "There is a limit of 100 API calls per day on your current plan. Would you like to upgrade for more access?"
        And a canned response, "As an Enterprise subscriber, you have Unlimited API access for your protein folding simulations. I can connect you with your dedicated support specialist to resolve any technical issues you're experiencing. Would you prefer a video call or screen sharing session?"
        And a canned response, "Please try resetting your API key in the account settings and clearing your cache."
        And a canned response, "We're experiencing some server issues at the moment. Please try again in an hour."
        And a canned response, "The protein folding simulation has certain parameter limitations. Could you share more details about your specific configuration?"
        When processing is triggered
        Then a single message event is emitted
        And the message contains information about unlimited API access and dedicated support options for Enterprise customers

    Scenario: The agent correctly fills in numeric field
        Given an agent whose job is to process orders for a specialty yarn and fabric shop
        And a customer named "Joanna"
        And a policy to check stock levels in the context variables when a customer makes a specific order
        And an empty session with "Joanna"
        And a context variable "Merino Wool Skein inventory count" set to "37" for "Joanna"
        And a context variable "Alpaca Blend Yarn inventory count" set to "12" for "Joanna"
        And a policy to include the current inventory count when confirming orders for yarn products
        And a customer message, "I'd like to order 5 skeins of your Merino Wool, please."
        And a canned response, "I've added {{generative.quantity}} skeins of Merino Wool to your order. We currently have {{generative.inventory_count}} in stock."
        And a canned response, "We're currently out of that item. Would you like to place a backorder?"
        And a canned response, "Would you like to view our other yarn options instead?"
        When processing is triggered
        Then a single message event is emitted
        # Adapted: assert the substituted quantities rather than near-verbatim text.
        And the message contains the text "5 skeins"
        And the message contains the text "37"
        And the message mentions that Merino Wool was added to the order and how many skeins are in stock

    Scenario: The agent adheres to policies in field extraction
        Given an agent whose job is to provide account information
        And a customer named "Alex Smith"
        And an empty session with "Alex Smith"
        And a context variable "account_balance" set to "1243.67" for "Alex Smith"
        And a policy to always round monetary amounts to the nearest dollar when responding to balance inquiries
        And a customer message, "What's my current account balance?"
        And a canned response, "Your current balance is ${{generative.account_balance}} as of today."
        And a canned response, "I apologize but I don't have this information available"
        When processing is triggered
        Then a single message event is emitted
        # Adapted: assert the rounded value (in any number formatting) rather
        # than the verbatim sentence.
        And the message mentions a balance of 1244 dollars (in any formatting), and not the unrounded 1243.67

    Scenario: Follow up canned response is selected when relevant
        Given an agent whose job is to schedule automatic vaccum cleaning services using robots
        And a policy to ensure that no pets and no children are in the house when a customer asks to schedule a deep-clean in a residential area
        And a customer message, "I need a deep-clean next Wednesday"
        And an agent message, "Great! I can schedule a deep-clean for you. Is it at the location of your last deep-clean?"
        And a customer message, "Yes"
        And an agent message, "Just to confirm, the location is your parents house, at 1414 2nd Avenue, correct?"
        And a customer message, "Yes"
        And a canned response, "Great! I'll schedule a deep-clean at {{generative.location}} at {{generative.desired_date}}."
        And a canned response, "For safety reasons, please ensure that no children are present at the house during the {{generative.service_type}}"
        And a canned response, "Unfortunately, I lack the information to complete this booking"
        And a canned response, "For safety reasons, please ensure that no pets are present at the house during the {{generative.service_type}}"
        And a canned response, "For safety reasons, please ensure that no one is at the house during the {{generative.service_type}}"
        When processing is triggered
        # Adapted: alpha emitted each follow-up canned response as its own
        # message; compass may compose the same substance into fewer messages.
        Then at least one message event is emitted
        And at least one message mentions that no pets should be present in the house during the service
        And at least one message mentions that no children should be present in the house during the service

    Scenario: Follow up canned response is selected based on unfulfilled policy
        Given an agent whose job is to book taxi rides
        And a policy to tell the customer to wait at curbside when a taxi booking is confirmed
        And a policy to confirm the taxi booking details from the book_taxi tool when a taxi booking was just confirmed
        And a customer message, "Can I get a taxi from my home to work in 20 minutes? You got my details, right?"
        And an agent message, "I have your home and work address"
        And an agent message, "Do you prefer paying by cash or credit"
        And a customer message, "Credit"
        And a tool event with data, {"tool_calls": [{"tool_id": "built-in:book_taxi", "arguments": {"departure": "customer-home", "arrival": "customer-work", "time": "12:00:00"}, "result": {"data": "ORDER STATUS: Confirmed, awaiting pick up"}}]}
        And a canned response, "Yes please"
        And a canned response, "Let me check that for you"
        And a canned response, "Your order is confirmed! A driver will be dispatched to {{generative.departure_address}} at the provided time"
        And a canned response, "How many passengers are in your party?"
        And a canned response, "Your driver will meet you at the curbside of your pickup location. Please be ready at the curb when they arrive"
        And a canned response, "Your order cannot be processed at this time"
        When processing is triggered
        # Adapted: alpha emitted each follow-up canned response as its own
        # message; compass may compose the same substance into fewer messages.
        Then at least one message event is emitted
        # Adapted: assert substance rather than verbatim canned sentences.
        And at least one message mentions that the order is confirmed and a driver will be dispatched
        And at least one message mentions meeting the driver at the curbside

    Scenario: Follow up canned response which uses fields is selected when relevant
        Given an agent whose job is to process insurance claims for auto accidents
        And a policy to provide claim reference number and estimated processing time when a claim is successfully submitted
        And a customer message, "I was in a fender bender yesterday and need to file a claim"
        And an agent message, "I'm sorry to hear about your accident. Let me help you file a claim. What's your policy number?"
        And a customer message, "It's POL-789456"
        And an agent message, "Thank you. Can you describe what happened and provide the date and location?"
        And a customer message, "Yesterday at 3pm, someone rear-ended me at the intersection of Oak and Main Street"
        And a tool event with data, {"tool_calls": [{"tool_id": "built-in:file_claim", "arguments": {"policy_number": "POL-789456", "accident_type": "rear_end", "date": "2024-01-14", "location": "Oak and Main Street"}, "result": {"data": "CLAIM STATUS: Filed successfully. claim_number: CLM-2024-789456. estimated_time : 5-7 days"}}]}
        And a canned response, "Your claim has been successfully filed. Your reference number is {{generative.claim_number}}."
        And a canned response, "The estimated processing time is {{generative.processing_time}} business days."
        And a canned response, "You'll receive an email confirmation shortly with all the details."
        And a canned response, "A claims adjuster will contact you within 24 hours."
        When processing is triggered
        # Adapted: alpha emitted each follow-up canned response as its own
        # message; compass may compose the same substance into fewer messages.
        Then at least one message event is emitted
        # Adapted: assert the substituted reference number and substance.
        And at least one message contains the text "CLM-2024-789456"
        And at least one message mentions an estimated processing time of 5-7 days

    Scenario: The agent doesn't send highly similar follow up canned responses instead of one 1
        Given the procedure called "Book Hotel Journey"
        And a customer message, "I need to book a hotel."
        And an agent message, "Alright, tell me the hotel name so I can check it out for you."
        And a customer message, "The Marriott Downtown"
        And an agent message, "Perfect, now when are you looking to stay—check-in and check-out dates?"
        And a customer message, "From September 10th to 25th"
        And an agent message, "And how many people will be staying with you?"
        And a customer message, "Just me and my wife"
        And a canned response, "Do you have a preference—single, double, or maybe a suite?"
        And a canned response, "What kind of room are you looking for? Single, double, or something fancier?"
        When processing is triggered
        Then a single message event is emitted
        # Adapted from alpha: assert the substance (one room-type question,
        # not two near-identical ones) rather than a verbatim canned choice.
        And the message contains exactly one question, asking what kind of room the customer prefers (e.g. single, double, or a suite)

    # Nearly identical to the previous scenario. We previously saw cases where this fail when the previous did not

    Scenario: The agent doesn't send highly similar follow up canned responses instead of one 2
        Given the procedure called "Book Hotel Journey"
        And a customer message, "I need to book a hotel."
        And an agent message, "Alright, tell me the hotel name so I can check it out for you."
        And a customer message, "The Marriott Downtown"
        And an agent message, "Perfect, now when are you looking to stay—check-in and check-out dates?"
        And a customer message, "From September 10th to 25th"
        And an agent message, "And how many people will be staying with you?"
        And a customer message, "Just me and my awesome wifey"
        And a canned response, "Do you have a preference—single, double, or maybe a suite?"
        And a canned response, "What kind of room are you looking for? Single, double, or something fancier?"
        When processing is triggered
        Then a single message event is emitted
        # Adapted from alpha: assert the substance (one room-type question,
        # not two near-identical ones) rather than a verbatim canned choice.
        And the message contains exactly one question, asking what kind of room the customer prefers (e.g. single, double, or a suite)

    Scenario: The agent doesn't send highly similar follow up canned responses instead of one 3
        Given the procedure called "Book Hotel Journey"
        And an agent whose job is to You're a sympathetic, sarcastically funny agent that helps customers in their hotel booking needs
        And a customer message, "I need to book a hotel."
        And an agent message, "Alright, tell me the hotel name so I can check it out for you."
        And a customer message, "the marriot hotel"
        And a customer message, "it seems cannot find the check-in and check-out options on the booking page."
        And an agent message, "You can set your check-in and check-out dates using the date picker at the top right of the booking page."
        And an agent message, "What dates are you planning to check in and check out?"
        And a customer message, "15th to the 23rd of september"
        And an agent message, "How many guests should I book the room for?"
        And a customer message, "Just me and my bro"
        And a canned response, "The hotel is open for those dates. Want me to secure the booking now?"
        And a canned response, "Okay, let me check the availability for you—one sec."
        And a canned response, "What dates are you planning to check in and check out?"
        And a canned response, "What's your budget range for this stay?"
        And a canned response, "Do you want me to include extras like breakfast, parking, or gym access?"
        And a canned response, "And how many people will be staying with you?"
        And a canned response, "Alright, tell me the hotel name so I can check it out for you."
        And a canned response, "You can set your check-in and check-out dates using the date picker at the top right of the booking page."
        And a canned response, "Done and dusted. Your booking is confirmed—now the only thing left is to enjoy it."
        And a canned response, "Do you have a preference—single, double, or maybe a suite?"
        And a canned response, "Perfect, now when are you looking to stay—check-in and check-out dates?"
        And a canned response, "Perfect, I'll confirm the booking right away."
        And a canned response, "Got it! Which hotel are you thinking of staying at?"
        And a canned response, "Bad news—the hotel's booked up for those preferences. Do you want me to suggest alternatives?"
        And a canned response, "Got it. About how much are you planning to spend per night?"
        And a canned response, "I'll run a quick check on the hotel's availability with your preferences"
        And a canned response, "Great choice! I'll go ahead and book the hotel for you now."
        And a canned response, "How many guests should I book the room for?"
        And a canned response, "You can set your check-in and check-out dates using the date picker at the top right of the booking page."
        And a canned response, "Any special amenities you'd like me to look for—like breakfast, a pool, or parking?"
        And a canned response, "hat kind of room are you looking for? Single, double, or something fancier?"
        When processing is triggered
        Then a single message event is emitted
        # Adapted from alpha: assert the substance (one room-type question,
        # not two near-identical ones) rather than a verbatim canned choice.
        And the message contains exactly one question - either asking what kind of room the customer prefers, or clarifying which specific hotel or location they mean
