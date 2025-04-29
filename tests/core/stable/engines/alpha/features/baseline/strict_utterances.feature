Feature: Strict Utterance
    Background:
        Given the alpha engine
        And an agent
        And that the agent uses the strict_utterance message composition mode
        And an empty session

    Scenario: The agent has no option to greet the customer (strict utterance)
        Given a guideline to greet with 'Howdy' when the session starts
        And an utterance, "Your account balance is {{balance}}"
        When processing is triggered
        Then a no-match message is emitted

    Scenario: The agent explains it cannot help the customer (strict utterance)
        Given a guideline to talk about savings options when the customer asks how to save money
        And a customer message, "Man it's hard to make ends meet. Do you have any advice?"
        And an utterance, "Your account balance is {{balance}}"
        When processing is triggered
        Then a single message event is emitted
        Then a no-match message is emitted

    Scenario: Adherence to guidelines without fabricating responses (strict utterance)
        Given a guideline "account_related_questions" to respond to the best of your knowledge when customers inquire about their account
        And a customer message, "What's my account balance?"
        And that the "account_related_questions" guideline is matched with a priority of 10 because "Customer inquired about their account balance."
        And an utterance, "Your account balance is {{balance}}"
        When messages are emitted
        Then a no-match message is emitted

    Scenario: Responding based on data the user is providing (strict utterance)
        Given a customer message, "I say that a banana is green, and an apple is purple. What did I say was the color of a banana?"
        And an utterance, "Sorry, I do not know"
        And an utterance, "the answer is {{generative.answer}}"
        When messages are emitted
        Then the message doesn't contain the text "Sorry"
        And the message contains the text "the answer is green"

    Scenario: Filling out fields from tool results (strict utterance)
        Given a guideline "retrieve_qualification_info" to explain qualification criteria when asked about position qualifications
        And the tool "get_qualification_info"
        And an association between "retrieve_qualification_info" and "get_qualification_info"
        And a customer message, "What are the requirements for the developer position?"
        And an utterance, "The requirement is {{qualification_info}}."
        When processing is triggered
        Then a single message event is emitted
        And the message contains the text "The requirement is 5+ years of experience."

    Scenario: Uttering agent and customer names (strict utterance)
        Given an agent named "Bozo" whose job is to sell pizza
        And that the agent uses the strict_utterance message composition mode
        And a customer named "Georgie Boy"
        And an empty session with "Georgie Boy"
        And a customer message, "What is your name?"
        And an utterance, "My name is {{std.agent.name}}, and you are {{std.customer.name}}."
        When messages are emitted
        Then a single message event is emitted
        And the message contains the text "My name is Bozo, and you are Georgie Boy."

    Scenario: Uttering context variables (strict utterance)
        Given a customer named "Georgie Boy"
        And a context variable "subscription_plan" set to "business" for "Georgie Boy"
        And an empty session with "Georgie Boy"
        And a customer message, "What plan am I on exactly?"
        And an utterance, "You're on the {{std.variables.subscription_plan|capitalize}} plan."
        When processing is triggered
        Then a single message event is emitted
        And the message contains the text "You're on the Business plan."
