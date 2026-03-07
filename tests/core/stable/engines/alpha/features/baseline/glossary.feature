Feature: Glossary
    Background:
        Given the alpha engine
        And an agent
        And that the agent uses the canned_fluid message composition mode
        And an empty session


    Scenario: The agent explains an ambiguous term token
        Given the term "token" defined as a digital token
        And a customer message, "What is a token?"
        When processing is triggered
        Then a single message event is emitted
        And the message contains that token is a digital token

    Scenario: The agent explains an ambiguous term wallet
        Given the term "wallet" defined as a digital wallet
        And a customer message, "What is a wallet?"
        When processing is triggered
        Then a single message event is emitted
        And the message contains that wallet is a digital wallet

    Scenario: The agent explains an ambiguous term mining
        Given the term "mining" defined as cryptocurrency mining
        And a customer message, "What is mining?"
        When processing is triggered
        Then a single message event is emitted
        And the message contains that mining means cryptocurrency mining

    Scenario: The agent explains an ambiguous term private key
        Given the term "private key" defined as a private key in cryptocurrency
        And a customer message, "What is a private key?"
        When processing is triggered
        Then a single message event is emitted
        And the message contains that private key means a private key in cryptocurrency

    Scenario: The agent explains an ambiguous term gas
        Given the term "gas" defined as a type of fee in Ethereum
        And a customer message, "What is gas?"
        When processing is triggered
        Then a single message event is emitted
        And the message contains that gas means a type of fee in Ethereum

    Scenario: The agent follows a guideline that mentions a term by name
        Given the term "walnut" defined as the name of an altcoin
        And a guideline to say "Keep your private key secure" when the customer asks how to protect their walnuts
        And a customer message, "How do you keep walnuts secure?"
        When processing is triggered
        Then a single message event is emitted
        And the message contains an instruction to keep the private key secure

    Scenario: The agent follows a guideline that refers to a term's definition
        Given the term "walnut" defined as the name of an altcoin
        And a guideline to say "Keep your private key secure" when the customer asks how to protect their financial assets
        And a customer message, "How do I protect my walnuts?"
        When processing is triggered
        Then a single message event is emitted
        And the message contains an instruction to keep the private key secure

    Scenario: The agent responds with a term retrieved from guideline content
        Given 50 random terms related to technology companies
        And the term "leaf" defined as a cryptocurrency wallet for walnut cryptocoins
        And a guideline to explain what a leaf is when the customer asks about IBM
        And a customer message, "Tell me about IBM"
        When processing is triggered
        Then a single message event is emitted
        And the message contains that a leaf as a cryptocurrency wallet for walnut cryptocoins

    Scenario: The agent responds with a term retrieved from tool content
        Given 50 random terms related to technology companies
        And the term "leaf" defined as a cryptocurrency wallet for walnut cryptocoins
        And a guideline "explain_terry" to fully elaborate on Terry's offering when the customer asks about Terry
        And the tool "get_terrys_offering"
        And an association between "explain_terry" and "get_terrys_offering"
        And a customer message, "Tell me about Terry"
        When processing is triggered
        Then a single message event is emitted
        And the message contains an explanation about a cryptocurrency wallet for walnut cryptocoins

    Scenario: The agent explains term without exposing the glossary itself
        Given the term "token" defined as a digital token
        And a customer message, "what is a token?"
        When processing is triggered
        Then a single message event is emitted
        And the message contains an explanation about what a token is, without mentioning that it appears in a glossary