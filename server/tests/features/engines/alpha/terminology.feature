Feature: Terminology Integration
    Background:
        Given the alpha engine
        And an agent
        And an empty session

    Scenario Outline: The agent explains an ambiguous term
        Given the term "<TERM_NAME>" defined as <TERM_DESCRIPTION>
        And a user message, "<USER_MESSAGE>"
        When processing is triggered
        Then a single message event is emitted
        And the message contains an explanation of <TERM_NAME> as <TERM_DESCRIPTION>
        Examples:
            | TERM_NAME   | USER_MESSAGE           | TERM_DESCRIPTION                |
            | token       | What is a token?       | a digital token                 |
            | wallet      | What is a wallet?      | a digital wallet                |
            | mining      | What is mining?        | cryptocurrency mining           |
            | private key | What is a private key? | a private key in cryptocurrency |
            | gas         | What is gas?           | a type of fee in Ethereum       |

    Scenario: The agent follows a guideline that mentions a term by name
        Given the term "walnut" defined as the name of an altcoin
        And a guideline to say "Keep your private key secure" when the user asks how to protect their walnuts
        And a user message, "How do you keep walnuts secure?"
        When processing is triggered
        Then a single message event is emitted
        And the message contains an instruction to keep the private key secure

    Scenario: The agent follows a guideline that refers to a term's definition
        Given the term "walnut" defined as the name of an altcoin
        And a guideline to say "Keep your private key secure" when the user asks how to protect their financial assets
        And a user message, "How do I protect my walnuts?"
        When processing is triggered
        Then a single message event is emitted
        And the message contains an instruction to keep the private key secure

    Scenario: The agent responds with a term retrieved from guideline content
        Given 50 random terms related to technology companies
        And the term "leaf" defined as a cryptocurrency wallet for walnut cryptocoins
        And a guideline to explain what a leaf is when the user asks about IBM
        And a user message, "Tell me about IBM"
        When processing is triggered
        Then a single message event is emitted
        And the message contains an explanation of a leaf as a cryptocurrency wallet for walnut cryptocoins

    Scenario: The agent responds with a term retrieved from tool content
        Given 50 random terms related to technology companies
        And the term "leaf" defined as a cryptocurrency wallet for walnut cryptocoins
        And a guideline "explain_terry" to fully elaborate on Terry's offering when the user asks about Terry
        And the tool "get_terrys_offering"
        And an association between "explain_terry" and "get_terrys_offering"
        And a user message, "Tell me about Terry"
        When processing is triggered
        Then a single message event is emitted
        And the message contains an explanation about a cryptocurrency wallet for walnut cryptocoins
