Feature: Terminology Integration
    Background:
        Given the alpha engine
        And an agent
        And an empty session

    Scenario Outline: The agent explains an ambiguous term
        Given the term "<TERM_NAME>" defined as <TERM_DESCRIPTION>
        And a user message, "<USER_MESSAGE>"
        When processing is triggered
        Then a single message event is produced
        And the message contains an explanation of <TERM_DESCRIPTION>
        Examples:
            | TERM_NAME   | USER_MESSAGE           | TERM_DESCRIPTION                      |
            | token       | What is a token?       | digital token                         |
            | wallet      | What is a wallet?      | digital wallet                        |
            | mining      | What is mining?        | cryptocurrency mining                 |
            | private key | What is a private key? | private key in cryptocurrency         |
            | gas         | What is gas?           | gas fees in Ethereum                  |

    Scenario: The agent explains multiple terms without explicit cryptocurrency context
        Given the term "walnut" defined as a name of an altcoin
        And the term "private key" defined as private key in cryptocurrency
        And a guideline to say "Keep your private key secure" when the user asks how to protect their walnuts
        And a user message, "How do you keep walnuts secure?"
        When processing is triggered
        Then a single message event is produced
        And the message contains an instruction to keep the private key secure

    Scenario: The agent follows a guideline based on term definition
        Given the term "private key" defined as private key in cryptocurrency
        And the term "walnut" defined as a name of an altcoin
        And a guideline to say "Keep your private key secure" when the user asks how to protect their assets
        And a user message, "How do I protect my walnuts?"
        When processing is triggered
        Then a single message event is produced
        And the message contains an instruction to keep the private key secure

    Scenario: The agent responds with a term retrieved from guideline content
        Given 50 random terms related to technology companies
        And the term "leaf" defined as a cryptocurrency wallet for walnut cryptocoins
        And a guideline to explain what a leaf is when the user asks about IBM
        And a user message, "Tell me about IBM"
        When processing is triggered
        Then a single message event is produced
        And the message contains an explanation about a cryptocurrency wallet for walnut cryptocoins

    Scenario: The agent responds with a term retrieved from tool content
        Given 50 random terms related to technology companies
        And the term "leaf" defined as a cryptocurrency wallet for walnut cryptocoins
        And a guideline "explain_about_terry" to get Terry's SaaS solutions explanation when the user asks about Terry solutions
        And the tool "terry_definition"
        And an association between "explain_about_terry" and "terry_definition"
        And a user message, "Tell me about Terry SaaS solutions"
        When processing is triggered
        Then a single message event is produced
        And the message contains an explanation about a cryptocurrency wallet for walnut cryptocoins