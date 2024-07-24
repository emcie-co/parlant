Feature: Terminology Integration
    Background:
        Given the alpha engine
        And an agent
        And an empty session

    Scenario Outline: The agent explains an ambiguous term
        Given a term of <TERM>
        And a user message, "<USER_MESSAGE>"
        When processing is triggered
        Then a single message event is produced
        And the message contains an explanation of a <TERM_DESCRIPTION>
        Examples:
            | TERM        | USER_MESSAGE           | TERM_DESCRIPTION                      |
            | token       | What is a token?       | digital token                         |
            | wallet      | What is a wallet?      | digital wallet                        |
            | mining      | What is mining?        | cryptocurrency mining                 |
            | private key | What is a private key? | private key in cryptocurrency         |
            | gas         | What is gas?           | gas fees in Ethereum                  |

    Scenario: The agent explains multiple terms without explicit cryptocurrency context
        Given a term of walnut
        And a term of private key
        And a guideline to say "Keep your private key secure" when the user asks how to protect their walnuts
        And a user message, "How do you keep a walnuts secure?"
        When processing is triggered
        Then a single message event is produced
        And the message contains instruction for keep the private key secure

    Scenario: The agent follows guideline based on term definition
        Given a term of private key
        And a term of walnut
        And a guideline to say "Keep your private key secure" when the user asks how to protect their assets
        And a user message, "How do I protect my walnuts?"
        When processing is triggered
        Then a single message event is produced
        And the message contains instruction for keep the private key secure