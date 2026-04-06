Feature: Context Variables
    Background:
        Given the alpha engine
        And an agent
        And that the agent uses the canned_fluid message composition mode
        And an empty session


    Scenario: The agent does not acknowledge values from other customers when the customer lacks a value
        Given a customer named "Keyleth"
        And a customer named "Vax"
        And a context variable "Power" set to "Stealth" for "Vax"
        And an empty session with "Keyleth"
        And a customer message, "Do you know my power?"
        When processing is triggered
        Then a single message event is emitted
        And the message contains no mention of the customer’s specific power

    Scenario: The agent selects variables that are specifically attached to the relevant customer
        Given a customer named "Keyleth"
        And a customer named "Vax"
        And a context variable "Power" set to "Magic" for "Keyleth"
        And a context variable "Power" set to "Stealth" for "Vax"
        And an empty session with "Vax"
        And a customer message, "Do you know my power?"
        When processing is triggered
        Then a single message event is emitted
        And the message mentions to the customer that their power is Stealth

    Scenario: The agent responds according to the updated value from the tool after the freshness rules are met
        Given a customer named "Keyleth"
        And a context variable "UserStamina" set to "80.0" for "Keyleth"
        And the context variable "UserStamina" has freshness rules of "0,15,30,45 * * * *"
        And the tool "get_keyleth_stamina"
        And the context variable "UserStamina" is connected to the tool "get_keyleth_stamina"
        And an empty session with "Keyleth"
        And a customer message, "What is my stamina?"
        When processing is triggered
        Then a single message event is emitted
        And the message mentions that stamina is 100

