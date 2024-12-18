Feature: Tools
    Background:
        Given the alpha engine
        And an agent
        And an empty session

    Scenario: Tool called again by context after customer response
        Given a guideline "retrieve_account_information", to retrieve account information when customers inquire about account-related information
        And the tool "get_account_balance"
        And an association between "retrieve_account_information" and "get_account_balance"
        And a customer message, "What is the balance of Larry David's account?"
        And a tool event with data, [{ "tool_calls": { "tool_id": "local:get_account_balance", "parameters": { "account_name": "Larry David"}, "result": { "data": 451000000, "metadata": {} }}}]
        And an agent message, "Larry David currently has 451 million dollars."
        And a customer message, "And what about now?"
        When processing is triggered
        Then a single tool calls event is emitted
        And the tool calls event contains 1 tool call(s)
        And the tool calls event contains a call to "get_account_balance" with Larry David's current balance