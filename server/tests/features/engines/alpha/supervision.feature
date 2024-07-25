Feature: Supervision
    Background:
        Given an empty session

    Scenario: Prioritizing freezing transactions over processing refunds
        Given a guideline "process_refund", to process refunds when a customer requests a refund
        And a guideline "freeze_transactions", to freeze all transactions when suspicious activity is detected on an account
        And a user message, "I believe there is an error with my last transaction. Can we process a refund?"
        And that the "freeze_transactions" guideline is proposed with a priority of 9 because "An error in the last transaction occurred, which is unusual and raises concerns."
        And that the "process_refund" guideline is proposed with a priority of 7 because "The customer explicitly requested a refund."
        When messages are produced
        Then the message should contain information that all transactions will be temporarily frozen and the refund will not be processed at this time.

    Scenario: Prioritizing transferring the upset customer to the manager over offering pizza
        Given a guideline "selling_pizza", to sell pizza when a new customer arrives
        And a guideline "customer_upset", to transfer the customer to the manager immediately when the customer is upset
        And a user message, "I came to eat some pizza, but the line was so long! I'm so upset!"
        And that the "selling_pizza" guideline is proposed with a priority of 7 because "The conversation was initiated, indicating a new customer has arrived."
        And that the "customer_upset" guideline is proposed with a priority of 8 because "The customer is visibly upset about the wait."
        When messages are produced
        Then the message should contain an apology and inform the customer about being transferred to the manager, explicitly omitting any offer to order pizza.

    Scenario: Adherence to guidelines without fabricating responses
        Given a guideline "account_related_questions", to respond to account-related questions about their account when users inquire about their account
        And a user message, "What's my account balance?"
        And that the "account_related_questions" guideline is proposed with a priority of 10 because "User inquired about their account balance."
        When messages are produced
        Then the message should contain information that the account balance is not available.
