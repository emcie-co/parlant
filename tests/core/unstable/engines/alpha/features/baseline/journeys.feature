Feature: Journeys
    Background:
        Given the alpha engine
        And an agent
        And an empty session

    Scenario: Simple journey is followed to inform decision
        Given a guideline "recommend_pizza" to recommend either tomato, mushrooms or pepperoni when the customer asks for topping recommendations
        And a journey titled Vegetarian Customers to Be aware that the customer is vegetarian. Only discuss vegetarian options with them. when the customer has a name that begins with R
        And a customer message, "Hey, there. How are you?"
        And an agent message, "I'm doing alright, thank you! What's your name?"
        And a customer message, "Rajon, have we spoken before? I want one large pie but I'm not sure which topping to get, what do you recommend?"
        When processing is triggered
        Then a single message event is emitted
        And the message contains recommendations for either mushrooms or tomatoes, but not pepperoni

    Scenario: Multistep journey invokes tool calls correctly
        Given a journey titled Reset Password Journey to follow these steps to reset a customers password: 1. ask for their account name 2. ask for their email or phone number 3. Wish them a good day and only proceed if they wish one back to you. Otherwise abort. 3. use the tool reset_password with the provided information 4. report the result to the customer when the customer wants to reset their password
        And the tool "reset_password"
        And a guideline "reset_password_guideline" to reset the customer's password using the associated tool when the customer requested to reset their password
        And an association between "reset_password_guideline" and "reset_password"
        And a customer message, "I want to reset my password"
        And an agent message, "I can help you do just that. What's your username?"
        And a customer message, "it's leonardo_barbosa_1982"
        And an agent message, "Great! And what's the account's associated email address or phone number?"
        And a customer message, "the email is leonardobarbosa@gmail.br"
        And an agent message, "Got it. Before proceeding to reset your password, I wanted to wish you a good day"
        And a customer message, "Thank you! Have a great day as well!"
        And that the "reset_password_guideline" guideline was matched in the previous iteration
        When detection and processing are triggered
        Then a single tool calls event is emitted
        And the tool calls event contains 1 tool call(s)
        And the tool calls event contains the tool reset password with username leonardo_barbosa_1982 and email leonardobarbosa@gmail.br 
        And a single message event is emitted
        And the message contains that the password was reset and an email with instructions was sent to the customer
