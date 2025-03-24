Feature: Composited Assembly
    Background:
        Given the alpha engine
        And an agent
        And that the agent uses the composited_assembly message composition mode
        And an empty session

    Scenario: The agent has no option to greet the customer (composited assembly)
        Given a guideline to greet with 'Howdy' when the session starts
        And a fragment, "Your account balance is {balance}"
        When processing is triggered
        Then a no-match message is emitted

    Scenario: The agent explains it cannot help the customer (composited assembly)
        Given a guideline to talk about savings options when the customer asks how to save money
        And a customer message, "Man it's hard to make ends meet. Do you have any advice?"
        And a fragment, "Your account balance is {balance}"
        And a fragment, "I cannot help with this inquiry"
        When processing is triggered
        Then a single message event is emitted
        And the message contains the text "I cannot help with this inquiry."

    Scenario: Responding based on data the user is providing (composited assembly)
        Given a customer message, "I say that a banana is green, and an apple is purple. What did I say was the color of a banana?"
        And a fragment, "Sorry"
        And a fragment, "I do not know"
        And a fragment, "the answer is {answer}"
        When messages are emitted
        Then the message doesn't contain the text "Sorry"
        And the message doesn't contain the text "I do not know"
        And the message contains the text "The answer is green"

    Scenario: Assemble a message out of many fragments (composited assembly)
        Given a customer message, "Please tell me a made up story about the origin of bananas. Whatever you imagine."
        And a fragment, "I'd love to"
        And a fragment, "a"
        And a fragment, "there {linking_verb}"
        And a fragment, "Of course"
        And a fragment, "it grew into {something}"
        And a fragment, "What if something did"
        And a fragment, "Let me stop you"
        And a fragment, "Right there"
        And a fragment, "Just around the corner"
        And a fragment, "tell you"
        And a fragment, "You wouldn't believe it"
        And a fragment, "If only I had known"
        And a fragment, "Against all odds"
        And a fragment, "Let's be honest"
        And a fragment, "For what it's worth"
        And a fragment, "In the blink of an eye"
        And a fragment, "So far, so good"
        And a fragment, "This changes everything"
        And a fragment, "and"
        And a fragment, "I see what you mean"
        And a fragment, "Once upon a time"
        And a fragment, "That reminds me"
        And a fragment, "No way out"
        And a fragment, "As luck would have it"
        And a fragment, "Hold on a second"
        And a fragment, "a seed of {object}"
        And a fragment, "The best is yet to come"
        And a fragment, "It is what it is"
        And a fragment, "In the grand scheme of things"
        And a fragment, "You're not wrong"
        And a fragment, "I'll take your word for it"
        And a fragment, "One step at a time"
        And a fragment, "On second thought"
        And a fragment, "Against my better judgment"
        And a fragment, "It goes without saying"
        And a fragment, "At the end of the day"
        And a fragment, "story about {something}"
        And a fragment, "Just a matter of time"
        And a fragment, "Take it or leave it"
        And a fragment, "Not in a million years"
        When messages are emitted
        Then the message contains a story about bananas

    Scenario: Message assembler filters out tool-provided qualification data that doesn't match tool-provided fragments
        Given a guideline "retrieve_qualification_info" to retrieve qualification requirements when asked about educational qualifications
        And the tool "get_qualification_info"
        And an association between "retrieve_qualification_info" and "get_qualification_info"
        And a customer message, "What are the education requirements for the position?"
        And a fragment, "Educational qualifications are:"
        And a fragment, "The minimum requirements include"
        And a fragment, "For this position"
        And a fragment, "we require"
        And a fragment, "Additionally"
        And a fragment, "while not required"
        And a fragment, "would be considered a plus"
        And a fragment, "In terms of experience"
        And a fragment, "we're looking for"
        And a fragment, "Key technical requirements include"
        And a fragment, "proficiency in"
        And a fragment, "expertise with"
        And a fragment, "demonstrated ability in"
        When processing is triggered
        Then a single tool calls event is emitted
        And a single message event is emitted
        And the tool calls event contains 1 tool call(s)
        And the tool calls event contains a call to "get_qualification_info"
        the tool call "get_qualification_info" contains fragments
        And the message contains the text "Bachelor's degree"
        And the message doesn't contain the text "3+ years working with cloud platforms"
        And the message doesn't contain the text "Experience leading technical teams"
        And the message doesn't contain the text "Strong communication abilities"

    Scenario: Message assembler selects only fragments that contain the qualification data from the tool
        Given a guideline "retrieve_qualification_info" to retrieve qualification requirements when asked about educational qualifications
        And the tool "get_minimum_qualification_info"
        And an association between "retrieve_qualification_info" and "get_minimum_qualification_info"
        And a customer message, "What are the education requirements for the position?"
        And a fragment, "Educational qualifications are:"
        And a fragment, "The minimum requirements include"
        And a fragment, "For this position"
        And a fragment, "we require"
        And a fragment, "Additionally"
        And a fragment, "while not required"
        And a fragment, "would be considered a plus"
        And a fragment, "In terms of experience"
        And a fragment, "we're looking for"
        And a fragment, "Key technical requirements include"
        And a fragment, "proficiency in"
        And a fragment, "expertise with"
        And a fragment, "demonstrated ability in"
        When processing is triggered
        Then a single tool calls event is emitted
        And the tool calls event contains 1 tool call(s)
        And the tool calls event contains a call to "get_minimum_qualification_info"
        the tool call "get_minimum_qualification_info" contains fragments
        And a single message event is emitted
        And the message uses the fragment "5+ years development"
        And the message doesn't use the fragment "3+ years working with cloud platforms"
        And the message doesn't use the fragment "Master degree"
