Feature: Error Handling in Alpha Engine
    Scenario: Failure to process a message emits an error status
        Given the alpha engine
        And a session with a single customer message
        And a faulty message production mechanism
        When processing is triggered
        Then a status event is emitted, encountering an error while processing event
