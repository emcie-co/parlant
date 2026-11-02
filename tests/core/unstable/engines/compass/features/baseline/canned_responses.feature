Feature: Canned responses
    Background:
        Given the engine
        And an agent
        And an empty session

    Scenario: The agent fills multiple fields in a veterinary appointment system
        Given an agent whose job is to schedule veterinary appointments and provide pet care information
        And a customer named "Joanna"
        And a context variable "next_available_date" set to "May 22" for "Joanna"
        And a context variable "vet_name" set to "Dr. Happypaws" for "Joanna"
        And a context variable "clinic_address" set to "155 Pawprint Lane" for "Joanna"
        And an empty session with "Joanna"
        And a policy to provide the next available appointment details when a customer requests a checkup for their pet
        And a customer message, "I need to schedule a routine checkup for my dog Max. He's a 5-year-old golden retriever."
        And a canned response, "Our next available appointment for {{generative.pet_name}} with {{generative.vet_name}} is on the {{generative.appointment_date}} at our clinic located at {{generative.clinic_address}}. For a {{generative.pet_age}}-year-old {{generative.pet_breed}}, we recommend {{generative.recommended_services}}."
        And a canned response, "We're fully booked at the moment. Please call back next week."
        And a canned response, "What symptoms is your dog experiencing?"
        And a canned response, "Would you prefer a morning or afternoon appointment?"
        And a canned response, "Our next available appointment is next Tuesday. Does that work for you?"
        When processing is triggered
        Then a single message event is emitted
        And the message contains "Max" in the {pet_name} field
        And the message contains "Dr. Happypaws" in the {vet_name} field
        And the message contains "May 22" in the {appointment_date} field
        And the message contains "155 Pawprint Lane" in the {clinic_address} field
        And the message contains "5" in the {pet_age} field
        And the message contains "golden retriever" in the {pet_breed} field
        And the message contains appropriate veterinary services for a middle-aged dog in the {recommended_services} field
