# Copyright 2026 Emcie Co Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from datetime import date, datetime
from enum import Enum
import enum
import json
from typing import Optional


from parlant.core.tools import ToolResult


class Categories(Enum):
    GRAPHICSCARD = "Graphics Card"
    PROCESSOR = "Processor"
    STORAGE = "Storage"
    POWER_SUPPLY = "Power Supply"
    MOTHERBOARD = "Motherboard"
    MEMORY = "Memory"
    CASE = "Case"
    CPUCOOLER = "CPU Cooler"
    MONITOR = "Monitor"
    KEYBOARD = "Keyboard"
    MOUSE = "Mouse"
    HEADSET = "Headset"
    AUDIO = "Audio"
    COOLING = "Cooling"
    ACCESSORIES = "Accessories"
    LIGHTING = "Lighting"
    NETWORKING = "Networking"
    LAPTOP = "Laptop"


class ElectronicProductType(Enum):
    MONITOR = "Monitor"
    KEYBOARD = "Keyboard"
    MOUSE = "Mouse"
    HEADSET = "Headset"
    AUDIO = "Audio"
    LAPTOP = "Laptop"
    OTHER = "Other"


def get_available_drinks() -> ToolResult:
    return ToolResult(["Sprite", "Coca Cola"])


def get_available_toppings() -> ToolResult:
    return ToolResult(["Pepperoni", "Mushrooms", "Olives"])


def expert_answer(user_query: str) -> ToolResult:
    answers = {"Hey, where are your offices located?": "Our Offices located in Tel Aviv"}
    return ToolResult(answers[user_query])


class ProductType(Enum):
    DRINKS = "drinks"
    TOPPINGS = "toppings"


def get_available_product_by_type(product_type: ProductType = ProductType.DRINKS) -> ToolResult:
    if product_type == ProductType.DRINKS:
        return get_available_drinks()
    elif product_type == ProductType.TOPPINGS:
        return get_available_toppings()
    else:
        return ToolResult([])


def add(first_number: int, second_number: int) -> ToolResult:
    return ToolResult(first_number + second_number)


def multiply(first_number: int, second_number: int) -> ToolResult:
    return ToolResult(
        first_number * second_number,
        canned_responses=["asd"],
    )


def get_account_balance(account_name: str) -> ToolResult:
    balances = {
        "Jerry Seinfeld": 1000000000,
        "Larry David": 450000000,
        "John Smith": 100,
    }
    return ToolResult(balances.get(account_name, -555))


def get_account_loans(account_name: str) -> ToolResult:
    portfolios = {
        "Jerry Seinfeld": 100,
        "Larry David": 50,
    }
    return ToolResult(portfolios[account_name])


def transfer_money(amount: int, from_account: str, to_account: str) -> ToolResult:
    return ToolResult(
        data=f"Transferred {amount} coins from {from_account} to {to_account} successfully."
    )


def get_terrys_offering() -> ToolResult:
    return ToolResult("Terry offers leaf")


def schedule() -> ToolResult:
    return ToolResult("Meeting got scheduled!")


def check_fruit_price(fruit: str) -> ToolResult:
    return ToolResult(f"1 kg of {fruit} costs 10$")


def check_vegetable_price(vegetable: str) -> ToolResult:
    return ToolResult(f"1 kg of {vegetable} costs 3$")


class ProductCategory(Enum):
    LAPTOPS = "laptops"
    PERIPHERALS = "peripherals"


def available_products_by_category(category: ProductCategory) -> ToolResult:
    products_by_category = {
        ProductCategory.LAPTOPS: ["Lenovo", "Dell"],
        ProductCategory.PERIPHERALS: ["Razer Keyboard", "Logitech Mouse"],
    }

    return ToolResult(products_by_category[category])


def available_products_by_categories(categories: list[ProductCategory]) -> ToolResult:
    products_by_category = {
        ProductCategory.LAPTOPS: ["Lenovo", "Dell"],
        ProductCategory.PERIPHERALS: ["Razer Keyboard", "Logitech Mouse"],
    }

    return ToolResult([products_by_category[category] for category in categories])


def recommend_drink(user_is_adult: bool) -> ToolResult:
    if user_is_adult:
        return ToolResult("Beer")
    else:
        return ToolResult("Soda")


def check_username_validity(name: str) -> ToolResult:
    return ToolResult(name != "Dukie")


def get_available_soups() -> ToolResult:
    return ToolResult("['Tomato', 'Turpolance', 'Pumpkin', 'Turkey Soup', 'Tom Yum', 'Onion']")


def fetch_account_balance() -> ToolResult:
    return ToolResult(data={"balance": 1000.0})


def get_keyleth_stamina() -> ToolResult:
    return ToolResult(data=100.0)


def consult_policy() -> ToolResult:
    policies = {
        "return_policy": "The return policy allows returns within 4 days and 4 hours from the time of purchase.",
        "warranty_policy": "All products come with a 1-year warranty.",
    }
    return ToolResult(policies)


def find_answer(inquiry: str) -> ToolResult:
    return ToolResult(f"The answer to '{inquiry}' is — you guessed it — 42")


def other_inquiries() -> ToolResult:
    return ToolResult("Sorry, we could not find a specific answer to your query.")


def try_unlock_card(last_6_digits: Optional[str] = None) -> ToolResult:
    try:
        if not last_6_digits:
            return ToolResult({"failure": "need to specify the last 6 digits of the card"})
        return ToolResult({"success": "card successfully unlocked"})
    except BaseException:
        return ToolResult({"failure": "system error"})


def pay_cc_bill(payment_date: str) -> ToolResult:
    _ = payment_date
    return ToolResult({"result": "success"})


def register_for_sweepstake(
    first_name: str,
    last_name: str,
    father_name: str,
    mother_name: str,
    entry_type: str,
    n_entries: int,
    donation_target: Optional[str] = None,
    donation_percent: Optional[int] = None,
) -> ToolResult:
    return ToolResult({"result": "success"})


class Employees(Enum):
    EMPLOYEE = "John n Coke"
    MANAGER = "Mike Andike"
    DIRECTOR = "Bruno Twix"
    CEO = "Jay Libelly"
    THAT_GUY = "Chris Pikrim"


def calculate_salary(
    name: Employees,
    manager: Employees,
    director: Employees,
    friend: Employees,
    mistress: Employees,
    cleaner: Employees,
) -> ToolResult:
    return ToolResult({"salary": 100})


def calculate_expected_salary(
    name: Employees,
    manager: Employees,
    director: Employees,
    friend: Employees,
    mistress: Employees,
    cleaner: Employees,
) -> ToolResult:
    return ToolResult({"salary": 100})


async def get_electronic_products_by_type(
    product_type: ElectronicProductType,
) -> ToolResult:
    """Get all products that match the specified product type"""
    with open("tests/data/get_products_by_type_data.json", "r") as f:
        database = json.load(f)
    products = [item for item in database if item["type"] == product_type.value]
    return ToolResult({"available_products": products})


def get_bookings(customer_id: str) -> ToolResult:
    if customer_id == "J2T3F00":
        return ToolResult(
            {
                "bookings": """| Booking ID | Start Date  | End Date    | From         | To           |
|------------|-------------|-------------|--------------|--------------|
| PUDW600P   | 2025-07-04  | 2025-07-10  | Los Angeles  | Denver       |
| CLPAJIHO   | 2025-07-01  | 2025-07-10  | Los Angeles  | Miami        |
| 47U0BZFO   | 2025-07-05  | 2025-07-15  | Houston      | Miami        |
| NOK9EHX0   | 2025-08-19  | 2025-08-22  | Phoenix      | Denver       |
| XRT125KL   | 2025-03-15  | 2025-03-20  | Seattle      | Chicago      |
| LMN789PQ   | 2025-04-01  | 2025-04-05  | Boston       | San Francisco|
| WYZ456AB   | 2025-06-22  | 2025-06-30  | Atlanta      | Las Vegas    |"""
            }
        )
    else:
        return ToolResult({"bookings": "No bookings found"})


def get_qualification_info() -> ToolResult:
    return ToolResult(
        data={"qualification_info": "5+ years of experience"},
        canned_response_fields={"qualification_info": "5+ years of experience"},
    )


def transfer_coins(amount: int, from_account: str, to_account: str, pincode: str) -> ToolResult:
    if from_account == "Mark Corrigan" and to_account == "Sophie Chapman":
        if pincode == "1234":
            return ToolResult(data="Transaction successful: Transaction number: 83933")
        else:
            return ToolResult(data="Transaction failed: incorrect pincode")
    return ToolResult(data="Transaction failed: one of the provided accounts does not exist")


async def search_electronic_products(
    keyword: str,
    product_type: Optional[ElectronicProductType] = None,
    min_price: Optional[int] = None,
    max_price: Optional[int] = None,
    in_stock_only: Optional[bool] = False,
    brand: Optional[str] = None,
) -> ToolResult:
    with open("tests/data/get_products_by_type_data.json", "r") as f:
        database = json.load(f)

    # Start with all products
    products = database

    # Filter by keyword (required parameter)
    keyword = keyword.lower()
    products = [
        item
        for item in products
        if keyword in item["title"].lower() or keyword in item["description"].lower()
    ]

    # Apply optional filters
    if product_type:
        products = [item for item in products if item["type"] == product_type]

    if min_price is not None:
        products = [item for item in products if item["price"] >= min_price]

    if max_price is not None:
        products = [item for item in products if item["price"] <= max_price]

    if in_stock_only:
        products = [item for item in products if item["qty"] > 0]

    if brand:
        products = [item for item in products if item["vendor"].lower() == brand.lower()]

    return ToolResult({"available_products": products, "total_results": len(products)})


async def search_products(
    keyword: str,
    product_type: Optional[ElectronicProductType] = None,
    min_price: Optional[int] = None,
    max_price: Optional[int] = None,
    in_stock_only: Optional[bool] = False,
    brand: Optional[str] = None,
) -> ToolResult:
    with open("tests/data/get_products_by_type_data.json", "r") as f:
        database = json.load(f)

    # Start with all products
    products = database

    # Filter by keyword (required parameter)
    keyword = keyword.lower()
    products = [
        item
        for item in products
        if keyword in item["title"].lower() or keyword in item["description"].lower()
    ]

    # Apply optional filters
    if product_type:
        products = [item for item in products if item["type"] == product_type]

    if min_price is not None:
        products = [item for item in products if item["price"] >= min_price]

    if max_price is not None:
        products = [item for item in products if item["price"] <= max_price]

    if in_stock_only:
        products = [item for item in products if item["qty"] > 0]

    if brand:
        products = [item for item in products if item["vendor"].lower() == brand.lower()]

    return ToolResult({"available_products": products, "total_results": len(products)})


def book_flight(
    departure_city: str,
    destination_city: str,
) -> ToolResult:
    return ToolResult(
        data={
            "departure_city": departure_city,
            "destination_city": destination_city,
        }
    )


def class_access_validator(age: int) -> ToolResult:
    if age >= 21:
        return ToolResult(data={"class": "business class"})
    else:
        return ToolResult(data={"class": "economy class"})


def send_email(to: str, subject: str, body: Optional[str] = None) -> ToolResult:
    return ToolResult(data=f"Email sent to {to} with subject '{subject}'.")


class Names(enum.Enum):
    ELIZABETH = "Elizabeth"
    MARRY = "Marry"


def schedule_meeting(
    participant: str, date: str, time: str, agenda: Optional[str] = None
) -> ToolResult:
    return ToolResult(data=f"Meeting scheduled with {', '.join(participant)} on {date} at {time}.")


def schedule_appointment(
    patient: str, doctor_name: str, date: str, time: str, reason: Optional[str] = None
) -> ToolResult:
    return ToolResult(
        data=f"Appointment scheduled for {patient} with {doctor_name} on {date} at {time}."
    )


def reschedule_appointment(
    patient: str, doctor_name: str, new_date: str, new_time: str
) -> ToolResult:
    return ToolResult(
        data=f"Appointment for {patient} with {doctor_name} has been rescheduled to {new_date} at {new_time}."
    )


def transfer_shekels(amount: int, from_account: str, to_account: str) -> ToolResult:
    return ToolResult(
        data=f"Transferred ₪{amount} from {from_account} to {to_account} successfully."
    )


def transfer_dollars(amount: int, from_account: str, to_account: str) -> ToolResult:
    return ToolResult(
        data=f"Transferred ₪{amount} from {from_account} to {to_account} successfully."
    )


async def reset_password(
    username: str,
    phone_number: Optional[str] = "",
    email: Optional[str] = "",
) -> ToolResult:
    if phone_number == "" and email == "":
        return ToolResult({"result": "no email or phone number provided - request rejected"})
    return ToolResult(
        {
            "result": f"password for {username} was reset. An email with further instructions was sent to the account's email address."
        }
    )


class MeetingLocation(Enum):
    ROOM = "meeting room"
    BOOTH = "phone booth"
    KITCHEN = "kitchen"


async def set_a_bbq_appointment(
    start_time: datetime,
    description: str,
    participants: list[str],
    participants_rating: Optional[list[float]] = None,
    end_time: Optional[datetime] = None,
    location: MeetingLocation = MeetingLocation.ROOM,
    alternative_locations: Optional[list[MeetingLocation]] = None,
    meat_to_buy_in_kg: Optional[float] = None,
    vegetarians: Optional[int] = None,
) -> ToolResult:
    return ToolResult(
        {
            "result": "success",
            "message": f"BBQ appointment set successfully in {location} at {start_time} with {len(participants)} participants ({vegetarians} vegetarians).",
            "description": description,
        },
    )


async def find_bbq_appointments(
    day: Optional[date] = None,
    participants: Optional[list[str]] = None,
    location: Optional[MeetingLocation] = MeetingLocation.ROOM,
) -> ToolResult:
    return ToolResult(
        {"result": "success"},
    )


def give_boolean_types(
    boolean: list[bool],
    optional_boolean: Optional[bool],
) -> ToolResult:
    return ToolResult(
        f"Types for boolean is: {type(boolean[0])} and optional boolean: {type(optional_boolean)}"
    )


def check_current_time() -> ToolResult:
    return ToolResult(data="Current time is 18:03", control={"lifespan": "response"})


def check_current_time_emit() -> ToolResult:
    return ToolResult(data="Current time is 9:59", control={"lifespan": "session"})


def availability_check() -> ToolResult:
    return ToolResult(data={"Luxury": False})


def check_customer_location() -> ToolResult:
    return ToolResult(data="Spain!!")


def schedule_appointment_2(date: datetime) -> ToolResult:
    # Simulate scheduling the appointment
    return ToolResult(data=f"Appointment scheduled for {date}")


def check_eligibility(account_id: int, amount: int) -> ToolResult:
    return ToolResult(
        data=f"Account {account_id} is eligible for a loan of {amount} over 24 months at a rate of 6.5% interest per month."
    )


def check_lab_results(name: str) -> ToolResult:
    return ToolResult(data=f"Lab results for {name}: {name} is as healthy as a horse.")


async def change_credit_limit(
    username: str,
    new_limit: float,
    current_limit: float,
) -> ToolResult:
    diff = abs(new_limit - current_limit)

    if diff <= 10_000:
        return ToolResult(
            {
                "result": f"Credit limit for {username} has been successfully changed from ${current_limit:,.2f} to ${new_limit:,.2f}."
            }
        )
    else:
        return ToolResult(
            {
                "result": f"Requested change from ${current_limit:,.2f} to ${new_limit:,.2f} exceeds $10,000. Supervisor approval is required."
            }
        )


async def get_credit_limit(username: str) -> ToolResult:
    def _mock_lookup_credit_limit(username: str) -> Optional[float]:
        mock_database = {
            "alice": 15000.0,
            "bob": 10000.0,
            "charlie": 20000.0,
        }
        return mock_database.get(username.lower(), 12000.0)  # default limit

    current_limit = _mock_lookup_credit_limit(username)
    return ToolResult(
        {
            "result": f"Current credit limit for {username} is ${current_limit:,.2f}.",
        }
    )


def list_cards() -> ToolResult:
    """List all cards associated with the customer's account"""
    return ToolResult(
        [
            {
                "card_id": 1,
                "card_name": "Chase Freedom",
                "card_number": "**** **** **** 1234",
                "card_type": "credit",
            },
            {
                "card_id": 2,
                "card_name": "Chase Sapphire",
                "card_number": "**** **** **** 5678",
                "card_type": "credit",
            },
        ]
    )


def lock_card(card_number: str, reason: str) -> ToolResult:
    """Lock a specific card for security reasons"""
    if reason.lower() in ["lost", "stolen"]:
        return ToolResult(
            {
                "result": "failure",
                "message": f"For lost or stolen cards ending in {card_number}, please call customer support at 123456789",
            }
        )
    else:
        return ToolResult(
            {
                "result": "success",
                "message": f"Card ending in {card_number} has been successfully locked for reason: {reason}",
            }
        )
