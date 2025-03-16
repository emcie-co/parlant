# Copyright 2024 Emcie Co Ltd.
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

import enum
from enum import Enum
import json
from typing import Optional

from parlant.core.tools import ToolResult

pizza_menu = [
    {
        "type": "Margherita",
        "price": 8.99,
        "amount": 10,
        "ingredients": ["tomato sauce", "mozzarella cheese", "fresh basil", "olive oil"],
    },
    {
        "type": "Pepperoni",
        "price": 10.99,
        "amount": 15,
        "ingredients": ["tomato sauce", "mozzarella cheese", "pepperoni slices"],
    },
    {
        "type": "BBQ Chicken",
        "price": 12.99,
        "amount": 18,
        "ingredients": [
            "bbq sauce",
            "mozzarella cheese",
            "grilled chicken",
            "red onions",
            "cilantro",
        ],
    },
    {
        "type": "Veggie Supreme",
        "price": 9.99,
        "amount": 10,
        "ingredients": [
            "tomato sauce",
            "mozzarella cheese",
            "bell peppers",
            "mushrooms",
            "red onions",
            "black olives",
            "spinach",
        ],
    },
    {
        "type": "Hawaiian",
        "price": 11.99,
        "amount": 12,
        "ingredients": ["tomato sauce", "mozzarella cheese", "ham", "pineapple"],
    },
    {
        "type": "Meat Lovers",
        "price": 13.99,
        "amount": 14,
        "ingredients": [
            "tomato sauce",
            "mozzarella cheese",
            "pepperoni",
            "sausage",
            "bacon",
            "ground beef",
            "ham",
        ],
    },
    {
        "type": "Buffalo Chicken",
        "price": 12.99,
        "amount": 8,
        "ingredients": [
            "buffalo sauce",
            "mozzarella cheese",
            "chicken",
            "red onions",
            "blue cheese crumbles",
        ],
    },
    {
        "type": "Four Cheese",
        "price": 10.99,
        "amount": 11,
        "ingredients": [
            "tomato sauce",
            "mozzarella cheese",
            "parmesan cheese",
            "gorgonzola cheese",
            "ricotta cheese",
        ],
    },
    {
        "type": "Mediterranean",
        "price": 11.99,
        "amount": 9,
        "ingredients": [
            "olive oil",
            "feta cheese",
            "sun-dried tomatoes",
            "kalamata olives",
            "artichoke hearts",
            "spinach",
        ],
    },
    {
        "type": "Supreme",
        "price": 14.99,
        "amount": 13,
        "ingredients": [
            "tomato sauce",
            "mozzarella cheese",
            "pepperoni",
            "sausage",
            "bell peppers",
            "onions",
            "mushrooms",
            "black olives",
        ],
    },
]


class Categories(enum.Enum):
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


class ProductType(enum.Enum):
    DRINKS = "drinks"
    TOPPINGS = "toppings"


def get_available_product_by_type(product_type: ProductType) -> ToolResult:
    if product_type == ProductType.DRINKS:
        return get_available_drinks()
    elif product_type == ProductType.TOPPINGS:
        return get_available_toppings()
    else:
        return ToolResult([])


def add(first_number: int, second_number: int) -> ToolResult:
    return ToolResult(first_number + second_number)


def multiply(first_number: int, second_number: int) -> ToolResult:
    return ToolResult(first_number * second_number)


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


def transfer_money(from_account: str, to_account: str) -> ToolResult:
    return ToolResult(True)


def get_terrys_offering() -> ToolResult:
    return ToolResult("Terry offers leaf")


def schedule() -> ToolResult:
    return ToolResult("Meeting got scheduled!")


def check_fruit_price(fruit: str) -> ToolResult:
    return ToolResult(f"1 kg of {fruit} costs 10$")


def check_vegetable_price(vegetable: str) -> ToolResult:
    return ToolResult(f"1 kg of {vegetable} costs 3$")


class ProductCategory(enum.Enum):
    LAPTOPS = "laptops"
    PERIPHERALS = "peripherals"


def available_products_by_category(category: ProductCategory) -> ToolResult:
    products_by_category = {
        ProductCategory.LAPTOPS: ["Lenovo", "Dell"],
        ProductCategory.PERIPHERALS: ["Razer Keyboard", "Logitech Mouse"],
    }

    return ToolResult(products_by_category[category])


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
        return ToolResult({"success": "card succesfuly unlocked"})
    except BaseException:
        return ToolResult({"failure": "system error"})


def pay_cc_bill(payment_date: str) -> ToolResult:
    _ = payment_date
    return ToolResult({"result": "success"})


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


class PizzaType(Enum):
    """Enum representing different types of pizzas available."""

    MARGHERITA = "Margherita"
    PEPPERONI = "Pepperoni"
    BBQ_CHICKEN = "BBQ Chicken"
    VEGGIE_SUPREME = "Veggie Supreme"
    HAWAIIAN = "Hawaiian"
    MEAT_LOVERS = "Meat Lovers"
    BUFFALO_CHICKEN = "Buffalo Chicken"
    FOUR_CHEESE = "Four Cheese"
    MEDITERRANEAN = "Mediterranean"
    SUPREME = "Supreme"


class PizzaIngredient(str, Enum):
    TOMATO_SAUCE = "tomato sauce"
    MOZZARELLA_CHEESE = "mozzarella cheese"
    FRESH_BASIL = "fresh basil"
    OLIVE_OIL = "olive oil"
    PEPPERONI_SLICES = "pepperoni slices"
    BBQ_SAUCE = "bbq sauce"
    GRILLED_CHICKEN = "grilled chicken"
    RED_ONIONS = "red onions"
    CILANTRO = "cilantro"
    BELL_PEPPERS = "bell peppers"
    MUSHROOMS = "mushrooms"
    BLACK_OLIVES = "black olives"
    SPINACH = "spinach"
    HAM = "ham"
    PINEAPPLE = "pineapple"
    PEPPERONI = "pepperoni"
    SAUSAGE = "sausage"
    BACON = "bacon"
    GROUND_BEEF = "ground beef"
    BUFFALO_SAUCE = "buffalo sauce"
    CHICKEN = "chicken"
    BLUE_CHEESE_CRUMBLES = "blue cheese crumbles"
    PARMESAN_CHEESE = "parmesan cheese"
    GORGONZOLA_CHEESE = "gorgonzola cheese"
    RICOTTA_CHEESE = "ricotta cheese"
    FETA_CHEESE = "feta cheese"
    SUN_DRIED_TOMATOES = "sun-dried tomatoes"
    KALAMATA_OLIVES = "kalamata olives"
    ARTICHOKE_HEARTS = "artichoke hearts"
    ONIONS = "onions"


async def get_products_by_ingredient(ingredient: PizzaIngredient) -> ToolResult:
    products = [item["type"] for item in pizza_menu if ingredient.value in item["ingredients"]]
    return ToolResult({"available_products": products})


async def get_availability_by_type_and_amount(product_type: PizzaType, amount: int) -> ToolResult:
    amount = 0
    available = False
    for item in pizza_menu:
        if item["type"] == product_type.value and item["amount"] >= amount:
            available = True
            amount = item["amount"]
            break
    return ToolResult({"product_availability": available, "amount": amount})


async def get_availability_by_type(product_type: PizzaType) -> ToolResult:
    available = False
    for item in pizza_menu:
        if item.get("type") == product_type.value and item["amount"]:
            available = True
            break
    return ToolResult({"product_availability": available})


async def get_menu() -> ToolResult:
    menu = ""
    for item in pizza_menu:
        if item["amount"] > 0:
            menu += f"""{item["type"]} ({item["ingredients"]}) : {item["price"]} $ \n"""
    return ToolResult({"menu": menu})


async def process_order(product_type: PizzaType, amount: int) -> ToolResult:
    for item in pizza_menu:
        if item["type"] == product_type.value and item["amount"] >= amount:
            item["amount"] -= amount
            amount = item["amount"]
            break
    return ToolResult({"new_amount": amount})
