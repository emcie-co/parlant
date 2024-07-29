from typing import Literal


def get_available_drinks() -> list[str]:
    return ["Sprite", "Coca Cola"]


def get_available_toppings() -> list[str]:
    return ["Pepperoni", "Mushrooms", "Olives"]


def expert_answer(user_query: str) -> str:
    answers = {"Hey, where are your offices located?": "Our Offices located in Tel Aviv"}
    return answers[user_query]


def get_available_product_by_type(product_type: Literal["drinks", "toppings"]) -> list[str]:
    if product_type == "drinks":
        return get_available_drinks()
    elif product_type == "toppings":
        return get_available_toppings()
    else:
        return []


def add(first_number: int, second_number: int) -> int:
    return first_number + second_number


def multiply(first_number: int, second_number: int) -> int:
    return first_number * second_number


def get_account_balance(account_name: str) -> int:
    balances = {
        "Jerry Seinfeld": 1000000000,
        "Larry David": 450000000,
        "John Smith": 100,
    }
    return balances.get(account_name, -1)


def get_account_loans(account_name: str) -> int:
    portfolios = {
        "Jerry Seinfeld": 100,
        "Larry David": 50,
    }
    return portfolios[account_name]


def transfer_money(from_account: str, to_account: str) -> bool:
    return True


def get_terrys_offering() -> str:
    return "Terry offers leaf"
