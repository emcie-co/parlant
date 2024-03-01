from ast import Dict
import asyncio
import requests
import json
from typing import List
import openai
import os
import hashlib

_hash_func = hashlib.sha256


# Your Zendesk domain, replace 'your_domain' with your actual Zendesk domain
domain = "emcie.zendesk.com"

# Your Zendesk email and API token, replace with your actual email and API token
email = "dor@emcie.co"
password = "Aa!@123456"

# API endpoint to create a new ticket
url = f"https://{domain}/api/v2/tickets.json"

OPENAI_API_KEY = "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"
OPENAI_API_KEY = "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"


client = openai.AsyncClient(api_key=OPENAI_API_KEY)
# The headers for authentication
from requests.auth import HTTPBasicAuth

headers = {
    "Content-Type": "application/json",
}


# The data for the new ticket
def add_ticket(ticket_data):
    # Make the POST request to create a new ticket
    response = requests.post(
        url, json={"ticket": ticket_data}, headers=headers, auth=HTTPBasicAuth(email, password)
    )

    # Check if the request was successful
    if response.status_code == 201:
        print("Ticket created successfully.")
        print(response.json())
    else:
        print("Failed to create ticket.")
        print(response.text)


def json_tickets_to_zendesk() -> None:
    with open("/Users/dorzo/Downloads/tickets_spot_netapp_random.json", "r") as _f:
        ticket_list = json.load(_f)["tickets"]
        for ticket in ticket_list:
            add_ticket(ticket)


def get_all_tickets() -> List:
    response = requests.get(url, headers=headers, auth=HTTPBasicAuth(email, password))
    return response.json()["tickets"]


async def get_document(ticket: dict) -> str:
    ticket_prompt = f"""Please summerize this zendesk ticket. the output will be a json object with one property which the key is summary and the value is the summarization. ticket subject: {ticket["subject"]}, ticket description: {ticket["description"]}"""
    result = await prompt(ticket_prompt)
    print(f"prompt_result: {result}")
    return result.split("\n\n")


async def preprocessing_ticket(ticket: dict) -> Dict:
    return [
        {
            "meta_data": {**{"tool": "zendesk"}, **ticket},
            "document": json.loads(doc)["summary"],
            "id": f"{int(_hash_func("zendesk".encode()).hexdigest(), 16) % 10**8}_{ticket["id"]}",
        }
        for doc in await get_document(ticket)
    ]


async def prompt(text):
    response = await client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": text}],
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content


async def get_documents() -> None:
    result = []
    for ticket in get_all_tickets():
        result.append(await preprocessing_ticket(ticket))
    with open("zendesk_docs", "w") as _f:
        json.dump(result, _f, indent=2)

asyncio.run(get_documents())
