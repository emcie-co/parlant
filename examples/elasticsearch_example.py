"""
Example: Using Elasticsearch with Parlant SDK

This example demonstrates how to use Elasticsearch as the persistence layer
for both document storage and vector storage in Parlant.

Prerequisites:
1. Install Elasticsearch: https://www.elastic.co/downloads/elasticsearch
2. Install the Python client: pip install elasticsearch
3. Configure environment variables (see below)

Environment Variables (optional - defaults will work with local Elasticsearch):
    export ELASTICSEARCH__HOST=localhost
    export ELASTICSEARCH__PORT=9200
    export ELASTICSEARCH__INDEX_PREFIX=parlant
"""

import asyncio
import os

from parlant.sdk import Server, NLPServices, tool, ToolContext, ToolResult


# Define a simple tool
@tool(name="get_weather")
async def get_weather(context: ToolContext, location: str) -> ToolResult:
    """Get the current weather for a location."""
    return ToolResult(
        data=f"The weather in {location} is sunny and 72°F"
    )


async def main() -> None:
    """Run the Elasticsearch example."""
    
    # Configure Elasticsearch connection (optional - uses defaults if not set)
    # os.environ["ELASTICSEARCH__HOST"] = "localhost"
    # os.environ["ELASTICSEARCH__PORT"] = "9200"
    # os.environ["ELASTICSEARCH__INDEX_PREFIX"] = "parlant_demo"
    
    print("Starting Parlant with Elasticsearch integration...")
    
    # Create server with Elasticsearch for both document and vector storage
    async with Server(
        nlp_service=NLPServices.openai,
        session_store="elasticsearch",      # Sessions in Elasticsearch
        customer_store="elasticsearch",     # Customers in Elasticsearch
        variable_store="elasticsearch",     # Variables in Elasticsearch
        vector_store="elasticsearch",       # Vectors in Elasticsearch
        cache_store="elasticsearch",        # Caches in Elasticsearch
        migrate=True,                       # Allow migrations
    ) as server:
        print("✓ Server started with Elasticsearch integration")
        
        # Create an agent
        agent = await server.create_agent(
            name="Weather Assistant",
            description="An AI assistant that helps with weather information",
        )
        print(f"✓ Created agent: {agent.name}")
        
        # Create a customer
        customer = await server.create_customer(
            name="Alice",
            metadata={
                "email": "alice@example.com",
                "location": "San Francisco",
            }
        )
        print(f"✓ Created customer: {customer.name}")
        
        # Create a glossary term (stored in Elasticsearch vector DB)
        term = await agent.create_term(
            name="Weather Forecast",
            description="A prediction of future weather conditions",
            synonyms=["forecast", "weather prediction", "outlook"]
        )
        print(f"✓ Created glossary term: {term.name}")
        
        # Create a context variable (stored in Elasticsearch document DB)
        variable = await agent.create_variable(
            name="preferred_units",
            description="User's preferred temperature units (Celsius or Fahrenheit)",
        )
        print(f"✓ Created context variable: {variable.name}")
        
        # Set variable value for the customer
        await variable.set_value_for_customer(customer, "Fahrenheit")
        print(f"✓ Set variable value for {customer.name}")
        
        # Create a guideline with a tool
        guideline = await agent.create_guideline(
            condition="when the user asks about weather",
            action="use the get_weather tool to provide current weather information",
            tools=[get_weather],
        )
        print(f"✓ Created guideline with tool: {guideline.condition}")
        
        # Create a canned response (stored in Elasticsearch vector DB)
        canned_response_id = await agent.create_canned_response(
            template="The weather in {{location}} is {{weather}}. Have a great day!",
            signals=["weather", "forecast"],
        )
        print(f"✓ Created canned response: {canned_response_id}")
        
        print("\n" + "=" * 60)
        print("Elasticsearch Integration Summary")
        print("=" * 60)
        print("\nData stored in Elasticsearch:")
        print("  • Document DB:")
        print(f"    - Customer: {customer.name}")
        print(f"    - Context variable: {variable.name}")
        print("  • Caches:")
        print("    - Embedding cache (cache_embeddings)")
        print("    - Evaluation cache (evaluation_cache)")
        print("  • Vector DB:")
        print(f"    - Glossary term: {term.name}")
        print(f"    - Canned response: {canned_response_id}")
        print(f"    - Guideline: {guideline.id}")
        print("\n" + "=" * 60)
        
        print("\nYou can now:")
        print("1. Check Elasticsearch indices:")
        print("   curl http://localhost:9200/_cat/indices?v")
        print("\n2. Query customer data:")
        print(f"   curl http://localhost:9200/parlant_customers/_search?pretty")
        print("\n3. Query vector data:")
        print(f"   curl http://localhost:9200/parlant_glossary_*/_search?pretty")
        print("\n" + "=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

