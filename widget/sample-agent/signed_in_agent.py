"""Agent Tom with real customer authentication — the playground's signed-in server.

The regular sample agent (`agent.py`, port 8810) runs in dev mode with no auth, so
the widget's Signed-in tab "works" against it without proving anything. This variant
runs the same agent on port **8811** with customer-JWT verification and
auto-provisioning turned on, so the signed-in flow is real: the widget's token is
verified, and an unknown customer is created on first contact from the token's
`sub` (id) and `name` claims.

Run it from the repo root (needs an OpenAI key for both chat and vision):

    OPENAI_API_KEY=sk-... uv run widget/sample-agent/signed_in_agent.py

Then mint a customer token with the matching demo secret:

    uv run widget/sample-agent/mint_token.py

In the playground, pick **Signed in**, set the server URL to
http://localhost:8811, and paste the minted token. The demo JWT secret below is
for local play only — a real deployment loads its secret from a secret manager
(see the README's "Signed-in setup").
"""

import asyncio

import parlant.sdk as p

from agent import describe_attached_image

DEMO_JWT_SECRET = "widget-demo-jwt-secret-0123456789abcdef"


async def main() -> None:
    async with p.Server(
        port=8811,
        tool_service_port=8812,
        models={"*": "openai"},
        auth=p.AuthConfig(
            admin_api_key="widget-demo-admin-key",
            customer_jwt_secret=DEMO_JWT_SECRET,
            guest_token_secret="widget-demo-guest-secret-0123456789abcdef",
            auto_provision_customers=True,
        ),
    ) as server:
        agent = await server.create_agent(
            name="Agent Tom",
            instructions=(
                "You are a friendly assistant for a product demo. Chat naturally, and "
                "greet the customer by name when you know it. Format replies with "
                "Markdown where it helps. Whenever the customer attaches an image, or "
                "asks what's in a photo they sent, describe it."
            ),
            id="tom",
            effort=p.Effort.MAX,
        )

        await agent.create_policy(
            content=(
                "When the customer attaches an image, or asks what's in a photo they sent, "
                "use the describe_attached_image tool and tell them what's in it."
            ),
            tools=[describe_attached_image],
        )


if __name__ == "__main__":
    asyncio.run(main())
