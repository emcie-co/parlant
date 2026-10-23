"""Agent Tom — the Parlant widget playground's sample agent.

It chats normally, and whenever a customer attaches an image it describes what's in it
using an OpenAI vision model. It runs a dev-mode Parlant server (no auth) on port 8810,
so the widget's guest flow works against it out of the box.

Run it from the repo root (needs an OpenAI key for both chat and vision):

    OPENAI_API_KEY=sk-... uv run widget/sample-agent/agent.py

Then open the playground (`cd widget && npm run dev`) — it already defaults to
http://localhost:8810 with agent id "tom", so just click Connect.

How the image reaches the model: Parlant only shows the engine a short text note for an
attachment (never the bytes), so the describe-image tool below pulls the attachment's URL
off the customer's message and calls the vision model itself. The playground's demo
uploader hands over a data: URL, so this works locally with no external storage.
"""

import asyncio
import os
import traceback

from openai import AsyncOpenAI

import parlant.sdk as p

# Any OpenAI vision-capable model.
VISION_MODEL = "gpt-5.6-luna"


@p.tool
async def describe_attached_image(filename: str = "") -> p.ToolResult:
    """Describe an image the customer attached.

    Pass `filename` (the name from the "[image: <name>]" note) to pick a specific picture;
    leave it blank to describe the most recent one. Works for an image from ANY message in
    the conversation, not just the latest.
    """
    interaction = p.Session.current.interaction

    # Every image attachment in the conversation, most-recent message first.
    images = [
        a
        for message in reversed(interaction.messages)
        for a in message.attachments
        if (a.get("media_type") or "").startswith("image/")
    ]

    if not images:
        # Distinct from "wrong name": no image reached the tool at all. If you hit this
        # after attaching one, the attachment isn't arriving on the customer message.
        return p.ToolResult(
            data="There's no image attached to this conversation yet — ask the customer to attach one."
        )

    # Prefer an exact file-name match; otherwise fall back to the most recent image, so a
    # slightly-off `filename` from the model still describes the right picture.
    match = next((a for a in images if a.get("filename") == filename), images[0])

    client = AsyncOpenAI()

    # OpenAI Responses API. `image_url` takes the URL (or, here, a self-contained data:
    # URL) directly as a string; `output_text` aggregates the reply text.
    try:
        response = await client.responses.create(
            model=VISION_MODEL,
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "Describe what's in this image in two or three sentences.",
                        },
                        {"type": "input_image", "image_url": match["url"], "detail": "auto"},
                    ],
                }
            ],
        )
    except Exception as exc:
        # Surface the real reason instead of an opaque "failed attempt". The most common
        # causes here are a VISION_MODEL your key can't call or that isn't vision-capable.
        traceback.print_exc()
        return p.ToolResult(
            data=f"The vision model ({VISION_MODEL}) couldn't describe the image: {type(exc).__name__}: {exc}"
        )

    description = response.output_text or "I couldn't read the image."

    # `data` is what the model sees (so the agent can talk about the picture). `metadata`
    # is frontend-only: this `widget` payload is rendered by the "picture_description"
    # renderer the playground registers (see widget/example/src/widgets.tsx) into a card
    # showing the thumbnail + description. The widget core is told nothing agent-specific.
    return p.ToolResult(
        data=description,
        metadata={
            "widget": {
                "type": "picture_description",
                "filename": match.get("filename") or filename,
                "description": description,
            }
        },
    )


async def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY — Agent Tom uses OpenAI for chat and vision.")

    async with p.Server(port=8810, models={"*": "openai"}) as server:
        agent = await server.create_agent(
            name="Agent Tom",
            instructions=(
                "You are a friendly assistant for a product demo. Chat naturally, and format "
                "replies with Markdown where it helps — bold for emphasis, bullet or numbered "
                "lists, and fenced code blocks for any code. Whenever the customer attaches an "
                "image, or asks what's in a photo they sent, describe it."
            ),
            id="tom",
            effort=p.Effort.MEDIUM,
        )

        await agent.create_policy(
            content=(
                "When the customer attaches an image, or asks what's in a photo they sent, "
                "use the describe_attached_image tool and tell them what's in it."
            ),
            tools=[describe_attached_image],
        )

        await agent.create_canned_response("Sorry, don't have that information.")

        await agent.create_canned_response("Can't do that, sorry mate.")


if __name__ == "__main__":
    asyncio.run(main())
