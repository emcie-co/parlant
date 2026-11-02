"""Mints a customer JWT for the signed-in playground server (signed_in_agent.py).

This plays the role of YOUR backend: in a real app, a server-side endpoint mints
this token for the logged-in user (see the widget README's "Signed-in setup").

    uv run widget/sample-agent/mint_token.py                    # Dana Cohen / crm-demo-1
    uv run widget/sample-agent/mint_token.py user-42 "Avi Levi" # your own id + name

Paste the printed token into the playground's Signed-in tab. Tokens expire after
an hour; run the script again for a fresh one.
"""

import sys
import time

import jwt

DEMO_JWT_SECRET = "widget-demo-jwt-secret-0123456789abcdef"

customer_id = sys.argv[1] if len(sys.argv) > 1 else "crm-demo-1"
name = sys.argv[2] if len(sys.argv) > 2 else "Dana Cohen"

print(
    jwt.encode(
        {"sub": customer_id, "name": name, "exp": int(time.time()) + 3600},
        DEMO_JWT_SECRET,
        algorithm="HS256",
    )
)
