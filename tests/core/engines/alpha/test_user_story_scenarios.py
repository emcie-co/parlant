from pytest_bdd import scenarios
from tests.core.engines.alpha.utils import load_steps


load_steps(
    "agents",
    "context_variables",
    "engines",
    "events",
    "guidelines",
    "sessions",
    "terms",
    "tools",
)

scenarios(
    *(
        f"core/engines/alpha/features/user_stories/{feature}.feature"
        for feature in ("conversation",)
    )
)
