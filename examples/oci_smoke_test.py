import asyncio
import os
from pathlib import Path
import tempfile

import parlant.sdk as p
from parlant.sdk import NLPServices


def _load_oci_env_from_yaml(yaml_path: Path) -> None:
    if not yaml_path.exists():
        return

    lines = yaml_path.read_text(encoding="utf-8").splitlines()
    in_oci = False
    key_indent = None
    key_map: dict[str, str] = {}
    private_key_lines: list[str] = []
    collecting_key = False
    key_block_indent = None

    for line in lines:
        if not in_oci:
            if line.strip() == "oci:":
                in_oci = True
                key_indent = len(line) - len(line.lstrip())
            continue

        if line.strip() and (len(line) - len(line.lstrip())) <= key_indent:
            break

        if collecting_key:
            indent = len(line) - len(line.lstrip())
            if key_block_indent is not None and indent <= key_block_indent:
                collecting_key = False
            else:
                private_key_lines.append(line[key_block_indent + 2 :])
                continue

        stripped = line.strip()
        if not stripped or ":" not in stripped:
            continue

        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip().strip('"')

        if key == "private_key_content" and value.startswith("|"):
            collecting_key = True
            key_block_indent = len(line) - len(line.lstrip())
            private_key_lines = []
            continue

        key_map[key] = value

    if not key_map:
        return

    if not os.environ.get("OCI_COMPARTMENT_ID") and key_map.get("compartment_id"):
        os.environ["OCI_COMPARTMENT_ID"] = key_map["compartment_id"]

    if not os.environ.get("OCI_USER") and key_map.get("user_id"):
        os.environ["OCI_USER"] = key_map["user_id"]

    if not os.environ.get("OCI_TENANCY") and key_map.get("tenancy"):
        os.environ["OCI_TENANCY"] = key_map["tenancy"]

    if not os.environ.get("OCI_FINGERPRINT") and key_map.get("fingerprint"):
        os.environ["OCI_FINGERPRINT"] = key_map["fingerprint"]

    if not os.environ.get("OCI_REGION") and key_map.get("region"):
        os.environ["OCI_REGION"] = key_map["region"]

    if not os.environ.get("OCI_KEY_FILE") and private_key_lines:
        key_path = Path(tempfile.gettempdir()) / "parlant_oci_key.pem"
        key_path.write_text("\n".join(private_key_lines) + "\n", encoding="utf-8")
        os.environ["OCI_KEY_FILE"] = str(key_path)


async def main() -> None:
    infrastructure = Path(
        "C:/Users/GLoverde/PycharmProjects/Obelix/config/infrastructure.yaml"
    )
    _load_oci_env_from_yaml(infrastructure)

    async with p.Server(nlp_service=NLPServices.oci) as server:
        agent = await server.create_agent(
            name="OCI Smoke Test",
            description="Agente di test",
        )

        await agent.create_guideline(
            condition="il cliente ti saluta",
            action="rispondi in maniera cordiale e chiedi il nome ",
        )

        await agent.create_guideline(
            condition="il cliente ti dice il suo nome",
            action="digli che è un bellissimo nome e che tu ti chiami Pinco, se il cliente si chiama Giulio digli che preferici non parlare",
        )


if __name__ == "__main__":
    asyncio.run(main())
