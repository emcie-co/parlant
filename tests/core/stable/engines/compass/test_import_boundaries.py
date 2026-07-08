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

"""The pro package plugs into ports defined by the core engine — never the
other way around. This guardrail keeps that import direction honest so the pro
package stays independently extractable.

Scope is deliberately ``parlant.core``: the server binary composes both sides
by design, and the NLP adapters currently reference pro schemas in their
hardcoded schema-to-model maps (to be revisited together with the
schematic-generator model-mapping seam).
"""

from pathlib import Path

import parlant.core

_PRO_PACKAGE = "parlant.core.engines.compass.pro"


def test_that_no_core_module_outside_the_pro_package_imports_from_it() -> None:
    core_root = Path(next(iter(parlant.core.__path__)))
    pro_root = core_root / "engines" / "compass" / "pro"

    offenders: list[str] = []

    for module in core_root.rglob("*.py"):
        if pro_root in module.parents:
            continue

        for line_number, line in enumerate(module.read_text().splitlines(), start=1):
            stripped = line.strip()
            if not stripped.startswith(("import ", "from ")):
                continue
            if _PRO_PACKAGE in stripped:
                offenders.append(f"{module.relative_to(core_root)}:{line_number}: {stripped}")

    assert not offenders, (
        "Core modules must not depend on the pro package (ports point the other way):\n"
        + "\n".join(offenders)
    )
