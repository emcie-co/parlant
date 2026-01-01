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

from parlant.core.app_modules.agents import AgentModule
from parlant.core.app_modules.capabilities import CapabilityModule
from parlant.core.app_modules.canned_responses import CannedResponseModule
from parlant.core.app_modules.context_variables import ContextVariableModule
from parlant.core.app_modules.evaluations import EvaluationModule
from parlant.core.app_modules.journeys import JourneyModule
from parlant.core.app_modules.relationships import RelationshipModule
from parlant.core.app_modules.services import ServiceModule
from parlant.core.app_modules.sessions import SessionModule
from parlant.core.app_modules.tags import TagModule
from parlant.core.app_modules.customers import CustomerModule
from parlant.core.app_modules.guidelines import GuidelineModule
from parlant.core.app_modules.glossary import GlossaryModule


class Application:
    def __init__(
        self,
        agent_module: AgentModule,
        session_module: SessionModule,
        service_module: ServiceModule,
        tag_module: TagModule,
        customer_module: CustomerModule,
        guideline_module: GuidelineModule,
        context_variable_module: ContextVariableModule,
        relationship_module: RelationshipModule,
        journey_module: JourneyModule,
        glossary_module: GlossaryModule,
        evaluation_module: EvaluationModule,
        capability_module: CapabilityModule,
        canned_response_module: CannedResponseModule,
    ) -> None:
        self.agents = agent_module
        self.sessions = session_module
        self.services = service_module
        self.tags = tag_module
        self.capabilities = capability_module
        self.variables = context_variable_module
        self.customers = customer_module
        self.guidelines = guideline_module
        self.relationships = relationship_module
        self.journeys = journey_module
        self.glossary = glossary_module
        self.evaluations = evaluation_module
        self.canned_responses = canned_response_module
