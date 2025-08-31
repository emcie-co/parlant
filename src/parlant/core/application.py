# Copyright 2025 Emcie Co Ltd.
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

from parlant.app_modules.agents import AgentModule
from parlant.app_modules.context_variables import ContextVariableModule
from parlant.app_modules.relationships import RelationshipModule
from parlant.app_modules.services import ServiceModule
from parlant.app_modules.sessions import SessionModule
from parlant.app_modules.tags import TagModule
from parlant.app_modules.customers import CustomerModule
from parlant.app_modules.guidelines import GuidelineModule


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
    ) -> None:
        self.agents = agent_module
        self.sessions = session_module
        self.services = service_module
        self.tags = tag_module
        self.variables = context_variable_module
        self.customers = customer_module
        self.guidelines = guideline_module
        self.relationships = relationship_module
