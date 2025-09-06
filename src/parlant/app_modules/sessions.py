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
from __future__ import annotations
from typing import Optional, Sequence
from datetime import datetime, timezone
from parlant.core.sessions import (SessionStore, SessionListener, SessionId, Session, AgentId, CustomerId, SessionUpdateParams)
from parlant.core.agents import AgentStore
from parlant.core.customers import CustomerStore
from parlant.core.nlp.service import NLPService
from parlant.core.loggers import Logger
from parlant.core.application import Application
from parlant.core.agents import AgentId
from parlant.core.customers import CustomerId
from parlant.api.authorization import AuthorizationPolicy

class SessionModule:
    def __init__(self, session_store: SessionStore,session_listener: SessionListener,agent_store: AgentStore,
        customer_store: CustomerStore,
        nlp_service: NLPService,
        logger: Logger,
        authorization_policy: AuthorizationPolicy,
        application: Application) -> None:
        self._session_store = session_store
        self._session_listener = session_listener
        self._agent_store = agent_store
        self._customer_store = customer_store
        self._nlp_service = nlp_service
        self._logger = logger
        self._authorization_policy = authorization_policy
        self._application = application
        
        async def create_session(
            self,
            agent_id: AgentId,
            customer_id: Optional[CustomerId] = None,
            title: Optional[str] = None,
            allow_greeting: bool = False,
        ) -> Session:
            await self._agent_store.read_agent(agent_id=agent_id)
            session = await self._session_store.create_session(
                creation_utc=datetime.now(timezone.utc),
                customer_id=customer_id,
                agent_id=agent_id,
                title=title,
            )
            if allow_greeting:
                await self._application.dispatch_processing_task(session)
            return session
        
        async def get_session(self, session_id: SessionId) -> Session:
            return await self._session_store.read_session(session_id=session_id)
        async def list_sessions(
            self,
            agent_id: Optional[AgentId] = None,
            customer_id: Optional[CustomerId] = None,
        ) -> Sequence[Session]:
            return await self._session_store.list_sessions(
                agent_id=agent_id,
                customer_id=customer_id,
            )
            
        async def update_session(
            self,
            session_id: SessionId,
            params: SessionUpdateParams,
            ) -> Session:
            return await self._session_store.update_session(
                session_id=session_id,
                params=params,
            )
        async def delete_session(self, session_id: SessionId) -> None:
            await self._session_store.read_session(session_id=session_id)
            await self._session_store.delete_session(session_id=session_id)
        async def delete_sessions(self, agent_id: Optional[AgentId] = None, customer_id: Optional[CustomerId] = None) -> None:  
            sessions = await self._session_store.list_sessions(
                agent_id=agent_id,
                customer_id=customer_id,
            )
            for session in sessions:
                await self._session_store.delete_session(session_id=session.id)
        