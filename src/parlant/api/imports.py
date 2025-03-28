"""Import module for the API package."""

from typing import Awaitable, Callable, TypeAlias

from fastapi import APIRouter, FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.types import Receive, Scope, Send
from lagom import Container

from parlant.adapters.loggers.websocket import WebSocketLogger
from parlant.api import agents, index
from parlant.api import sessions
from parlant.api import glossary
from parlant.api import guidelines
from parlant.api import context_variables as variables
from parlant.api import services
from parlant.api import tags
from parlant.api import customers
from parlant.api import logs
from parlant.api import fragments
from parlant.core.context_variables import ContextVariableStore
from parlant.core.contextual_correlator import ContextualCorrelator
from parlant.core.agents import AgentStore
from parlant.core.common import ItemNotFoundError, generate_id
from parlant.core.customers import CustomerStore
from parlant.core.evaluations import EvaluationStore, EvaluationListener
from parlant.core.fragments import FragmentStore
from parlant.core.guideline_connections import GuidelineConnectionStore
from parlant.core.guidelines import GuidelineStore
from parlant.core.guideline_tool_associations import GuidelineToolAssociationStore
from parlant.core.nlp.service import NLPService
from parlant.core.services.tools.service_registry import ServiceRegistry
from parlant.core.sessions import SessionListener, SessionStore
from parlant.core.glossary import GlossaryStore
from parlant.core.services.indexing.behavioral_change_evaluation import BehavioralChangeEvaluator
from parlant.core.logging import Logger
from parlant.core.application import Application
from parlant.core.tags import TagStore
from parlant.dspy_integration.server_integration import setup_dspy_routes

ASGIApplication: TypeAlias = Callable[
    [
        Scope,
        Receive,
        Send,
    ],
    Awaitable[None],
]

__all__ = [
    'APIRouter',
    'ASGIApplication',
    'Application',
    'BehavioralChangeEvaluator',
    'Container',
    'CORSMiddleware',
    'FastAPI',
    'HTTPException',
    'Logger',
    'Receive',
    'Request',
    'Response',
    'RedirectResponse',
    'Scope',
    'Send',
    'StaticFiles',
    'WebSocketLogger',
    'agents',
    'customers',
    'fragments',
    'glossary',
    'guidelines',
    'index',
    'logs',
    'services',
    'sessions',
    'setup_dspy_routes',
    'tags',
    'variables',
    # Core types
    'AgentStore',
    'ContextVariableStore',
    'ContextualCorrelator',
    'CustomerStore',
    'EvaluationListener',
    'EvaluationStore',
    'FragmentStore',
    'GlossaryStore',
    'GuidelineConnectionStore',
    'GuidelineStore',
    'GuidelineToolAssociationStore',
    'ItemNotFoundError',
    'NLPService',
    'ServiceRegistry',
    'SessionListener',
    'SessionStore',
    'TagStore',
    'generate_id',
] 