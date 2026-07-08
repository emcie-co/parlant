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

from enum import Enum
from typing import Annotated, Optional, TypeAlias

from fastapi import APIRouter, HTTPException, Path, Request, status
from pydantic import Field

from parlant.api.common import apigen_config
from parlant.api.authorization import AuthorizationPolicy, Operation
from parlant.core.agents import AgentId
from parlant.core.common import DefaultBaseModel, ItemNotFoundError, UniqueId
from parlant.core.services.training_service import TrainingJob, TrainingService, TrainingStatus

API_GROUP = "train"


class TrainingStatusDTO(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class TrainingCreationParamsDTO(DefaultBaseModel):
    agent_ids: list[str] = Field(
        default_factory=list,
        description="Agents to train; empty (or omitted) trains every agent",
    )


class TrainingJobDTO(DefaultBaseModel):
    id: str = Field(description="Unique identifier of the training job")
    status: TrainingStatusDTO = Field(description="Current status of the training job")
    percentage: float = Field(description="Completion percentage in the range [0, 100]")
    error: Optional[str] = Field(default=None, description="Failure reason, when status is failed")


JobIdPath: TypeAlias = Annotated[
    str,
    Path(description="Unique identifier of the training job"),
]


def _job_to_dto(job: TrainingJob) -> TrainingJobDTO:
    return TrainingJobDTO(
        id=job.id,
        status=TrainingStatusDTO(job.status.value),
        percentage=job.percentage,
        error=job.error,
    )


def create_router(
    authorization_policy: AuthorizationPolicy,
    training_service: TrainingService,
) -> APIRouter:
    router = APIRouter()

    @router.post(
        "",
        status_code=status.HTTP_201_CREATED,
        operation_id="create_training",
        response_model=TrainingJobDTO,
        **apigen_config(group_name=API_GROUP, method_name="create"),
    )
    async def create_training(
        request: Request,
        params: Optional[TrainingCreationParamsDTO] = None,
    ) -> TrainingJobDTO:
        """Starts a background training run.

        Trains every agent by default, or only the agents named in ``agent_ids``.
        Returns immediately with the new job, whose progress can be polled via
        ``GET /train/{job_id}``.
        """
        await authorization_policy.authorize(request=request, operation=Operation.CREATE_TRAINING)

        agent_ids = [AgentId(agent_id) for agent_id in (params.agent_ids if params else [])]

        try:
            job_id = await training_service.create_training_task(agent_ids=agent_ids or None)
        except ItemNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(exc),
            )

        return _job_to_dto(await training_service.read_training_job(job_id))

    @router.get(
        "/{job_id}",
        operation_id="read_training",
        response_model=TrainingJobDTO,
        responses={status.HTTP_404_NOT_FOUND: {"description": "Training job not found"}},
        **apigen_config(group_name=API_GROUP, method_name="retrieve"),
    )
    async def read_training(request: Request, job_id: JobIdPath) -> TrainingJobDTO:
        """Retrieves the current status and progress of a training job."""
        await authorization_policy.authorize(request=request, operation=Operation.READ_TRAINING)

        try:
            return _job_to_dto(await training_service.read_training_job(UniqueId(job_id)))
        except ItemNotFoundError:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Training job not found",
            )

    return router


__all__ = ["create_router", "TrainingStatus"]
