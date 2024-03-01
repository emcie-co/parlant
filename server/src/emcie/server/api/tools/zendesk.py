
import datetime
from typing import List, Literal, NewType
from pydantic import BaseModel


TicketId = NewType("TicketId", str)
TicketStatus = Literal["pending", "open", "solved", "closed"]


class TicketDTO(BaseModel):
    id: TicketId
    status: TicketStatus
    subject: str
    description: str
    priority: str
    tags: list
    created_at: datetime
    updated_at: datetime
    assignee_id: int
    requester_id: int

class ListTickets(BaseModel):
    messages: List[TicketDTO]