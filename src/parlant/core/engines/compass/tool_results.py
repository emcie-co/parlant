from parlant.core.engines.compass.tracing import CompassTracer
from parlant.core.pricing import PricingRegistry
from parlant.core.tools import ToolId, ToolResult
from parlant.core.tracer import Tracer


def record_tool_result(
    tracer: Tracer,
    pricing_registry: PricingRegistry,
    tool: ToolId,
    tool_call_id: str,
    result: ToolResult,
) -> None:
    pricing_registry.report_tool_costs(tracer.trace_id, result.costs)
    CompassTracer(tracer).tool_result(tool, tool_call_id, result)
