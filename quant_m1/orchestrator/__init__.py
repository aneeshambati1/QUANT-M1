"""Orchestrator: decision cycle, scheduling, signal mapping, execution wiring.

run_cycle remains pure. generate_intents and dispatch_intents are the explicit
execution boundary and always route through the Kernel.
"""
from .schedule import SessionSchedule, Phase
from .cycle import CycleRecord, run_cycle, apply_realized
from .mapping import blend_signals
from .execution import generate_intents, dispatch_intents, ExecutionReport

__all__ = [
    "SessionSchedule", "Phase", "CycleRecord", "run_cycle", "apply_realized",
    "blend_signals", "generate_intents", "dispatch_intents", "ExecutionReport",
]