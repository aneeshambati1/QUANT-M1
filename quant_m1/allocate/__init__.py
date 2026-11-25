"""Capital Allocation Super-Engine (Organ 7). Pure computation."""
from .engine import solve_allocation, AllocationResult, AllocationError

__all__ = ["solve_allocation", "AllocationResult", "AllocationError"]
