from abc import ABC, abstractmethod
from typing import Any, Dict

class BaseBackend(ABC):
    """
    The universal interface between TriRuntime's OS-level logic 
    and the underlying hardware/inference engine.
    """
    
    @abstractmethod
    def load_base_model(self, model_name: str) -> None:
        """Loads the shared foundation weights into memory."""
        pass

    @abstractmethod
    def allocate_namespace(self, tenant_id: str) -> None:
        """Provisions a new isolated memory space for a tenant."""
        pass

    @abstractmethod
    def route_pointers(self, tenant_id: str) -> Any:
        """Executes the O(1) context switch to route requests to the tenant."""
        pass

    @abstractmethod
    def snapshot_state(self, tenant_id: str) -> Dict[str, Any]:
        """Captures the current parameter state of a namespace."""
        pass

    @abstractmethod
    def rollback_state(self, tenant_id: str, snapshot: Dict[str, Any]) -> None:
        """Restores a namespace to a previous state instantly."""
        pass

    @abstractmethod
    def free_namespace(self, tenant_id: str) -> None:
        """Destroys a tenant's namespace and frees the memory."""
        pass