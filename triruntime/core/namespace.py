from triruntime.backends.base import BaseBackend

class NamespaceManager:
    """
    Manages the lifecycle of tenant namespaces regardless of the underlying backend.
    """
    def __init__(self, backend: BaseBackend):
        self.backend = backend

    def fork(self, tenant_id: str) -> None:
        print(f"[Kernel] Provisioning isolated namespace for: {tenant_id}")
        self.backend.allocate_namespace(tenant_id)

    def switch_context(self, tenant_id: str) -> None:
        # Route pointers to the active tenant
        return self.backend.route_pointers(tenant_id)

    def snapshot(self, tenant_id: str) -> dict:
        print(f"[Kernel] Capturing snapshot for: {tenant_id}")
        return self.backend.snapshot_state(tenant_id)

    def rollback(self, tenant_id: str, state_dict: dict) -> None:
        print(f"[Kernel] Rolling back state for: {tenant_id}")
        self.backend.rollback_state(tenant_id, state_dict)

    def kill(self, tenant_id: str) -> None:
        print(f"[Kernel] Destroying namespace: {tenant_id}")
        self.backend.free_namespace(tenant_id)