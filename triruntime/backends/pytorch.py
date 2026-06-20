import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .base import BaseBackend
from typing import Any, Dict

class PyTorchBackend(BaseBackend):
    def __init__(self):
        self.base_model = None
        self.tokenizer = None
        self.namespaces = {}

    def load_base_model(self, model_name: str) -> None:
        print(f"[PyTorch Backend] Loading base model: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )

    def allocate_namespace(self, tenant_id: str) -> None:
        # Create a lightweight dictionary of pointers referencing the base model
        self.namespaces[tenant_id] = {
            name: param.clone().detach() 
            for name, param in self.base_model.named_parameters()
        }

    def route_pointers(self, tenant_id: str) -> Any:
        if tenant_id not in self.namespaces:
            raise KeyError(f"Namespace {tenant_id} does not exist.")
        # Simulating the context switch by returning the pointer dict
        return self.namespaces[tenant_id]

    def snapshot_state(self, tenant_id: str) -> Dict[str, Any]:
        return {
            k: v.clone() for k, v in self.namespaces[tenant_id].items()
        }

    def rollback_state(self, tenant_id: str, snapshot: Dict[str, Any]) -> None:
        for k in self.namespaces[tenant_id]:
            self.namespaces[tenant_id][k].copy_(snapshot[k])

    def free_namespace(self, tenant_id: str) -> None:
        if tenant_id in self.namespaces:
            del self.namespaces[tenant_id]