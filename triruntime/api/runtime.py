from triruntime.backends.pytorch import PyTorchBackend
from triruntime.core.namespace import NamespaceManager

class TriRuntime:
    """
    The main public interface for TriLab's multi-tenant AI operating system.
    """
    def __init__(self, model_name: str, backend_type: str = "pytorch"):
        print(f"[*] Booting TriRuntime (Backend: {backend_type.upper()})...")
        
        # 1. Initialize the correct backend muscle
        if backend_type == "pytorch":
            self.backend = PyTorchBackend()
        else:
            raise ValueError(f"Backend '{backend_type}' is not supported yet.")
        
        # 2. Load the base foundation model into the GPU
        self.backend.load_base_model(model_name)
        
        # 3. Connect the Muscle to our Brain (Namespace Manager)
        self.namespaces = NamespaceManager(self.backend)
        
        # Expose the tokenizer for easy access
        self.tokenizer = self.backend.tokenizer