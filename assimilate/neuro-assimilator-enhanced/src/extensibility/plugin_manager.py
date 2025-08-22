from typing import List, Dict, Any
import yaml
import os

class PluginManager:
    def __init__(self, manifest_path: str):
        self.manifest_path = manifest_path
        self.plugins = self.load_plugins()

    def load_plugins(self) -> Dict[str, Any]:
        if not os.path.exists(self.manifest_path):
            return {}
        with open(self.manifest_path, 'r') as file:
            return yaml.safe_load(file)

    def register_plugin(self, plugin_name: str, plugin_info: Dict[str, Any]) -> None:
        self.plugins[plugin_name] = plugin_info
        self.save_plugins()

    def unregister_plugin(self, plugin_name: str) -> None:
        if plugin_name in self.plugins:
            del self.plugins[plugin_name]
            self.save_plugins()

    def save_plugins(self) -> None:
        with open(self.manifest_path, 'w') as file:
            yaml.dump(self.plugins, file)

    def get_plugin(self, plugin_name: str) -> Any:
        return self.plugins.get(plugin_name)

    def list_plugins(self) -> List[str]:
        return list(self.plugins.keys())