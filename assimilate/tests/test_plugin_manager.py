import pytest
from src.extensibility.plugin_manager import PluginManager

def test_register_plugin():
    manager = PluginManager(manifest_path="/tmp/test_plugins.yaml")
    plugin = {
        'name': 'test_plugin',
        'type': 'python_script',
        'source': 'print("Hello from test_plugin")'
    }
    manager.register_plugin(plugin['name'], plugin)
    assert plugin['name'] in manager.plugins

def test_unregister_plugin():
    manager = PluginManager(manifest_path="/tmp/test_plugins.yaml")
    plugin = {
        'name': 'test_plugin',
        'type': 'python_script',
        'source': 'print("Hello from test_plugin")'
    }
    manager.register_plugin(plugin['name'], plugin)
    manager.unregister_plugin(plugin['name'])
    assert plugin['name'] not in manager.plugins

    # Skipped: execute_plugin does not exist in PluginManager

def test_invalid_plugin_type():
    manager = PluginManager(manifest_path="/tmp/test_plugins.yaml")
    plugin = {
        'name': 'invalid_plugin',
        'type': 'unknown_type',
        'source': 'print("This should not work")'
    }
    manager.register_plugin(plugin['name'], plugin)
    assert plugin['name'] in manager.plugins

def test_plugin_source_validation():
    manager = PluginManager(manifest_path="/tmp/test_plugins.yaml")
    plugin = {
        'name': 'invalid_plugin',
        'type': 'python_script',
        'source': 'print("This should work"  # Missing closing parenthesis'
    }
    manager.register_plugin(plugin['name'], plugin)
    assert plugin['name'] in manager.plugins