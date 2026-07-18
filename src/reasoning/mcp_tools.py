"""MCP-style tools the GemmaEngine can invoke."""
import json
import logging
from typing import Any, Dict, List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('HMMAF.Reasoning.MCPTools')


class MCPTools:
    def __init__(self, world_model=None):
        self.world_model = world_model
        self.tools = {
            'search_memory': self.search_memory,
            'update_world_model': self.update_world_model,
            'get_system_status': self.get_system_status,
            'speak': self.speak,
        }
        self.tts_callback = None

    def set_tts_callback(self, callback):
        """Allow the orchestrator/UI to install a TTS hook for the 'speak' tool."""
        self.tts_callback = callback

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        if tool_name not in self.tools:
            logger.error(f"Tool '{tool_name}' not found.")
            return {'error': f"Tool '{tool_name}' not found.", 'status': 'error'}
        logger.info(f"Calling tool: {tool_name} with arguments: {arguments}")
        try:
            return await self.tools[tool_name](**arguments)
        except TypeError as te:
            return {'error': f"Invalid arguments for '{tool_name}': {te}", 'status': 'error'}
        except Exception as e:
            logger.error(f"Error calling '{tool_name}': {e}")
            return {'error': str(e), 'status': 'error'}

    async def search_memory(self, query: str) -> Dict[str, Any]:
        if not self.world_model:
            return {'results': [], 'status': 'error', 'message': 'World Model not initialized'}
        results = await self.world_model.query(query)
        return {'results': results, 'status': 'success'}

    async def update_world_model(self, entity_id: str, attributes: Dict[str, Any]) -> Dict[str, Any]:
        if not self.world_model:
            return {'status': 'error', 'message': 'World Model not initialized', 'entity_id': entity_id}
        ok = await self.world_model.update(entity_id, attributes)
        return {'status': 'success' if ok else 'error', 'entity_id': entity_id}

    async def get_system_status(self) -> Dict[str, Any]:
        return {
            'status': 'operational',
            'components': {
                'orchestrator': 'running',
                'perception': 'active',
                'memory': 'active',
            },
        }

    async def speak(self, text: str) -> Dict[str, Any]:
        """Synthesize a spoken response to the user via the Expression worker."""
        if self.tts_callback is None:
            return {'status': 'error', 'message': 'No TTS callback registered'}
        try:
            await self.tts_callback(text)
            return {'status': 'success', 'spoken_text': text}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        raw = [
            {
                'name': 'search_memory',
                'description': 'Search the World Model for historical context or specific object states.',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'query': {'type': 'string', 'description': 'The semantic query to search for.'},
                    },
                    'required': ['query'],
                },
            },
            {
                'name': 'update_world_model',
                'description': 'Updates the current state of an entity in the World Model.',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'entity_id': {'type': 'string', 'description': "Unique entity id e.g. 'person_1'."},
                        'attributes': {'type': 'object', 'description': 'Key-value pairs of new state.'},
                    },
                    'required': ['entity_id', 'attributes'],
                },
            },
            {
                'name': 'get_system_status',
                'description': 'Return current operational status of the system.',
                'parameters': {'type': 'object', 'properties': {}},
            },
            {
                'name': 'speak',
                'description': 'Speak a short natural-language response aloud to the user via TTS.',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'text': {'type': 'string', 'description': 'The text to speak (1-3 sentences).'},
                    },
                    'required': ['text'],
                },
            },
        ]
        return [{'type': 'function', 'function': t} for t in raw]
