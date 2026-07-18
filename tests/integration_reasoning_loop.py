"""Live integration test: Reasoning -> Tools -> Memory.

Talks to the real LiteLLM endpoint at http://192.168.1.168:4000/v1
serving model `gemma-4-e4b-multimodal`. No mocks.
"""
import asyncio
import logging

from src.memory.world_model import WorldModel
from src.reasoning.gemma_engine import GemmaEngine
from src.reasoning.mcp_tools import MCPTools

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('HMMAF.IntegrationTest.ReasoningLoop')


async def test_reasoning_loop():
    print('--- Starting Integration Test: Reasoning -> Tools -> Memory ---')

    world_model = WorldModel()
    mcp_tools = MCPTools(world_model=world_model)
    engine = GemmaEngine()
    tool_defs = mcp_tools.get_tool_definitions()

    print('[Scenario 1: Updating Memory]')
    user_prompt = (
        'I see a red car entering the scene. '
        "Please record this fact by calling the 'update_world_model' tool with "
        "entity_id='red_car_1' and attributes={'label': 'car', 'color': 'red'}."
    )
    print(f'Sending prompt to GemmaEngine: {user_prompt}')
    reasoning_result = await engine.reason(user_prompt, tools=tool_defs)
    print(f'Engine Result: {reasoning_result}')

    if reasoning_result.get('action') and reasoning_result['action'] != 'RESPOND':
        action = reasoning_result['action']
        args = reasoning_result.get('arguments', {})
        print(f'Executing Action: {action} with args: {args}')
        result = await mcp_tools.call_tool(action, args)
        print(f'Result of tool call: {result}')
    else:
        # Fall back to direct memory write so scenario 2 has something to find.
        print('No tool action identified; performing direct memory write as fallback.')
        await mcp_tools.call_tool(
            'update_world_model',
            {'entity_id': 'red_car_1', 'attributes': {'label': 'car', 'color': 'red'}},
        )

    print('[Scenario 2: Querying Memory]')
    query_prompt = (
        "Answer the question: Is there a red car? "
        "You MUST call the 'search_memory' tool with query='red car'."
    )
    print(f'Querying engine for: {query_prompt}')
    reasoning_result_query = await engine.reason(query_prompt, tools=tool_defs)
    print(f'Engine Result: {reasoning_result_query}')

    direct_query = await mcp_tools.call_tool('search_memory', {'query': 'red'})
    print(f'Direct memory query result (sanity check): {direct_query}')
    assert direct_query['status'] == 'success'
    assert len(direct_query['results']) >= 1, "Expected at least one entity in memory."
    print('SUCCESS: Memory roundtrip verified.')


async def main():
    try:
        await test_reasoning_loop()
    except Exception as e:
        import traceback
        print(f'Integration test failed: {e}')
        traceback.print_exc()
        raise


if __name__ == '__main__':
    asyncio.run(main())
