import asyncio
import json
import httpx
from mcp.server import Server
from mcp.types import Tool, TextContent

app = Server("lingbot-mcp")

# Public endpoint of the RunPod
RUNPOD_API_URL = "https://ploz0v5qipca1f-8888.proxy.runpod.net/generate"

@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="generate_lingbot_video",
            description="Generate a video using LingBot-World 14B.",
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "The prompt describing the video to generate"},
                    "image_path": {"type": "string", "description": "Path to the image reference (e.g. 'examples/03/image.jpg')"},
                    "action_path": {"type": "string", "description": "Path to the action reference (e.g. 'examples/03')"},
                    "frame_num": {"type": "integer", "description": "Number of frames to generate (e.g. 361)"}
                },
                "required": ["prompt"]
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name != "generate_lingbot_video":
        raise ValueError(f"Unknown tool: {name}")

    payload = {
        "prompt": arguments.get("prompt"),
    }
    if "image_path" in arguments:
        payload["image_path"] = arguments["image_path"]
    if "action_path" in arguments:
        payload["action_path"] = arguments["action_path"]
    if "frame_num" in arguments:
        payload["frame_num"] = arguments["frame_num"]

    async with httpx.AsyncClient(timeout=3600.0) as client:
        try:
            response = await client.post(RUNPOD_API_URL, json=payload)
            response.raise_for_status()
            data = response.json()
            if data.get("status") == "success":
                return [TextContent(type="text", text=f"Success! Output:\n{data.get('output')}")]
            else:
                return [TextContent(type="text", text=f"Generation failed:\n{data.get('error')}")]
        except Exception as e:
            return [TextContent(type="text", text=f"Error connecting to RunPod server: {str(e)}")]

async def main():
    try:
        from mcp.server.stdio import stdio_server
    except ImportError:
        print("MCP SDK not found. Install it with: pip install 'mcp[cli]' httpx")
        return

    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())
