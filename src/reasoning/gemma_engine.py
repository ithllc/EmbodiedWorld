"""Gemma 4 reasoning engine via LiteLLM.

Configured for the workspace-owner's local LiteLLM gateway at
http://192.168.1.168:4000/v1, serving `gemma-4-e4b-multimodal`.
"""
import json
import logging
import asyncio
import os
from typing import Any, Dict, List, Optional

import litellm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('HMMAF.Reasoning.GemmaEngine')


DEFAULT_API_BASE = os.environ.get('HMMAF_LLM_API_BASE', 'http://192.168.1.168:4000/v1')
DEFAULT_MODEL = os.environ.get('HMMAF_LLM_MODEL', 'gemma-4-e4b-multimodal')
DEFAULT_API_KEY = os.environ.get('HMMAF_LLM_API_KEY', 'local-execution-key')


class GemmaEngine:
    def __init__(
        self,
        api_base: str = DEFAULT_API_BASE,
        model: str = DEFAULT_MODEL,
        api_key: str = DEFAULT_API_KEY,
        timeout: float = 60.0,
    ):
        self.api_base = api_base
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    async def reason(
        self,
        prompt: str,
        context: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        images: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Drive the LLM. Returns {reasoning, action, arguments}.

        `images` is a list of data URLs (`data:image/jpeg;base64,...`) or http URLs.
        When provided, the user message is built in OpenAI multimodal format.
        """
        if context:
            full_prompt = f"Context: {context}\nTask:\n{prompt}"
        else:
            full_prompt = prompt

        system_instruction = (
            "You are HMMAF, a high-performance multimodal reasoning agent. "
            "You receive structured vision events from a YOLO perception module and natural-language "
            "speech from the user. Be concise. When you choose to call a tool, only call one tool per turn. "
            "When you choose to respond to the user directly, do so naturally in 1-3 sentences."
        )

        if images:
            user_content: List[Dict[str, Any]] = [{"type": "text", "text": full_prompt}]
            for img in images:
                user_content.append({"type": "image_url", "image_url": {"url": img}})
            user_msg = {"role": "user", "content": user_content}
        else:
            user_msg = {"role": "user", "content": full_prompt}

        completion_kwargs: Dict[str, Any] = {
            "model": f"openai/{self.model}",
            "messages": [{"role": "system", "content": system_instruction}, user_msg],
            "api_base": self.api_base,
            "api_key": self.api_key,
            "timeout": self.timeout,
        }
        if tools:
            completion_kwargs["tools"] = tools

        logger.info("Sending reasoning request to Gemma 4 ...")
        try:
            response = await litellm.acompletion(**completion_kwargs)
            message = response.choices[0].message

            result: Dict[str, Any] = {}
            reasoning_text = getattr(message, 'reasoning', None) or message.content or ""
            result['reasoning'] = reasoning_text

            tool_calls = getattr(message, 'tool_calls', None)
            if tool_calls:
                tc = tool_calls[0]
                result['action'] = tc.function.name
                try:
                    result['arguments'] = json.loads(tc.function.arguments)
                except (TypeError, json.JSONDecodeError):
                    result['arguments'] = {}
            else:
                result['action'] = 'RESPOND'

            return result
        except Exception as e:
            logger.error(f"LLM Reason Error: {e}")
            raise


async def main():
    engine = GemmaEngine()
    result = await engine.reason("In one sentence, what is a person?")
    print("Result:", result)


if __name__ == '__main__':
    asyncio.run(main())
