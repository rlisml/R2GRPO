import os
import sys
import threading
import asyncio
import uuid
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any, AsyncGenerator
import logging

# -----------------------------------------------------------------------------
# Dynamic Dependency Imports
# -----------------------------------------------------------------------------
try:
    from openai import OpenAI, AsyncOpenAI
except ImportError:
    pass

try:
    import google.generativeai as genai
except ImportError:
    pass

# vLLM & HF Imports
try:
    from vllm import LLM, SamplingParams
    # Import Async Engine
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    from vllm.engine.arg_utils import AsyncEngineArgs
except ImportError:
    pass

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
except ImportError:
    pass

# -----------------------------------------------------------------------------
# Core Implementation
# -----------------------------------------------------------------------------

@dataclass
class LLMResponse:
    text: str
    raw: Optional[Dict[str, Any]] = None

class CoreLLM:
    """
    Unified LLM core class supporting async concurrency.
    1. local      : Hugging Face Transformers (Thread Pool wrapped)
    2. local_vllm : vLLM (AsyncLLMEngine w/ Continuous Batching)
    3. api        : Official OpenAI API (Async Client)
    4. openrouter : OpenRouter API (Async Client)
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.mode = config.get('mode', 'api') 
        self.default_temperature = config.get('temperature', 0.7)
        self.default_max_new_tokens = config.get('max_new_tokens', 4096)
        
        self.client = None       # Sync Client
        self.aclient = None      # Async Client
        self.llm_engine = None   # vLLM Async Engine
        self.model = None        # HF Model
        self.tokenizer = None
        
        self._hf_lock = threading.Lock()
        
        logging.info(f"[{self.__class__.__name__}] Initializing in '{self.mode}' mode...")
        
        if self.mode == 'local_vllm':
            self._init_vllm_async()
        elif self.mode == 'local':
            self._init_huggingface()
        elif self.mode == 'api':
            self._init_official_api()
        elif self.mode == 'openrouter':
            self._init_openrouter()
        else:
            raise ValueError(f"Unknown LLM mode: {self.mode}")

    def _init_vllm_async(self):
        """Initialize vLLM Async Engine"""
        cfg = self.config.get('local_config', {})
        model_path = cfg.get('model_path', "")
        gpu_util = cfg.get('vllm_gpu_util', 0.90)
        max_len = cfg.get('max_model_len', 8192)
        
        logging.info(f"  > [vLLM-Async] Loading: {model_path}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Local model path not found: {model_path}")

        try:
            # Build async engine args
            engine_args = AsyncEngineArgs(
                model=model_path,
                trust_remote_code=True,
                gpu_memory_utilization=gpu_util,
                max_model_len=max_len,
                dtype="auto"
            )
            self.llm_engine = AsyncLLMEngine.from_engine_args(engine_args)
            
            # Load tokenizer for token counting auxiliary tasks
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        except NameError:
            raise ImportError("vLLM not installed.")

    def _init_huggingface(self):
        cfg = self.config.get('local_config', {})
        model_path = cfg.get('model_path', "")
        device_map = cfg.get('device_map', "auto")
        
        logging.info(f"  > [HF] Loading: {model_path}")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Local model path not found: {model_path}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, 
                device_map=device_map, 
                trust_remote_code=True,
                torch_dtype="auto"
            )
        except NameError:
            raise ImportError("Transformers/Torch not installed.")

    def _init_official_api(self):
        cfg = self.config.get('api_config', {})
        self.model_name = cfg.get('model_name', 'gpt-4o')
        api_key = os.environ.get("OPENAI_API_KEY") or cfg.get('api_key')
        base_url = cfg.get('base_url')

        logging.info(f"  > [Official API] Model: {self.model_name}")
        
        if not api_key:
            logging.warning("Warning: OPENAI_API_KEY not found in env or config.")

        try:
            self.client = OpenAI(api_key=api_key, base_url=base_url)
            self.aclient = AsyncOpenAI(api_key=api_key, base_url=base_url)
        except NameError:
            raise ImportError("openai package not installed.")

    def _init_openrouter(self):
        cfg = self.config.get('openrouter_config', {})
        self.model_name = cfg.get('model_name', 'openai/gpt-4o')
        api_key = os.environ.get("OPENROUTER_API_KEY") or cfg.get('api_key')
        base_url = cfg.get('base_url', "https://openrouter.ai/api/v1")
        
        logging.info(f"  > [OpenRouter] Model: {self.model_name}")
        
        if not api_key:
            logging.warning("Warning: OPENROUTER_API_KEY not found in env or config.")

        try:
            self.client = OpenAI(api_key=api_key, base_url=base_url)
            self.aclient = AsyncOpenAI(api_key=api_key, base_url=base_url)
        except NameError:
            raise ImportError("openai package not installed.")

    # =========================================================================
    # Async Generation Methods (Core)
    # =========================================================================
    async def generate_async(
        self, 
        prompt: str, 
        *, 
        system: Optional[str] = None, 
        temperature: Optional[float] = None,
        max_new_tokens: Optional[int] = None
    ) -> LLMResponse:
        """
        Unified async generation entry point.
        """
        if temperature is None: temperature = self.default_temperature
        if max_new_tokens is None: max_new_tokens = self.default_max_new_tokens

        if self.mode == 'local_vllm':
            return await self._generate_vllm_async_impl(prompt, system, temperature, max_new_tokens)
        elif self.mode == 'local':
            # HF local generation is blocking, run in thread pool.
            return await asyncio.to_thread(self._generate_hf, prompt, system, temperature, max_new_tokens)
        elif self.mode in ['api', 'openrouter']:
            return await self._generate_api_async_impl(prompt, system, temperature, max_new_tokens)
        else:
            return LLMResponse(text="")

    async def _generate_vllm_async_impl(self, prompt: str, system: Optional[str], temperature: float, max_tokens: int) -> LLMResponse:
        messages = []
        if system: messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        text_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        sampling_params = SamplingParams(
            temperature=temperature, 
            max_tokens=max_tokens
        )
        
        request_id = f"req-{uuid.uuid4()}"
        
        # AsyncLLMEngine.generate returns an AsyncGenerator
        results_generator = self.llm_engine.generate(text_prompt, sampling_params, request_id)
        
        final_output = None
        async for request_output in results_generator:
            final_output = request_output
            
        return LLMResponse(text=final_output.outputs[0].text)

    async def _generate_api_async_impl(self, prompt: str, system: Optional[str], temperature: float, max_tokens: int) -> LLMResponse:
        messages = []
        if system: messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        extra_headers = {}
        if self.mode == 'openrouter':
            extra_headers = {
                "HTTP-Referer": "https://github.com/data-gen-workflow",
                "X-Title": "DataGenWorkflow"
            }

        try:
            response = await self.aclient.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_headers=extra_headers
            )
            return LLMResponse(text=response.choices[0].message.content)
        except Exception as e:
            logging.error(f"API Async Error ({self.mode}): {e}")
            return LLMResponse(text="")

    # =========================================================================
    # Synchronous Generation Methods (Backward Compatibility)
    # =========================================================================
    def generate(
        self, 
        prompt: str, 
        *, 
        system: Optional[str] = None, 
        temperature: Optional[float] = None,
        max_new_tokens: Optional[int] = None
    ) -> LLMResponse:
        """
        Unified generation entry point. If in vLLM mode, forces execution via async loop to utilize AsyncLLMEngine.
        """
        if temperature is None: temperature = self.default_temperature
        if max_new_tokens is None: max_new_tokens = self.default_max_new_tokens

        # Dispatch call
        if self.mode == 'local_vllm':
            # vLLM now uses only async engine, must run loop
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            if loop.is_running():
                import nest_asyncio
                nest_asyncio.apply()
                return loop.run_until_complete(self.generate_async(prompt, system=system, temperature=temperature, max_new_tokens=max_new_tokens))
            else:
                return loop.run_until_complete(self.generate_async(prompt, system=system, temperature=temperature, max_new_tokens=max_new_tokens))
        
        elif self.mode == 'local':
            return self._generate_hf(prompt, system, temperature, max_new_tokens)
        elif self.mode in ['api', 'openrouter']:
            return self._generate_api_common(prompt, system, temperature, max_new_tokens)
        else:
            return LLMResponse(text="")

    def _generate_hf(self, prompt: str, system: Optional[str], temperature: float, max_tokens: int) -> LLMResponse:
        with self._hf_lock:
            messages = []
            if system: messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            
            inputs = self.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
            ).to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs, 
                    max_new_tokens=max_tokens, 
                    temperature=temperature,
                    do_sample=(temperature > 0), 
                    pad_token_id=self.tokenizer.eos_token_id
                )
            text = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
            return LLMResponse(text=text)

    def _generate_api_common(self, prompt: str, system: Optional[str], temperature: float, max_tokens: int) -> LLMResponse:
        messages = []
        if system: messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        extra_headers = {}
        if self.mode == 'openrouter':
            extra_headers = {
                "HTTP-Referer": "https://github.com/data-gen-workflow",
                "X-Title": "DataGenWorkflow"
            }

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_headers=extra_headers
            )
            return LLMResponse(text=response.choices[0].message.content)
        except Exception as e:
            logging.error(f"API Error ({self.mode}): {e}")
            return LLMResponse(text="")