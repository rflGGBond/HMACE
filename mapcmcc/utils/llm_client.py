import json
import os
from typing import Dict, Any, Optional
import warnings
import openai
import torch
import ast
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# # Try to import torch and transformers for local model support
# try:
    
#     _LOCAL_DEPS_AVAILABLE = True
# except ImportError:
#     _LOCAL_DEPS_AVAILABLE = False

class LLMClient:
    """
    A simple client to interact with an LLM provider (e.g., OpenAI, Anthropic, or Local).
    Supports local models deployed in a specific directory.
    """
    def __init__(self, provider: str = "local", api_key: Optional[str] = None, model: str = "Qwen2.5-14B", model_root: str = "../../../models"):
        self.provider = provider
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.model_root = model_root
        
        self.pipeline = None
        
        if self.provider == "local":
            self._init_local_model()

    def _init_local_model(self):
        """Initialize the local model and tokenizer."""
        model_path = os.path.join(self.model_root, self.model)
        if not os.path.exists(model_path):
            # Try to see if the user provided a full path or a relative path that exists
            if os.path.exists(self.model):
                model_path = self.model
            else:
                raise FileNotFoundError(f"Model path not found: {model_path}")
        
        print(f"Loading local model from {model_path}...")
        try:
            # Use device_map="auto" to handle large models if accelerate is installed
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            self.model_obj = AutoModelForCausalLM.from_pretrained(
                model_path, 
                device_map="auto", 
                torch_dtype="auto", 
                trust_remote_code=True
            )
            
            self.pipeline = pipeline(
                "text-generation",
                model=self.model_obj,
                tokenizer=self.tokenizer
            )
            print("Local model loaded successfully.")
        except Exception as e:
            raise RuntimeError(f"Failed to load local model: {e}")

    def get_completion(self, system_prompt: str, user_prompt: str, response_format: str = "json") -> str:
        """
        Sends a prompt to the LLM and returns the response content.
        """
        if self.provider == "mock":
            return self._mock_response(system_prompt, user_prompt)
        
        elif self.provider == "local":
            response = self._local_response(system_prompt, user_prompt, response_format)
            if response_format == "json":
                return self._clean_and_extract_json(response)
            return response

        elif self.provider == "openai":
            try:
                import openai
            except ImportError:
                raise ImportError("OpenAI provider requires 'openai' package. Please install it.")

            client = openai.OpenAI(api_key=self.api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
            
            # Prepare messages
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            kwargs = {
                "model": self.model,
                "messages": messages,
            }
            
            if response_format == "json":
                kwargs["response_format"] = {"type": "json_object"}
            
            response = client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content
            if response_format == "json":
                return self._clean_and_extract_json(content)
            return content
            
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _clean_and_extract_json(self, text: str) -> str:
        """
        Extracts JSON substring from text and attempts to fix common issues.
        """
        # Debug print to see what LLM is actually returning
        # print(f"DEBUG: Raw LLM response: {text[:200]}..." if len(text) > 200 else f"DEBUG: Raw LLM response: {text}")

        # Remove Markdown code blocks if present
        if "```json" in text:
            try:
                text = text.split("```json")[1].split("```")[0]
            except IndexError:
                pass
        elif "```" in text:
            try:
                text = text.split("```")[1].split("```")[0]
            except IndexError:
                pass
        
        text = text.strip()
        
        # Find the first '{' and the last '}'
        start = text.find("{")
        end = text.rfind("}")
        
        if start != -1 and end != -1:
            extracted_text = text[start:end+1]
            
            # 1. Try standard JSON load
            try:
                json.loads(extracted_text)
                return extracted_text
            except json.JSONDecodeError as e:
                # Handle "Extra data" error by truncating at the error position
                if e.msg.startswith("Extra data"):
                    try:
                        truncated_text = extracted_text[:e.pos]
                        json.loads(truncated_text)
                        return truncated_text
                    except:
                        pass
                pass
                
            # 2. Try ast.literal_eval (handles Python-style dicts with single quotes, etc.)
            try:
                # ast.literal_eval is safe for literal structures
                py_obj = ast.literal_eval(extracted_text)
                return json.dumps(py_obj)
            except (ValueError, SyntaxError):
                pass
                
            # 3. Simple manual fixes (last resort)
            # Replace single quotes with double quotes (risky if content has quotes)
            # This is a heuristic attempt
            try:
                fixed_text = extracted_text.replace("'", '"').replace("True", "true").replace("False", "false").replace("None", "null")
                json.loads(fixed_text)
                return fixed_text
            except json.JSONDecodeError:
                pass
                
            # If all parsing attempts fail, raise ValueError instead of returning invalid string
            raise ValueError(f"Failed to parse JSON from extracted text: {extracted_text[:100]}...")
            
        # If no JSON object found, raise ValueError to be caught by caller
        raise ValueError(f"No JSON object found in LLM response: {text[:100]}...")


    def _local_response(self, system_prompt: str, user_prompt: str, response_format: str) -> str:
        """
        Generates a response using the locally loaded model.
        """
        # Construct a prompt. This template might need adjustment based on the specific model (e.g. ChatML for Qwen)
        # For simplicity, we'll use a basic structure or the tokenizer's chat template if available.
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            # Try to use apply_chat_template if the tokenizer supports it
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            # Fallback for models without chat template in tokenizer config
            prompt = f"System: {system_prompt}\nUser: {user_prompt}\nAssistant:"

        # Generation parameters
        gen_kwargs = {
            "max_new_tokens": 512,
            "do_sample": True,
            "temperature": 0.8,
            "top_p": 0.9,
            "repetition_penalty": 1.1,
            "return_full_text": False,
        }
        
        outputs = self.pipeline(prompt, **gen_kwargs)
        generated_text = outputs[0]["generated_text"]
        
        # Extract the assistant's response. 
        # Since return_full_text=False, generated_text is just the new content.
        response = generated_text.strip()

        # If JSON is requested, try to ensure valid JSON (basic check)
        if response_format == "json":
            # Just return it, the prompt should have instructed JSON output. 
            # We could add a validator here if needed.
            pass
            
        return response

    def _mock_response(self, system_prompt: str, user_prompt: str) -> str:
        """
        Generates a fake JSON response based on keywords in the prompt.
        This allows testing the flow without paying for tokens.
        """
        # Detect if this is a Community Agent or Meta Agent request
        if "Community Agent" in system_prompt:
            # Simulate a decision to adjust parameters slightly
            return json.dumps({
                "reasoning": "Performance is stable, increasing exploration slightly.",
                "action_type": "adjust_parameters",
                "parameters": {
                    "cr1": 0.4,
                    "cr2": 0.4,
                    "beta": 2.5,
                    "alpha": 10.0
                },
                "candidate_seed_set": None
            })
        
        elif "Meta Agent" in system_prompt:
            # Simulate a decision to keep baselines
            return json.dumps({
                "reasoning": "Global convergence is proceeding normally. No merges needed yet.",
                "global_baselines": {
                    "cr1": 0.3, 
                    "cr2": 0.3,
                    "beta": 2.0,
                    "alpha": 12.0
                },
                "budget_adjustments": {},
                "merge_suggestions": []
            })
            
        return "{}"
