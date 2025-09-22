from core.model_runtime.model_providers.__base.large_language_model import LargeLanguageModel
from core.model_runtime.model_providers.__base.ai_model import LLMResult
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class OmanCultureLargeLanguageModel(LargeLanguageModel):
    def _load_model(self, model_path):
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
        return model, tokenizer

    def _invoke(self, model, credentials, prompt_messages, model_parameters, tools=None, stop=None, stream=True, user=None):
        # Load model and tokenizer if not already loaded
        if not hasattr(self, "_hf_model"):
            self._hf_model, self._hf_tokenizer = self._load_model(credentials["model_path"])
        # Concatenate all prompt messages for chat-style input
        if prompt_messages:
            prompt = "\n".join([msg.content for msg in prompt_messages if hasattr(msg, "content")])
        else:
            prompt = ""
        # Tokenize and move to model device
        inputs = self._hf_tokenizer(prompt, return_tensors="pt").to(self._hf_model.device)
        with torch.no_grad():
            output = self._hf_model.generate(
                **inputs,
                max_new_tokens=model_parameters.get("max_tokens", 128),
                temperature=model_parameters.get("temperature", 0.7),
                top_p=model_parameters.get("top_p", 0.95),
                do_sample=True,
            )
        result = self._hf_tokenizer.decode(output[0], skip_special_tokens=True)
        return LLMResult(text=result)

    def get_num_tokens(self, model, credentials, prompt_messages, tools=None):
        # Use GPT-2 tokenizer as an approximation
        if prompt_messages:
            prompt = "\n".join([msg.content for msg in prompt_messages if hasattr(msg, "content")])
        else:
            prompt = ""
        return self._get_num_tokens_by_gpt2(prompt)

    def validate_credentials(self, model, credentials):
        # Check if model path exists and is loadable
        import os
        if not os.path.exists(credentials["model_path"]):
            raise ValueError("Model path does not exist.")

    def get_customizable_model_schema(self, model, credentials):
        from core.model_runtime.model_providers.__base.ai_model import AIModelEntity, ParameterRule, ParameterType, I18nObject, FetchFrom, ModelPropertyKey, ModelType
        rules = [
            ParameterRule(
                name='temperature', type=ParameterType.FLOAT, use_template='temperature',
                label=I18nObject(en_US='Temperature')
            ),
            ParameterRule(
                name='top_p', type=ParameterType.FLOAT, use_template='top_p',
                label=I18nObject(en_US='Top P')
            ),
            ParameterRule(
                name='max_tokens', type=ParameterType.INT, use_template='max_tokens',
                min=1, default=128,
                label=I18nObject(en_US='Max Tokens')
            ),
        ]
        entity = AIModelEntity(
            model=model,
            label=I18nObject(en_US=model),
            fetch_from=FetchFrom.CUSTOMIZABLE_MODEL,
            model_type=ModelType.LLM,
            model_properties={ModelPropertyKey.MODE: ModelType.LLM},
            parameter_rules=rules
        )
        return entity

    @property
    def _invoke_error_mapping(self):
        from core.model_runtime.model_providers.__base.ai_model import InvokeConnectionError, InvokeServerUnavailableError
        import requests
        return {
            InvokeConnectionError: [requests.exceptions.ConnectionError],
            InvokeServerUnavailableError: [RuntimeError],
        }
