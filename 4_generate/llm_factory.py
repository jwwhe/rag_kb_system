"""
================================================================================
  Layer 4 - 生成层: 多 LLM 统一工厂
  功能：
    - DeepSeek / Qwen / Ollama 统一封装
    - 配置一键切换
    - 统一的 llm_call(prompt) -> str 接口
================================================================================
"""
from typing import Optional, Callable

from openai import OpenAI

from config.settings import LLMConfig, Settings, get_settings_cached
from utils.exceptions import GenerationError
from utils.logger import get_logger

logger = get_logger(__name__)


class LLMFactory:
    """
    多 LLM 统一工厂。
    支持 DeepSeek、Qwen（兼容 OpenAI API）、Ollama 本地模型。
    所有 LLM 通过统一的 callable 接口调用。
    """

    def __init__(self, config: Optional[LLMConfig] = None, settings: Optional[Settings] = None):
        """
        初始化 LLM 工厂。

        Args:
            config:   LLM 配置
            settings: 全局配置
        """
        self.settings = settings or get_settings_cached()
        self.config = config or self.settings.llm
        self._client: Optional[OpenAI] = None
        self._client_key: str = ""
        self._ollama_client = None
        self._current_model: str = ""
        self._chat_model = None
        self._chat_model_key: str = ""

        logger.info(
            f"LLMFactory 初始化完成 | 当前 LLM: {self.config.active_llm}"
        )

    def get_llm_callable(self) -> Callable[[str], str]:
        """
        获取统一的 LLM 调用函数。

        Returns:
            Callable[[str], str]: LLM 调用函数，输入 prompt 返回 response
        """
        # 每次调用前确保客户端配置最新（支持运行时切换）
        self._init_client()

        def llm_call(prompt: str) -> str:
            """统一的 LLM 调用接口。"""
            return self._call(prompt)

        return llm_call

    def call(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        直接调用 LLM。

        Args:
            prompt:        用户提示词
            system_prompt: 系统提示词（可选）

        Returns:
            str: LLM 响应文本
        """
        self._init_client()
        return self._call(prompt, system_prompt)

    def get_chat_model(self):
        """
        获取 LangChain 原生 ChatModel（用于 LCEL 链 + RunnableWithMessageHistory）。
        返回 ChatOpenAI 或 ChatOllama，与当前 active_llm 配置保持一致。
        同一模型只构建一次并复用（内部持有 HTTP 连接池，重复构建会丢失 keep-alive）。

        Returns:
            BaseChatModel: LangChain 聊天模型实例
        """
        active = self.config.active_llm

        if active in ("deepseek", "qwen"):
            from langchain_openai import ChatOpenAI
            if active == "deepseek":
                api_key = self.config.deepseek_api_key
                api_base = self.config.deepseek_api_base
                model = self.config.deepseek_model
            else:
                api_key = self.config.qwen_api_key
                api_base = self.config.qwen_api_base
                model = self.config.qwen_model

            cache_key = f"{active}|{api_base}|{model}"
            if self._chat_model_key == cache_key and self._chat_model is not None:
                return self._chat_model
            if not api_key:
                raise GenerationError(
                    f"{active.upper()} API Key 未配置，请设置环境变量"
                )
            chat_model = ChatOpenAI(
                api_key=api_key,
                base_url=api_base,
                model=model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                timeout=self.config.request_timeout,
            )

        elif active == "ollama":
            from langchain_ollama import ChatOllama

            cache_key = f"ollama|{self.config.ollama_host}|{self.config.ollama_model}"
            if self._chat_model_key == cache_key and self._chat_model is not None:
                return self._chat_model
            chat_model = ChatOllama(
                base_url=self.config.ollama_host,
                model=self.config.ollama_model,
                temperature=self.config.temperature,
                num_predict=self.config.max_tokens,
            )

        else:
            raise GenerationError(f"不支持的 LLM 类型: {active}")

        self._chat_model = chat_model
        self._chat_model_key = cache_key
        return chat_model

    def _init_client(self):
        """初始化或更新 LLM 客户端连接。"""
        active = self.config.active_llm

        if active == "ollama":
            self._init_ollama()
        elif active in ("deepseek", "qwen"):
            self._init_openai_compatible(active)
        else:
            raise GenerationError(f"不支持的 LLM 类型: {active}")

    def _init_openai_compatible(self, llm_type: str):
        """初始化 OpenAI 兼容客户端（DeepSeek / Qwen）。"""
        if llm_type == "deepseek":
            api_key = self.config.deepseek_api_key
            api_base = self.config.deepseek_api_base
            model = self.config.deepseek_model
        else:  # qwen
            api_key = self.config.qwen_api_key
            api_base = self.config.qwen_api_base
            model = self.config.qwen_model

        if not api_key:
            raise GenerationError(
                f"{llm_type.upper()} API Key 未配置，请设置环境变量或修改配置文件"
            )

        # 配置未变时复用已有客户端（新建 OpenAI 客户端会重建连接池，
        # 每次问答都握手 TLS 会额外付出数百毫秒）
        client_key = f"{llm_type}|{api_base}|{model}"
        if self._client_key == client_key and self._client is not None:
            return

        self._client = OpenAI(api_key=api_key, base_url=api_base)
        self._client_key = client_key
        self._current_model = model
        self._ollama_client = None

    def _init_ollama(self):
        """初始化 Ollama 本地模型连接。"""
        try:
            import ollama
            self._ollama_client = ollama
            self._current_model = self.config.ollama_model
            self._client = None
            self._client_key = ""
        except ImportError:
            raise GenerationError("ollama 库未安装，请执行: pip install ollama")

    def _call(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        实际 LLM 调用逻辑。

        Args:
            prompt:        用户提示词
            system_prompt: 系统提示词

        Returns:
            str: LLM 响应

        Raises:
            GenerationError: 调用失败（含重试）
        """
        max_retries = self.config.max_retries

        for attempt in range(1, max_retries + 1):
            try:
                if self._ollama_client is not None:
                    return self._call_ollama(prompt, system_prompt)
                elif self._client is not None:
                    return self._call_openai(prompt, system_prompt)
                else:
                    raise GenerationError("LLM 客户端未初始化")
            except GenerationError:
                raise
            except Exception as e:
                logger.warning(
                    f"LLM 调用失败 (尝试 {attempt}/{max_retries}): {e}"
                )
                if attempt >= max_retries:
                    raise GenerationError(
                        f"LLM 调用失败（已重试 {max_retries} 次）: {str(e)}"
                    ) from e

        raise GenerationError("LLM 调用失败：未知错误")

    def _call_openai(
        self, prompt: str, system_prompt: Optional[str] = None
    ) -> str:
        """OpenAI 兼容 API 调用。"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self._client.chat.completions.create(
            model=self._current_model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            timeout=self.config.request_timeout,
        )
        return response.choices[0].message.content or ""

    def _call_ollama(
        self, prompt: str, system_prompt: Optional[str] = None
    ) -> str:
        """Ollama 本地模型调用。"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Ollama Python API
        response = self._ollama_client.chat(
            model=self._current_model,
            messages=messages,
            options={
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
            },
        )
        return response.get("message", {}).get("content", "")


# ========== 全局单例 ==========

_llm_factory_instance: Optional[LLMFactory] = None


def get_llm() -> LLMFactory:
    """
    获取 LLM 工厂全局单例。

    Returns:
        LLMFactory: LLM 工厂实例
    """
    global _llm_factory_instance
    if _llm_factory_instance is None:
        _llm_factory_instance = LLMFactory()
    return _llm_factory_instance
