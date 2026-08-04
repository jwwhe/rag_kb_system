"""
================================================================================
  Layer 5 - 增强层: Web 搜索兜底（国内网络适配版）
  功能：
    - 知识库无匹配时自动联网搜索
    - 多搜索后端自动切换：Bing → DuckDuckGo → Baidu
    - 严格区分内网/外网知识，外网内容标注来源 URL
================================================================================
"""
import importlib
import re
import time
from typing import List, Dict, Any, Optional, Callable

import httpx
from langchain_core.documents import Document

from config.settings import EnhanceConfig, get_settings_cached
from utils.exceptions import EnhanceError
from utils.logger import get_logger

_pg = importlib.import_module("4_generate.prompts")
PromptTemplates = _pg.PromptTemplates

logger = get_logger(__name__)

# 搜索结果的正则模式（Bing）
_BING_RESULT_RE = re.compile(
    r'<li\s+class="b_algo"[^>]*>.*?'
    r'<h2><a\s+href="([^"]+)"[^>]*>(.*?)</a></h2>.*?'
    r'<p[^>]*>(.*?)</p>',
    re.DOTALL | re.IGNORECASE,
)

# 简易结果提取（备用）
_BING_LINK_RE = re.compile(
    r'<a\s+(?:class="[^"]*"?\s*)?href="(https?://[^"]+)"[^>]*>(.*?)</a>',
    re.DOTALL,
)


class WebFallback:
    """
    Web 搜索兜底，国内网络环境优化版。
    多后端自动切换：Bing（国内可用）→ DuckDuckGo → Baidu。
    """

    def __init__(self, config: Optional[EnhanceConfig] = None):
        self.config = config or get_settings_cached().enhance
        self._http_client = None
        logger.info(
            f"WebFallback 初始化完成 | 启用: {self.config.enable_web_fallback}"
        )

    @property
    def http_client(self) -> httpx.Client:
        if self._http_client is None:
            self._http_client = httpx.Client(
                timeout=float(self.config.web_search_timeout),
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/120.0.0.0 Safari/537.36"
                    ),
                    "Accept-Language": "zh-CN,zh;q=0.9",
                },
                follow_redirects=True,
            )
        return self._http_client

    # ==================== 搜索入口 ====================

    def search(self, query: str) -> List[Dict[str, Any]]:
        """
        搜索外网信息，自动选择可用后端。

        Args:
            query: 搜索查询

        Returns:
            List[Dict]: [{"title": ..., "url": ..., "snippet": ...}, ...]
        """
        if not self.config.enable_web_fallback:
            logger.info("Web 搜索已禁用")
            return []

        max_results = self.config.web_search_max_results

        # 按优先级尝试各搜索后端
        backends = [
            ("Bing", self._search_bing),
            ("DuckDuckGo", self._search_duckduckgo),
            ("Baidu", self._search_baidu),
        ]

        for name, func in backends:
            try:
                results = func(query, max_results)
                if results:
                    logger.info(f"Web 搜索({name}): {len(results)} 条结果")
                    return results
            except Exception as e:
                logger.warning(f"搜索后端 {name} 失败: {e}")
                continue

        logger.warning("所有搜索后端均失败")
        return []

    # ==================== Bing 搜索（国内首选）====================

    def _search_bing(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """
        Bing 搜索。国内可直接访问。

        Args:
            query:       搜索查询
            max_results: 最大结果数

        Returns:
            List[Dict]: 搜索结果列表
        """
        url = "https://www.bing.com/search"
        params = {
            "q": query,
            "count": str(max_results),
            "setlang": "zh-CN",
            "form": "QBLH",
        }

        resp = self.http_client.get(url, params=params)
        resp.raise_for_status()
        html = resp.text

        results = []
        # 使用正则提取搜索结果
        for match in _BING_RESULT_RE.finditer(html):
            href = match.group(1)
            title = self._strip_html(match.group(2))
            snippet = self._strip_html(match.group(3))
            if href and title and not href.startswith("javascript"):
                results.append({
                    "title": title.strip(),
                    "url": href.strip(),
                    "snippet": snippet.strip(),
                })
            if len(results) >= max_results:
                break

        # 正则提取失败时用简单模式
        if not results:
            results = self._extract_links_fallback(html, max_results)

        return results

    # ==================== DuckDuckGo（备用）====================

    def _search_duckduckgo(
        self, query: str, max_results: int
    ) -> List[Dict[str, Any]]:
        """DuckDuckGo 搜索（需要外网环境）。"""
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            logger.warning("duckduckgo-search 未安装")
            return []

        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", ""),
                })
        return results

    # ==================== Baidu 搜索（国内兜底）====================

    def _search_baidu(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """
        Baidu 搜索。纯国内环境最终兜底。
        注：Baidu 反爬严格，仅做简单尝试。
        """
        url = "https://www.baidu.com/s"
        params = {"wd": query, "rn": str(max_results)}

        try:
            resp = self.http_client.get(url, params=params)
            resp.raise_for_status()
            return self._extract_links_fallback(resp.text, max_results)
        except Exception:
            return []

    # ==================== 生成回答 ====================

    def build_web_answer(
        self,
        question: str,
        web_results: List[Dict[str, Any]],
        llm_call: Callable[[str], str],
    ) -> str:
        """
        基于网络搜索结果生成回答。

        Args:
            question:    用户问题
            web_results: 网络搜索结果
            llm_call:    LLM 调用函数

        Returns:
            str: 带外网标注的答案
        """
        if not web_results:
            return PromptTemplates.NO_KNOWLEDGE_RESPONSE

        web_context = self._format_web_results(web_results)

        system_prompt = PromptTemplates.WEB_FALLBACK_SYSTEM_PROMPT
        user_prompt = PromptTemplates.WEB_FALLBACK_USER_PROMPT_TEMPLATE.format(
            web_results=web_context, question=question
        )

        try:
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            answer = llm_call(full_prompt)
            return (
                f"{PromptTemplates.EXTERNAL_SOURCE_PREFIX}\n\n{answer.strip()}"
            )
        except Exception as e:
            raise EnhanceError(f"Web 兜底回答生成失败: {str(e)}") from e

    # ==================== 结果转换 ====================

    def web_results_to_documents(
        self, web_results: List[Dict[str, Any]]
    ) -> List[Document]:
        """将网络搜索结果转为 Document 列表（供 CitationTracer 使用）。"""
        documents = []
        for i, result in enumerate(web_results, 1):
            doc = Document(
                page_content=(
                    f"{result.get('title', '')}\n{result.get('snippet', '')}"
                ),
                metadata={
                    "file_name": result.get("url", "未知URL"),
                    "page": "N/A",
                    "source_type": "external",
                    "citation_id": i,
                    "similarity": 0.0,
                    "url": result.get("url", ""),
                    "title": result.get("title", ""),
                },
            )
            documents.append(doc)
        return documents

    # ==================== 私有方法 ====================

    def _format_web_results(self, web_results: List[Dict[str, Any]]) -> str:
        """格式化网络搜索结果为 LLM 可读文本。"""
        parts = []
        for i, r in enumerate(web_results, 1):
            parts.append(
                f"[来源-{i}]\n"
                f"标题: {r.get('title', 'N/A')}\n"
                f"链接: {r.get('url', 'N/A')}\n"
                f"内容: {r.get('snippet', 'N/A')}"
            )
        return "\n\n".join(parts)

    def _extract_links_fallback(
        self, html: str, max_results: int
    ) -> List[Dict[str, Any]]:
        """简易链接提取（各搜索引擎通用备用方案）。"""
        results = []
        seen = set()
        for match in _BING_LINK_RE.finditer(html):
            href = match.group(1)
            text = self._strip_html(match.group(2))
            if (
                href
                and text
                and len(text) > 5
                and not href.startswith("javascript")
                and not any(skip in href.lower() for skip in
                           ["bing.com", "microsoft.com", "baidu.com", ".gov.cn/"])
            ):
                if href not in seen:
                    seen.add(href)
                    results.append({
                        "title": text.strip()[:100],
                        "url": href.strip(),
                        "snippet": text.strip(),
                    })
            if len(results) >= max_results:
                break
        return results

    @staticmethod
    def _strip_html(text: str) -> str:
        """去除 HTML 标签。"""
        return re.sub(r"<[^>]+>", "", text).replace("&quot;", '"')
