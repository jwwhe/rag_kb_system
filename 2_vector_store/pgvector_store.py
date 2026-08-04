"""
================================================================================
  Layer 2 - 向量存储层: PGvector 实现（生产环境）
  功能：
    - 基于 PostgreSQL + pgvector 扩展的向量存储
    - 完整实现 BaseVectorStore 全部接口
    - 支持余弦相似度检索（<=> 算子）
    - 支持元数据过滤（source_type / file_name / knowledge_base）
  面试要点：
    - pgvector 是 PostgreSQL 的向量扩展，安装：CREATE EXTENSION vector;
    - 向量类型 vector(1024) 对应 bge-m3 输出维度
    - 余弦距离算子：<=> （值越小越相似，需转换为 similarity = 1 - distance）
    - 索引：HNSW 或 IVFFlat，本实现用 HNSW
================================================================================
"""
import uuid
import json
from typing import List, Dict, Optional, Any

from langchain_core.documents import Document

from config.settings import VectorStoreConfig, get_settings_cached
from .base import BaseVectorStore
from utils.exceptions import VectorStoreError
from utils.logger import get_logger

logger = get_logger(__name__)


class PGVectorStore(BaseVectorStore):
    """
    基于 PostgreSQL + pgvector 的向量存储实现。
    用于生产环境，支持 ACID 事务、丰富元数据过滤、SQL 生态。
    """

    def __init__(self, config: Optional[VectorStoreConfig] = None):
        """
        初始化 PGvector 向量存储。

        Args:
            config: 向量存储配置（默认从全局配置读取）

        Raises:
            VectorStoreError: 数据库连接失败或 pgvector 扩展缺失
        """
        self.config = config or get_settings_cached().vector_store

        try:
            import psycopg
            from pgvector.psycopg import register_vector
        except ImportError as e:
            raise VectorStoreError(
                "缺少依赖 psycopg / pgvector，请执行: pip install psycopg[binary] pgvector"
            ) from e

        # 构建连接字符串
        dsn = (
            f"host={self.config.pg_host} port={self.config.pg_port} "
            f"dbname={self.config.pg_database} user={self.config.pg_user} "
            f"password={self.config.pg_password}"
        )

        try:
            # psycopg v3，autocommit=True 避免事务包装 DDL
            self._conn = psycopg.connect(dsn, autocommit=True)
            # 注册 pgvector 类型，使 psycopg 能自动序列化 list -> vector
            register_vector(self._conn)

            # 验证连接 + 扩展
            with self._conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")

            # 确保表和索引存在
            self._ensure_schema()

            with self._conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM {self.config.pg_table_name}")
                count = cur.fetchone()[0]

            logger.info(
                f"PGVectorStore 初始化完成 | "
                f"地址: {self.config.pg_host}:{self.config.pg_port} | "
                f"DB: {self.config.pg_database} | "
                f"表: {self.config.pg_table_name} | "
                f"当前文档数: {count}"
            )
        except VectorStoreError:
            raise
        except Exception as e:
            raise VectorStoreError(
                f"PGvector 连接失败 [{self.config.pg_host}:{self.config.pg_port}]: {str(e)}"
            ) from e

    # ==================== 入库 ====================

    def add_documents(self, documents: List[Document]) -> bool:
        """
        批量入库带嵌入向量的文档。

        Args:
            documents: 携带 embedding 的 Document 列表

        Returns:
            bool: 入库成功
        """
        if not documents:
            logger.warning("文档列表为空，跳过入库")
            return False

        table = self.config.pg_table_name
        rows = []
        for doc in documents:
            emb = doc.metadata.pop("embedding", None)
            if emb is None:
                logger.warning(f"文档 {doc.metadata.get('chunk_id', '?')} 缺少嵌入向量，跳过")
                continue

            chunk_id = doc.metadata.get("chunk_id", str(uuid.uuid4()))
            # 提取关键字段（便于过滤检索），其余打包到 meta_json
            file_name = doc.metadata.get("file_name", "unknown")
            page = doc.metadata.get("page", 0)
            doc_id = doc.metadata.get("doc_id", "")
            source_type = doc.metadata.get("source_type", "论文原文")
            knowledge_base = doc.metadata.get("knowledge_base", "default")

            rows.append((
                chunk_id, doc.page_content, emb, file_name, page,
                doc_id, source_type, knowledge_base, json.dumps(doc.metadata, ensure_ascii=False)
            ))

        if not rows:
            raise VectorStoreError("没有有效的文档可入库（缺少嵌入向量）")

        sql = f"""
            INSERT INTO {table}
                (chunk_id, content, embedding, file_name, page,
                 doc_id, source_type, knowledge_base, meta_json)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (chunk_id) DO UPDATE SET
                content = EXCLUDED.content,
                embedding = EXCLUDED.embedding,
                file_name = EXCLUDED.file_name,
                page = EXCLUDED.page,
                doc_id = EXCLUDED.doc_id,
                source_type = EXCLUDED.source_type,
                knowledge_base = EXCLUDED.knowledge_base,
                meta_json = EXCLUDED.meta_json
        """
        try:
            with self._conn.cursor() as cur:
                cur.executemany(sql, rows)
            logger.info(f"PGvector 入库成功: {len(rows)} 条文档")
            return True
        except Exception as e:
            raise VectorStoreError(f"PGvector 入库失败: {str(e)}") from e

    # ==================== 删除 ====================

    def delete_by_file(self, file_name: str) -> int:
        """
        按文件名删除该文件的所有向量块。

        Args:
            file_name: 源文件名

        Returns:
            int: 删除的块数量
        """
        table = self.config.pg_table_name
        try:
            with self._conn.cursor() as cur:
                cur.execute(
                    f"DELETE FROM {table} WHERE file_name = %s RETURNING chunk_id",
                    (file_name,),
                )
                deleted = cur.fetchall()
            logger.info(f"已删除文件 '{file_name}' 的 {len(deleted)} 个向量块")
            return len(deleted)
        except Exception as e:
            raise VectorStoreError(f"删除文件失败: {str(e)}") from e

    def clear_collection(self) -> bool:
        """
        清空当前知识库所有数据（TRUNCATE）。
        """
        table = self.config.pg_table_name
        try:
            with self._conn.cursor() as cur:
                cur.execute(f"TRUNCATE TABLE {table}")
            logger.info(f"知识库已清空: {table}")
            return True
        except Exception as e:
            raise VectorStoreError(f"清空知识库失败: {str(e)}") from e

    # ==================== 检索 ====================

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 8,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        向量相似度检索（余弦相似度）。

        Args:
            query_embedding: 查询向量
            top_k:           返回数量
            filter_dict:     元数据过滤条件
                             支持 file_name / source_type / knowledge_base / doc_id

        Returns:
            List[Document]: 相似文档列表，metadata 中包含 similarity 分数
        """
        table = self.config.pg_table_name
        # 构造 WHERE 子句
        where_clauses = []
        params: List[Any] = []
        if filter_dict:
            for key in ("file_name", "source_type", "knowledge_base", "doc_id"):
                if key in filter_dict:
                    where_clauses.append(f"{key} = %s")
                    params.append(filter_dict[key])

        where_sql = (" WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

        # <=> 余弦距离算子；ORDER BY ... ASC 距离越小越相似
        sql = f"""
            SELECT chunk_id, content, file_name, page, doc_id,
                   source_type, knowledge_base, meta_json,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM {table}{where_sql}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """
        # 参数顺序：query（算 similarity）, query（ORDER BY）, where 参数, top_k
        sql_params = [query_embedding] + params + [query_embedding, top_k]

        try:
            with self._conn.cursor() as cur:
                cur.execute(sql, sql_params)
                rows = cur.fetchall()
        except Exception as e:
            raise VectorStoreError(f"PGvector 检索失败: {str(e)}") from e

        return self._rows_to_documents(rows)

    def search_with_filter(
        self,
        query_embedding: List[float],
        metadata_filter: Dict[str, Any],
        top_k: int = 8,
    ) -> List[Document]:
        """
        带元数据过滤的向量检索（用于知识库隔离 / 多源筛选）。
        """
        return self.search(query_embedding, top_k, metadata_filter)

    # ==================== MMR 候选拉取（供 Layer 3 使用） ====================

    def search_candidates(
        self,
        query_embedding: List[float],
        fetch_k: int = 50,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        拉取 fetch_k 个候选（供 MMR 多样性重排使用）。

        Args:
            query_embedding: 查询向量
            fetch_k:         候选数量
            filter_dict:     元数据过滤条件

        Returns:
            List[Document]: 候选文档列表（含相似度分数）
        """
        return self.search(query_embedding, fetch_k, filter_dict)

    # ==================== 全量获取 ====================

    def get_all_documents(self) -> List[Document]:
        """
        获取知识库中全部文档（供 BM25 索引构建和元数据浏览）。
        """
        table = self.config.pg_table_name
        sql = f"""
            SELECT chunk_id, content, file_name, page, doc_id,
                   source_type, knowledge_base, meta_json
            FROM {table}
        """
        try:
            with self._conn.cursor() as cur:
                cur.execute(sql)
                rows = cur.fetchall()
        except Exception as e:
            raise VectorStoreError(f"获取全部文档失败: {str(e)}") from e

        return self._rows_to_documents(rows, with_similarity=False)

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        获取知识库统计信息。
        """
        table = self.config.pg_table_name
        try:
            with self._conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM {table}")
                total = cur.fetchone()[0]
                cur.execute(
                    f"SELECT file_name, COUNT(*) FROM {table} GROUP BY file_name"
                )
                file_rows = cur.fetchall()
        except Exception as e:
            raise VectorStoreError(f"获取统计信息失败: {str(e)}") from e

        file_names = [r[0] for r in file_rows]
        return {
            "store_type": "pgvector",
            "collection_name": f"{self.config.pg_database}.{table}",
            "total_chunks": total,
            "total_files": len(file_names),
            "file_names": sorted(file_names),
        }

    # ==================== 私有方法 ====================

    def _ensure_schema(self):
        """确保表和索引存在。"""
        table = self.config.pg_table_name
        dim = self.config.pg_vector_size

        # 建表（chunk_id 主键，embedding vector(1024)）
        create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS {table} (
                chunk_id        TEXT PRIMARY KEY,
                content         TEXT NOT NULL,
                embedding       vector({dim}) NOT NULL,
                file_name       TEXT NOT NULL DEFAULT 'unknown',
                page            INT NOT NULL DEFAULT 0,
                doc_id          TEXT NOT NULL DEFAULT '',
                source_type     TEXT NOT NULL DEFAULT '论文原文',
                knowledge_base  TEXT NOT NULL DEFAULT 'default',
                meta_json       JSONB,
                created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """
        # 索引：加速元数据过滤
        idx_file = f"CREATE INDEX IF NOT EXISTS idx_{table}_file_name ON {table}(file_name)"
        idx_source = f"CREATE INDEX IF NOT EXISTS idx_{table}_source_type ON {table}(source_type)"
        idx_kb = f"CREATE INDEX IF NOT EXISTS idx_{table}_knowledge_base ON {table}(knowledge_base)"
        # HNSW 向量索引（余弦距离）
        idx_vec = (
            f"CREATE INDEX IF NOT EXISTS idx_{table}_embedding_hnsw "
            f"ON {table} USING hnsw (embedding vector_cosine_ops)"
        )

        with self._conn.cursor() as cur:
            cur.execute(create_table_sql)
            cur.execute(idx_file)
            cur.execute(idx_source)
            cur.execute(idx_kb)
            cur.execute(idx_vec)

    def _rows_to_documents(
        self, rows: List[tuple], with_similarity: bool = True
    ) -> List[Document]:
        """
        将 SQL 查询结果转换为 LangChain Document 列表。

        行结构：(chunk_id, content, file_name, page, doc_id,
                source_type, knowledge_base, meta_json[, similarity])
        """
        documents = []
        for row in rows:
            chunk_id = row[0]
            content = row[1] or ""
            # 解析 meta_json 还原完整元数据
            meta = {}
            if row[7]:
                try:
                    meta = json.loads(row[7]) if isinstance(row[7], str) else dict(row[7])
                except Exception:
                    meta = {}

            # 用结构化字段覆盖（保证类型一致）
            meta["chunk_id"] = chunk_id
            meta["file_name"] = row[2]
            meta["page"] = row[3]
            meta["doc_id"] = row[4]
            meta["source_type"] = row[5]
            meta["knowledge_base"] = row[6]

            if with_similarity and len(row) >= 9:
                # similarity = 1 - cosine_distance，已 SQL 计算
                sim = float(row[8]) if row[8] is not None else 0.0
                meta["similarity"] = round(sim, 6)

            documents.append(Document(page_content=content, metadata=meta))

        return documents
