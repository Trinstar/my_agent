import yaml
from typing import List, Dict, Any
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from my_agent.utils.logs import get_logger
from my_agent.utils.rag.data_preparation import DataPreparationModule
from my_agent.utils.rag.index_instruction import IndexConstructionModule
logger = get_logger()


class RAGModule:
    def __init__(self, rag_config: dict):
        """
        初始化RAG模块

        Args:
            config_path: 配置文件路径
        """

        self.config: dict = rag_config
        self.topk: int = self.config.get('top_k', 3)

        self._data_prep = DataPreparationModule(
            data_path=self.config['file_data_path'])
        self._index_module = IndexConstructionModule(
            index_save_path=self.config['file_index_path'],
            model_name=self.config['embedding_model']
        )

        self.documents = None
        self.chunks = None
        self.vectorstore = None
        self.vector_retriever = None
        self.bm25_retriever = None

    def prepare_data(self):
        self.documents = self._data_prep.load_documents()  # List[Document]
        self.chunks = self._data_prep.chunk_documents()   # List[Document]

    def load_index(self):
        self.vectorstore = self._index_module.load_index()  # FAISS Object

    def build_and_save_index(self):
        if self.chunks is None:
            raise ValueError("请先准备数据后再构建索引")
        self.vectorstore = self._index_module.build_vector_index(self.chunks)
        self._index_module.save_index()

    def index_similarity_search(self, query: str) -> List[Document]:
        res_chunks = self.vectorstore.similarity_search(query=query, k=self.topk)
        res_docs = self._data_prep.get_parent_documents(res_chunks, self.documents)
        return res_docs

    def _setup_retrievers(self):
        self.vector_retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.topk*2}
        )
        self.bm25_retriever = BM25Retriever.from_documents(
            self.chunks,
            k=self.topk*2
        )

    def hybrid_search(self, query: str) -> List[Document]:
        """
        混合检索 - 结合向量检索和BM25检索，使用RRF重排

        Args:
            query: 查询文本
            top_k: 返回结果数量

        Returns:
            检索到的文档列表
        """
        # 分别获取向量检索和BM25检索结果
        if self.vector_retriever is None or self.bm25_retriever is None:
            self._setup_retrievers()
        vector_docs = self.vector_retriever.invoke(query)
        bm25_docs = self.bm25_retriever.invoke(query)

        # 使用RRF重排
        reranked_docs = self._rrf_rerank(vector_docs, bm25_docs)
        reranked_docs = reranked_docs[:self.topk*2]
        res_docs = self._data_prep.get_parent_documents(reranked_docs, self.documents)
        if len(res_docs) > self.topk:
            res_docs = res_docs[:self.topk]
        return res_docs

    def _rrf_rerank(self, vector_docs: List[Document], bm25_docs: List[Document], c: int = 60) -> List[Document]:
        """
        使用RRF (Reciprocal Rank Fusion) 算法重排文档

        Args:
            vector_docs: 向量检索结果
            bm25_docs: BM25检索结果
            c: RRF参数，用于平滑排名

        Returns:
            重排后的文档列表
        """
        doc_scores = {}
        doc_objects = {}

        # 计算向量检索结果的RRF分数
        for rank, doc in enumerate(vector_docs):
            # 使用文档内容的哈希作为唯一标识
            doc_id = hash(doc.page_content)
            doc_objects[doc_id] = doc

            # RRF公式: 1 / (k + rank)
            rrf_score = 1.0 / (c + rank + 1)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score

            logger.debug(f"向量检索 - 文档{rank+1}: RRF分数 = {rrf_score:.4f}")

        # 计算BM25检索结果的RRF分数
        for rank, doc in enumerate(bm25_docs):
            doc_id = hash(doc.page_content)
            doc_objects[doc_id] = doc

            rrf_score = 1.0 / (c + rank + 1)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score

            logger.debug(f"BM25检索 - 文档{rank+1}: RRF分数 = {rrf_score:.4f}")

        # 按最终RRF分数排序
        sorted_docs = sorted(doc_scores.items(),
                             key=lambda x: x[1], reverse=True)

        # 构建最终结果
        reranked_docs = []
        for doc_id, final_score in sorted_docs:
            if doc_id in doc_objects:
                doc = doc_objects[doc_id]
                # 将RRF分数添加到文档元数据中
                doc.metadata['rrf_score'] = final_score
                reranked_docs.append(doc)
                logger.debug(
                    f"最终排序 - 文档: {doc.page_content[:50]}... 最终RRF分数: {final_score:.4f}")

        logger.info(
            f"RRF重排完成: 向量检索{len(vector_docs)}个文档, BM25检索{len(bm25_docs)}个文档, 合并后{len(reranked_docs)}个文档")

        return reranked_docs
