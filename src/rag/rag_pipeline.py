"""
RAG (Retrieval-Augmented Generation) 检索增强生成系统
大模型应用开发
支持：文档加载、文本分块、向量化、语义检索、LLM生成回答
"""
import os
import numpy as np
from typing import List, Dict, Tuple


class DocumentLoader:
    """文档加载器：支持多种格式"""

    @staticmethod
    def load_txt(file_path: str) -> str:
        """加载文本文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    @staticmethod
    def load_directory(dir_path: str, extensions: List[str] = None) -> List[Dict]:
        """加载目录下所有文档"""
        if extensions is None:
            extensions = ['.txt', '.md']
        documents = []
        for root, _, files in os.walk(dir_path):
            for file in files:
                if any(file.endswith(ext) for ext in extensions):
                    filepath = os.path.join(root, file)
                    content = DocumentLoader.load_txt(filepath)
                    documents.append({
                        'path': filepath,
                        'name': file,
                        'content': content
                    })
        return documents


class TextSplitter:
    """文本分块器：递归字符分割 + 重叠"""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50,
                 separators: List[str] = None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ['\n\n', '\n', '. ', ' ', '']

    def split_text(self, text: str) -> List[str]:
        """递归分割文本"""
        chunks = self._recursive_split(text, self.separators)
        return chunks

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        """递归分割"""
        final_chunks = []
        separator = separators[-1]
        for s in separators:
            if s in text:
                separator = s
                break

        splits = text.split(separator) if separator else list(text)
        merged_splits = []
        current = ''

        for split in splits:
            if len(current) + len(split) + len(separator) <= self.chunk_size:
                current += split + separator
            else:
                if current:
                    merged_splits.append(current.strip())
                if len(split) > self.chunk_size:
                    if separators.index(separator) < len(separators) - 1:
                        sub_chunks = self._recursive_split(split, separators[separators.index(separator) + 1:])
                        merged_splits.extend(sub_chunks)
                    else:
                        merged_splits.append(split[:self.chunk_size])
                else:
                    current = split + separator

        if current:
            merged_splits.append(current.strip())

        # 合并小块 + 重叠
        for chunk in merged_splits:
            if not final_chunks or len(final_chunks[-1]) + len(chunk) > self.chunk_size:
                final_chunks.append(chunk)
            else:
                final_chunks[-1] += ' ' + chunk

        return final_chunks


class SimpleEmbedding:
    """
    简易向量化器
    实际项目中可替换为 OpenAI Embedding / BGE / Sentence-BERT
    这里使用 TF-IDF + 降维作为演示
    """

    def __init__(self, method='tfidf', max_features=1000):
        self.method = method
        self.max_features = max_features
        self.vocab = None
        self.idf = None

    def fit(self, texts: List[str]):
        """构建词汇表和IDF"""
        # 简单分词（按空格和标点）
        tokenized = [self._tokenize(text) for text in texts]

        # 构建词汇表
        word_freq = {}
        for tokens in tokenized:
            for token in set(tokens):
                word_freq[token] = word_freq.get(token, 0) + 1

        # 取高频词
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        self.vocab = {word: idx for idx, (word, _) in enumerate(sorted_words[:self.max_features])}

        # 计算IDF
        n_docs = len(texts)
        self.idf = np.zeros(len(self.vocab))
        for word, idx in self.vocab.items():
            df = word_freq[word]
            self.idf[idx] = np.log((n_docs + 1) / (df + 1)) + 1

        return self

    def encode(self, texts: List[str]) -> np.ndarray:
        """编码文本为向量"""
        if isinstance(texts, str):
            texts = [texts]

        vectors = np.zeros((len(texts), len(self.vocab)))
        for i, text in enumerate(texts):
            tokens = self._tokenize(text)
            tf = {}
            for token in tokens:
                if token in self.vocab:
                    tf[token] = tf.get(token, 0) + 1
            for token, count in tf.items():
                idx = self.vocab[token]
                vectors[i, idx] = count * self.idf[idx]

            # L2归一化
            norm = np.linalg.norm(vectors[i])
            if norm > 0:
                vectors[i] /= norm

        return vectors

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """简单分词"""
        text = text.lower()
        for char in '.,!?;:()[]{}"\'':
            text = text.replace(char, ' ')
        return text.split()


class VectorStore:
    """向量存储与检索"""

    def __init__(self, embedding_model: SimpleEmbedding):
        self.embedding_model = embedding_model
        self.documents = []
        self.vectors = None

    def add_documents(self, documents: List[Dict]):
        """添加文档到向量库"""
        self.documents.extend(documents)
        texts = [doc['content'] for doc in documents]
        new_vectors = self.embedding_model.encode(texts)
        if self.vectors is None:
            self.vectors = new_vectors
        else:
            self.vectors = np.vstack([self.vectors, new_vectors])

    def similarity_search(self, query: str, k: int = 5) -> List[Tuple[Dict, float]]:
        """语义检索：返回最相似的k个文档"""
        query_vector = self.embedding_model.encode([query])[0]
        # 余弦相似度（向量已归一化，直接点积）
        similarities = self.vectors @ query_vector
        top_k_indices = np.argsort(similarities)[::-1][:k]

        results = []
        for idx in top_k_indices:
            results.append((self.documents[idx], float(similarities[idx])))
        return results


class SimpleLLM:
    """
    简易LLM接口
    实际项目中可替换为 OpenAI API / 本地大模型 / 豆包API
    这里使用模板生成作为演示
    """

    def __init__(self, api_key: str = None, model: str = 'gpt-3.5-turbo'):
        self.api_key = api_key or os.environ.get('OPENAI_API_KEY', '')
        self.model = model

    def generate(self, prompt: str, context: str = '') -> str:
        """生成回答"""
        if self.api_key:
            # 实际调用OpenAI API
            try:
                import openai
                openai.api_key = self.api_key
                full_prompt = f"基于以下上下文回答问题。如果上下文中没有相关信息，请说'根据现有资料无法回答'。\n\n上下文：\n{context}\n\n问题：{prompt}\n\n回答："
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[{"role": "user", "content": full_prompt}],
                    temperature=0.3,
                    max_tokens=500
                )
                return response.choices[0].message.content
            except ImportError:
                pass

        # 演示模式：基于上下文的模板回答
        if context:
            return f"根据检索到的资料：\n{context[:500]}...\n\n（注：这是演示模式，实际使用请配置LLM API）"
        return "未检索到相关信息，无法回答该问题。"


class RAGPipeline:
    """RAG 完整流水线"""

    def __init__(self, chunk_size=500, chunk_overlap=50, top_k=5):
        self.text_splitter = TextSplitter(chunk_size, chunk_overlap)
        self.embedding_model = SimpleEmbedding()
        self.vector_store = VectorStore(self.embedding_model)
        self.llm = SimpleLLM()
        self.top_k = top_k
        self._is_fitted = False

    def ingest(self, documents: List[Dict]):
        """摄入文档：分块 -> 向量化 -> 存储"""
        print(f"Ingesting {len(documents)} documents...")

        # 文本分块
        all_chunks = []
        for doc in documents:
            chunks = self.text_splitter.split_text(doc['content'])
            for i, chunk in enumerate(chunks):
                all_chunks.append({
                    'content': chunk,
                    'source': doc.get('name', 'unknown'),
                    'chunk_id': i
                })

        print(f"  Generated {len(all_chunks)} chunks")

        # 训练向量化器
        texts = [chunk['content'] for chunk in all_chunks]
        self.embedding_model.fit(texts)
        self._is_fitted = True

        # 存储到向量库
        self.vector_store.add_documents(all_chunks)
        print(f"  Vector store size: {len(self.vector_store.documents)}")

    def query(self, question: str) -> Dict:
        """查询：检索 -> 生成"""
        if not self._is_fitted:
            return {'answer': '请先摄入文档', 'sources': []}

        # 检索
        results = self.vector_store.similarity_search(question, k=self.top_k)

        # 构建上下文
        context_parts = []
        sources = []
        for doc, score in results:
            context_parts.append(f"[来源: {doc['source']}, 相关度: {score:.3f}]\n{doc['content']}")
            sources.append({'source': doc['source'], 'score': score, 'content': doc['content'][:200]})
        context = '\n\n'.join(context_parts)

        # 生成回答
        answer = self.llm.generate(question, context)

        return {
            'question': question,
            'answer': answer,
            'sources': sources,
            'context': context
        }


# 使用示例
if __name__ == '__main__':
    # 1. 创建RAG流水线
    rag = RAGPipeline(chunk_size=300, chunk_overlap=50, top_k=3)

    # 2. 摄入文档
    docs = [
        {'name': 'product_manual.txt',
         'content': '本产品支持图像分类功能，支持ResNet和ViT模型。'
                    '训练时使用混合精度可以加速50%。模型支持导出为ONNX格式部署。'},
        {'name': 'api_docs.txt',
         'content': 'API服务运行在8000端口，支持/predict和/gradcam接口。'
                    '请求需要上传图片文件，返回JSON格式的预测结果。'},
        {'name': 'deployment_guide.txt',
         'content': '使用Docker部署镜像，docker-compose up启动服务。'
                    '支持GPU加速，需要安装nvidia-docker。健康检查在/health路径。'},
    ]
    rag.ingest(docs)

    # 3. 查询
    result = rag.query("如何部署这个模型？")
    print(f"\n问题: {result['question']}")
    print(f"回答: {result['answer']}")
    print("\n参考来源:")
    for src in result['sources']:
        print(f"  - {src['source']} (相关度: {src['score']:.3f})")
