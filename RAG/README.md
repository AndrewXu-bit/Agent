# RAG 知识库系统

基于 LlamaIndex 构建的完整 RAG（检索增强生成）知识库系统，支持本地文档索引和智能问答。

## 📋 功能特性

- **文档加载**: 使用 Unstructured 支持多种格式（PDF、Word、PPT、Excel、TXT、Markdown等）
- **文本分割**: RecursiveCharacterTextSplitter 智能分块，保持语义完整性
- **嵌入模型**: BGE-M3 多语言嵌入模型，强大的语义表示能力
- **向量数据库**: Chroma 轻量级向量数据库，支持持久化存储
- **重排序**: BGE-Reranker-Large 精确评估查询与文档相关性
- **混合检索**: EnsembleRetriever 结合向量检索和BM25关键词检索
- **LlamaIndex集成**: 完整的检索增强生成流程

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备文档

将您的文档放入 `data/documents` 目录（或自定义目录）：

```bash
mkdir -p data/documents
# 复制您的 PDF、Word、TXT 等文档到此目录
```

### 3. 运行示例

```bash
python RAG/example.py
```

## 💻 代码示例

### 基本使用

```python
from RAG.engine import RAGEngine

# 1. 初始化 RAG 引擎
engine = RAGEngine(
    chunk_size=512,           # 文本块大小
    chunk_overlap=50,         # 文本块重叠
    chroma_collection="my_knowledge",
    chroma_persist_dir="./chroma_db",
    use_reranker=True,        # 启用重排序
    use_hybrid_retrieval=True, # 启用混合检索
)

# 2. 构建索引（从文档目录）
doc_count = engine.build_index("./data/documents", recursive=True)
print(f"成功索引 {doc_count} 个文档")

# 3. 查询知识库
result = engine.query("什么是机器学习？", top_k=3)

print(f"问题: {result['question']}")
print(f"答案: {result['answer']}")
print(f"参考来源: {len(result['contexts'])} 条")

for ctx in result['contexts']:
    print(f"  [{ctx['index']}] 相关性: {ctx['score']:.3f}")
    print(f"  内容: {ctx['text'][:100]}...")
```

### 仅查询模式

```python
# 加载已有的知识库（不重建索引）
engine = RAGEngine(
    chroma_collection="my_knowledge",
    chroma_persist_dir="./chroma_db",
)

# 直接查询
result = engine.query("深度学习有哪些应用？")
```

## 📊 架构说明

```
┌─────────────┐
│  文档加载    │ ← Unstructured (PDF/Word/TXT/etc)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  文本分割    │ ← RecursiveCharacterTextSplitter
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  嵌入模型    │ ← BGE-M3
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  向量数据库  │ ← Chroma
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  混合检索    │ ← Vector + BM25 (EnsembleRetriever)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  重排序      │ ← BGE-Reranker-Large
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  LLM 生成    │ ← 基于检索结果生成答案
└─────────────┘
```

## ⚙️ 配置参数

### RAGEngine 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `chunk_size` | int | 512 | 文本块大小（字符数） |
| `chunk_overlap` | int | 50 | 文本块重叠大小 |
| `embedding_model_name` | str | "BAAI/bge-m3" | 嵌入模型名称 |
| `chroma_collection` | str | "rag_knowledge" | Chroma集合名称 |
| `chroma_persist_dir` | str | "./chroma_db" | 向量数据库持久化目录 |
| `reranker_model_name` | str | "BAAI/bge-reranker-large" | 重排序模型名称 |
| `similarity_top_k` | int | 10 | 向量检索返回数量 |
| `bm25_top_k` | int | 10 | BM25检索返回数量 |
| `rerank_top_k` | int | 5 | 重排序后返回数量 |
| `use_reranker` | bool | True | 是否使用重排序 |
| `use_hybrid_retrieval` | bool | True | 是否使用混合检索 |

## 📁 项目结构

```
RAG/
├── __init__.py          # 模块初始化
├── document_loader.py   # 文档加载器 (Unstructured)
├── text_splitter.py     # 文本分割器 (RecursiveCharacterTextSplitter)
├── embedding.py         # 嵌入模型 (BGE-M3)
├── vector_store.py      # 向量数据库 (Chroma)
├── reranker.py          # 重排序器 (BGE-Reranker-Large)
├── retriever.py         # 混合检索器 (EnsembleRetriever)
├── engine.py            # RAG 引擎（整合所有组件）
├── example.py           # 使用示例
└── README.md            # 本文档
```

## 🔧 高级用法

### 自定义文本块大小

```python
engine = RAGEngine(
    chunk_size=1024,      # 更大的文本块
    chunk_overlap=100,    # 更大的重叠
)
```

### 禁用重排序（更快但可能精度略低）

```python
engine = RAGEngine(
    use_reranker=False,
)
```

### 仅使用向量检索（不使用混合检索）

```python
engine = RAGEngine(
    use_hybrid_retrieval=False,
)
```

### 获取统计信息

```python
stats = engine.get_stats()
print(f"文档数量: {stats['document_count']}")
print(f"配置: {stats}")
```

### 清空索引

```python
engine.clear_index()
```

## 📝 注意事项

1. **首次运行会下载模型**：BGE-M3 和 BGE-Reranker-Large 模型较大（约2GB），首次运行需要时间下载
2. **GPU加速**：如果有CUDA，会自动使用GPU加速；否则使用CPU
3. **内存占用**：处理大量文档时注意内存使用，可以调整 `chunk_size`
4. **持久化存储**：Chroma数据库会自动保存到磁盘，下次启动可直接查询

## 🐛 常见问题

### Q: 如何处理中文文档？
A: BGE-M3 原生支持多语言，包括中文，无需特殊配置。

### Q: 如何添加更多文档到现有索引？
A: 再次调用 `engine.build_index(new_docs_dir)` 即可，新文档会追加到现有索引。

### Q: 如何提高检索精度？
A: 
- 启用重排序：`use_reranker=True`
- 启用混合检索：`use_hybrid_retrieval=True`
- 增加 `rerank_top_k` 值
- 调整 `chunk_size` 以适应您的文档

### Q: 如何降低内存使用？
A:
- 减小 `chunk_size`
- 减少 `similarity_top_k` 和 `bm25_top_k`
- 使用 CPU 而非 GPU（设置 `device="cpu"`）

## 📄 许可证

MIT License

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！
