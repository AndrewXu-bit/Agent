# RAG 知识库快速开始指南

## 📦 第一步：安装依赖

```bash
cd D:\Code\Agent
pip install -r requirements.txt
```

> **注意**：首次安装会下载较大的模型文件（约2GB），请耐心等待。

## 📚 第二步：准备文档

创建文档目录并放入您的文档：

```bash
mkdir data\documents
```

支持的文档格式：
- PDF (.pdf)
- Word (.docx, .doc)
- PowerPoint (.pptx, .ppt)
- Excel (.xlsx, .xls)
- 文本文件 (.txt, .md)
- HTML (.html, .htm)

## 🧪 第三步：测试模块

```bash
python RAG\test_rag.py
```

如果看到 "🎉 所有测试通过"，说明安装成功！

## 🚀 第四步：运行示例

### 方式1：使用示例脚本

```bash
python RAG\example.py
```

这会：
1. 初始化 RAG 引擎
2. 从 `data/documents` 目录加载文档并构建索引
3. 演示几个预设问题的查询
4. 询问是否进入交互问答模式

### 方式2：编写自己的代码

创建 `my_rag_app.py`:

```python
from RAG.engine import RAGEngine

# 1. 初始化
print("初始化 RAG 引擎...")
engine = RAGEngine(
    chunk_size=512,
    chunk_overlap=50,
    chroma_collection="my_docs",
    chroma_persist_dir="./chroma_db",
)

# 2. 构建索引
print("构建索引...")
doc_count = engine.build_index("./data/documents")
print(f"✅ 索引了 {doc_count} 个文档")

# 3. 查询
question = "什么是人工智能？"
print(f"\n问题: {question}")
result = engine.query(question, top_k=3)
print(f"答案:\n{result['answer']}")
```

运行：
```bash
python my_rag_app.py
```

## 💡 第五步：集成到您的应用

### 与飞书机器人集成

在 `feishu/bot.py` 中集成 RAG：

```python
from RAG.engine import RAGEngine

class FeishuBot:
    def __init__(self, ...):
        # ... 现有代码 ...
        
        # 初始化 RAG 引擎
        self.rag_engine = RAGEngine(
            chroma_collection="feishu_knowledge",
            chroma_persist_dir="./chroma_db",
        )
    
    async def _process_with_agent(self, user_message: str):
        """增强版：先检索知识库，再调用 Agent"""
        
        # 1. 先从知识库检索相关信息
        rag_result = self.rag_engine.query(user_message, top_k=3)
        
        # 2. 如果有相关知识，结合到 prompt 中
        if rag_result['has_answer']:
            context = "\n".join([ctx['text'] for ctx in rag_result['contexts']])
            enhanced_prompt = f"""
基于以下知识库信息回答问题：

知识库内容：
{context}

用户问题：{user_message}

请根据知识库内容回答，如果知识库中没有相关信息，请说明。
"""
            result = await self.agent.run(enhanced_prompt)
        else:
            # 没有相关知识，直接使用原问题
            result = await self.agent.run(user_message)
        
        return result
```

## 🔧 常见问题

### Q1: 第一次运行很慢？
A: 正常！首次运行会下载 BGE-M3 和 BGE-Reranker-Large 模型（约2GB）。后续运行会快很多。

### Q2: 如何查看已索引的文档数量？
```python
stats = engine.get_stats()
print(f"文档数量: {stats['document_count']}")
```

### Q3: 如何添加新文档？
```python
# 直接再次调用 build_index 即可
engine.build_index("./new_documents")
```

### Q4: 如何清空索引？
```python
engine.clear_index()
```

### Q5: CPU 还是 GPU？
A: 系统会自动检测。如果有 CUDA，会使用 GPU；否则使用 CPU。

## 📊 性能优化建议

1. **提高检索速度**：
   ```python
   engine = RAGEngine(
       use_reranker=False,          # 禁用重排序（更快）
       use_hybrid_retrieval=False,  # 仅使用向量检索
   )
   ```

2. **提高检索精度**：
   ```python
   engine = RAGEngine(
       similarity_top_k=20,         # 增加候选数量
       bm25_top_k=20,
       rerank_top_k=10,             # 重排序后返回更多
       use_reranker=True,           # 启用重排序
       use_hybrid_retrieval=True,   # 启用混合检索
   )
   ```

3. **调整文本块大小**：
   - 短文档/精确查询：`chunk_size=256`
   - 长文档/概括性问题：`chunk_size=1024`

## 🎯 下一步

- 阅读 [RAG/README.md](README.md) 了解详细配置
- 查看 [RAG/example.py](example.py) 学习完整示例
- 根据您的需求调整参数

祝您使用愉快！🎉
