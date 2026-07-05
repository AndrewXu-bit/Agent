# 📚 RAG 知识库使用指南

## 🚀 快速开始

### 第一步：准备文档

1. **创建文档目录**
```powershell
mkdir data\documents
```

2. **放入您的文档**
将需要索引的文档复制到 `data/documents` 目录，支持格式：
- PDF (.pdf)
- Word (.docx, .doc)
- PowerPoint (.pptx, .ppt)
- Excel (.xlsx, .xls)
- 文本文件 (.txt, .md)
- HTML (.html, .htm)

### 第二步：构建索引（文档入库）

#### 方式 1：使用默认配置（最简单）

```powershell
python RAG\build_index.py
```

这会：
- 从 `./data/documents` 读取所有文档
- 使用默认配置构建索引
- 保存到 `./chroma_db` 目录

#### 方式 2：自定义配置

```powershell
# 指定文档目录
python RAG\build_index.py D:\MyDocs

# 自定义文本块大小和集合名称
python RAG\build_index.py --chunk-size 1024 --collection my_docs

# 禁用重排序（更快但精度略低）
python RAG\build_index.py --no-reranker

# 查看所有选项
python RAG\build_index.py --help
```

**参数说明：**
- `docs_dir`: 文档目录路径（默认: `./data/documents`）
- `--collection`: Chroma 集合名称（默认: `rag_knowledge`）
- `--persist-dir`: 数据库存储位置（默认: `./chroma_db`）
- `--chunk-size`: 文本块大小（默认: 512，越大上下文越多）
- `--chunk-overlap`: 文本块重叠（默认: 50）
- `--no-reranker`: 禁用重排序（提高速度）
- `--no-hybrid`: 禁用混合检索（仅向量检索）

### 第三步：查询知识库

#### 方式 1：单次查询

```powershell
# 直接提问
python RAG\query.py "什么是机器学习？"

# 指定返回结果数量
python RAG\query.py "深度学习的应用" --top-k 5

# 指定知识库
python RAG\query.py "AI" --collection my_docs
```

#### 方式 2：交互模式（推荐）

```powershell
# 进入交互问答
python RAG\query.py --interactive

# 或者简写
python RAG\query.py -i
```

在交互模式中：
- 输入问题后按回车得到答案
- 输入 `quit` 或 `exit` 退出
- 可以连续提问

## 📋 完整工作流程示例

### 示例 1：基本使用

```powershell
# 1. 准备文档
mkdir data\documents
# 复制一些 PDF/TXT 文档到 data\documents

# 2. 构建索引
python RAG\build_index.py

# 3. 查询
python RAG\query.py "文档中提到了什么？"
```

### 示例 2：多个知识库

```powershell
# 为不同主题创建独立的知识库

# 技术文档库
python RAG\build_index.py docs\tech --collection tech_docs

# 产品文档库
python RAG\build_index.py docs\product --collection product_docs

# 分别查询
python RAG\query.py "API 如何使用？" --collection tech_docs
python RAG\query.py "产品功能" --collection product_docs
```

### 示例 3：性能优化

```powershell
# 快速模式（牺牲一点精度换取速度）
python RAG\build_index.py --no-reranker --no-hybrid

# 高精度模式（更准确但较慢）
python RAG\build_index.py --chunk-size 1024 --chunk-overlap 100

# 查询时也禁用重排序
python RAG\query.py "问题" --no-reranker
```

## 🔧 高级用法

### 在代码中使用

```python
from RAG.engine import RAGEngine

# 1. 构建索引
engine = RAGEngine(
    chunk_size=512,
    chroma_collection="my_knowledge",
    chroma_persist_dir="./chroma_db",
)

# 从目录构建
doc_count = engine.build_index("./data/documents")
print(f"索引了 {doc_count} 个文档")

# 2. 查询
result = engine.query("您的问题", top_k=3)

# 查看结果
print(result['answer'])
for ctx in result['contexts']:
    print(f"相关性: {ctx['score']}")
    print(f"内容: {ctx['text']}")
```

### 与飞书机器人集成

在 `feishu/bot.py` 中添加 RAG 支持：

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
        """增强版：结合 RAG 和 Agent"""
        
        # 1. 先从知识库检索
        rag_result = self.rag_engine.query(user_message, top_k=3)
        
        if rag_result['has_answer']:
            # 2. 有相关知识，结合到 prompt
            contexts = "\n".join([
                ctx['text'] for ctx in rag_result['contexts']
            ])
            
            enhanced_prompt = f"""
基于以下知识库信息回答问题：

{contexts}

用户问题：{user_message}

请根据知识库内容回答。
"""
            result = await self.agent.run(enhanced_prompt)
        else:
            # 3. 没有相关知识，直接回答
            result = await self.agent.run(user_message)
        
        return result
```

## 📊 管理知识库

### 查看统计信息

```python
from RAG.engine import RAGEngine

engine = RAGEngine(
    chroma_collection="rag_knowledge",
    chroma_persist_dir="./chroma_db",
)

stats = engine.get_stats()
print(f"文档数量: {stats['document_count']}")
print(f"配置: {stats}")
```

### 添加新文档

```python
# 直接再次调用 build_index，新文档会追加
engine.build_index("./new_documents")
```

### 清空知识库

```python
engine.clear_index()
```

### 删除旧知识库

```powershell
# 删除整个数据库目录
Remove-Item -Recurse -Force .\chroma_db
```

## ⚙️ 调优建议

### 提高检索精度

```powershell
# 1. 增加候选数量
python RAG\build_index.py --chunk-size 1024 --chunk-overlap 100

# 2. 查询时返回更多结果
python RAG\query.py "问题" --top-k 10

# 3. 保持重排序启用（默认）
```

### 提高响应速度

```powershell
# 1. 禁用重排序
python RAG\build_index.py --no-reranker
python RAG\query.py "问题" --no-reranker

# 2. 减小文本块
python RAG\build_index.py --chunk-size 256

# 3. 减少返回数量
python RAG\query.py "问题" --top-k 2
```

### 处理大型文档

```powershell
# 使用更大的文本块
python RAG\build_index.py --chunk-size 2048 --chunk-overlap 200
```

### 处理专业术语多的文档

```powershell
# 使用较小的文本块保持精确性
python RAG\build_index.py --chunk-size 256 --chunk-overlap 30
```

## 🐛 常见问题

### Q1: 如何知道索引是否成功？
A: 运行后会显示处理的文档数量。也可以查询测试：
```powershell
python RAG\query.py --interactive
```

### Q2: 可以修改已索引的文档吗？
A: 需要先清空再重建：
```python
engine.clear_index()
engine.build_index("./updated_documents")
```

### Q3: 索引占用多少空间？
A: 取决于文档数量和大小。一般文本文档的向量索引约为原文的 10-20%。

### Q4: 首次运行很慢？
A: 正常！首次会下载 BGE-M3 模型（约 2GB）。后续会快很多。

### Q5: 如何备份知识库？
A: 直接复制 `chroma_db` 目录即可：
```powershell
Copy-Item -Recurse .\chroma_db .\chroma_db_backup
```

## 📝 最佳实践

1. **文档组织**：按主题分类存放文档，为每个主题创建独立的知识库
2. **文本块大小**：一般文档用 512，技术文档用 1024，法律文档用 256
3. **重排序**：对精度要求高时启用，对速度要求高时禁用
4. **定期更新**：文档更新后重新构建索引
5. **备份重要数据**：定期备份 `chroma_db` 目录

## 🎯 下一步

- 查看 [README.md](README.md) 了解详细技术架构
- 查看 [QUICKSTART.md](QUICKSTART.md) 快速上手
- 运行 `python RAG\example.py` 查看完整示例

祝您使用愉快！🎉
