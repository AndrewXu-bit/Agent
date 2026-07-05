"""
RAG 知识库使用示例

演示如何使用 RAG 引擎构建知识库并进行问答。
"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from RAG.engine import RAGEngine


def example_basic_usage():
    """基本使用示例"""
    print("=" * 60)
    print("RAG 知识库 - 基本使用示例")
    print("=" * 60)
    
    # 1. 初始化 RAG 引擎
    print("\n📦 初始化 RAG 引擎...")
    engine = RAGEngine(
        chunk_size=512,
        chunk_overlap=50,
        chroma_collection="demo_knowledge",
        chroma_persist_dir="./data/chroma_db",
        similarity_top_k=10,
        bm25_top_k=10,
        rerank_top_k=5,
        use_reranker=True,
        use_hybrid_retrieval=True,
    )
    
    # 2. 构建索引（从本地文档目录）
    docs_dir = "./data/documents"  # 替换为您的文档目录
    
    if Path(docs_dir).exists():
        print(f"\n📚 构建知识库索引: {docs_dir}")
        doc_count = engine.build_index(docs_dir, recursive=True)
        print(f"✅ 成功索引 {doc_count} 个文档")
    else:
        print(f"\n⚠️  文档目录不存在: {docs_dir}")
        print("请先创建文档目录并放入一些测试文档")
        return
    
    # 3. 查询知识库
    questions = [
        "什么是机器学习？",
        "如何训练神经网络？",
        "深度学习有哪些应用？",
    ]
    
    print("\n" + "=" * 60)
    print("开始问答测试")
    print("=" * 60)
    
    for question in questions:
        print(f"\n❓ 问题: {question}")
        print("-" * 60)
        
        result = engine.query(question, top_k=3)
        
        print(f"💡 答案:")
        print(result["answer"])
        
        if result["contexts"]:
            print(f"\n📖 参考来源 ({len(result['contexts'])} 条):")
            for ctx in result["contexts"]:
                print(f"  [{ctx['index']}] 相关性: {ctx['score']:.3f}")
                print(f"      {ctx['text'][:100]}...")
        
        print()
    
    # 4. 查看统计信息
    print("\n" + "=" * 60)
    print("知识库统计信息")
    print("=" * 60)
    stats = engine.get_stats()
    print(f"文档数量: {stats['document_count']}")
    print(f"使用重排序: {stats['use_reranker']}")
    print(f"使用混合检索: {stats['use_hybrid_retrieval']}")


def example_query_only():
    """仅查询示例（不重建索引）"""
    print("\n" + "=" * 60)
    print("RAG 知识库 - 仅查询示例")
    print("=" * 60)
    
    # 加载已有的知识库
    engine = RAGEngine(
        chroma_collection="demo_knowledge",
        chroma_persist_dir="./data/chroma_db",
    )
    
    # 交互式问答
    print("\n进入问答模式（输入 'quit' 或 'exit' 退出）\n")
    
    while True:
        question = input("❓ 您的问题: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("再见！")
            break
        
        if not question:
            continue
        
        result = engine.query(question, top_k=3)
        print(f"\n💡 {result['answer']}\n")


if __name__ == "__main__":
    # 运行基本示例
    example_basic_usage()
    
    # 询问是否进入交互模式
    try:
        choice = input("\n是否进入交互问答模式? (y/n): ").strip().lower()
        if choice == 'y':
            example_query_only()
    except KeyboardInterrupt:
        print("\n\n程序已退出")
