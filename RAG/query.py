"""
知识库查询脚本 - 对已构建的知识库进行问答查询

用法:
    python query.py "您的问题"
    python query.py --interactive    # 交互模式

示例:
    python query.py "什么是机器学习？"
    python query.py --collection my_docs --interactive
"""

import sys
import argparse
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from RAG.engine import RAGEngine


def query_knowledge_base(
    question: str,
    collection_name: str = "rag_knowledge",
    persist_dir: str = "./chroma_db",
    top_k: int = 3,
    use_reranker: bool = True,
):
    """
    查询知识库
    
    Args:
        question: 问题文本
        collection_name: Chroma 集合名称
        persist_dir: Chroma 持久化目录
        top_k: 返回的结果数量
        use_reranker: 是否使用重排序
    """
    print("=" * 70)
    print("🔍 RAG 知识库 - 查询")
    print("=" * 70)
    
    # 初始化 RAG 引擎（加载已有索引）
    print(f"\n📂 加载知识库: {collection_name}")
    print(f"   - 存储位置: {persist_dir}")
    
    try:
        engine = RAGEngine(
            chroma_collection=collection_name,
            chroma_persist_dir=persist_dir,
            use_reranker=use_reranker,
            use_hybrid_retrieval=True,
        )
        
        # 检查是否有文档
        stats = engine.get_stats()
        doc_count = stats['document_count']
        
        if doc_count == 0:
            print("\n⚠️  警告: 知识库中没有文档")
            print("请先运行 build_index.py 构建索引")
            return None
        
        print(f"✅ 知识库加载成功 (共 {doc_count} 个文档片段)")
        
    except Exception as e:
        print(f"\n❌ 加载知识库失败: {e}")
        print("\n可能的原因:")
        print("  1. 知识库尚未构建，请先运行: python RAG/build_index.py")
        print("  2. 路径或集合名称不正确")
        import traceback
        traceback.print_exc()
        return None
    
    # 执行查询
    print(f"\n❓ 问题: {question}")
    print("-" * 70)
    
    try:
        result = engine.query(question, top_k=top_k, use_reranker=use_reranker)
        
        if not result or not result.get('has_answer'):
            print("\n⚠️  未找到相关知识库内容")
            return None
        
        # 显示答案
        print(f"\n💡 答案:")
        print(result['answer'])
        
        # 显示参考来源
        if result.get('contexts'):
            print(f"\n📖 参考来源 ({len(result['contexts'])} 条):")
            print("-" * 70)
            
            for ctx in result['contexts']:
                score = ctx.get('score', 0)
                text = ctx.get('text', '')
                metadata = ctx.get('metadata', {})
                source = metadata.get('source', '未知')
                filename = metadata.get('filename', '未知文件')
                
                print(f"\n[{ctx['index']}] 相关性分数: {score:.4f}")
                print(f"     来源文件: {filename}")
                print(f"     完整路径: {source}")
                print(f"     内容预览: {text[:200]}...")
        
        return result
        
    except Exception as e:
        print(f"\n❌ 查询失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def interactive_mode(
    collection_name: str = "rag_knowledge",
    persist_dir: str = "./chroma_db",
    top_k: int = 3,
    use_reranker: bool = True,
):
    """
    交互式查询模式
    
    Args:
        collection_name: Chroma 集合名称
        persist_dir: Chroma 持久化目录
        top_k: 返回的结果数量
        use_reranker: 是否使用重排序
    """
    print("=" * 70)
    print("💬 RAG 知识库 - 交互式问答")
    print("=" * 70)
    
    # 初始化 RAG 引擎
    print(f"\n📂 加载知识库: {collection_name}")
    
    try:
        engine = RAGEngine(
            chroma_collection=collection_name,
            chroma_persist_dir=persist_dir,
            use_reranker=use_reranker,
            use_hybrid_retrieval=True,
        )
        
        stats = engine.get_stats()
        doc_count = stats['document_count']
        
        if doc_count == 0:
            print("\n⚠️  警告: 知识库中没有文档")
            print("请先运行 build_index.py 构建索引")
            return
        
        print(f"✅ 知识库加载成功 (共 {doc_count} 个文档片段)")
        
    except Exception as e:
        print(f"\n❌ 加载知识库失败: {e}")
        return
    
    print("\n" + "=" * 70)
    print("进入问答模式")
    print("输入问题后按回车，输入 'quit' 或 'exit' 退出")
    print("=" * 70)
    
    while True:
        try:
            question = input("\n❓ 您的问题: ").strip()
            
            if not question:
                continue
            
            if question.lower() in ['quit', 'exit', 'q', '退出']:
                print("\n👋 再见！")
                break
            
            # 执行查询
            result = engine.query(question, top_k=top_k, use_reranker=use_reranker)
            
            if result and result.get('has_answer'):
                print(f"\n💡 {result['answer']}")
                
                if result.get('contexts'):
                    print(f"\n📖 找到 {len(result['contexts'])} 条相关参考")
            else:
                print("\n⚠️  未找到相关知识库内容")
        
        except KeyboardInterrupt:
            print("\n\n👋 再见！")
            break
        except EOFError:
            print("\n\n👋 再见！")
            break


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="RAG 知识库 - 查询工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python query.py "什么是机器学习？"              # 单次查询
  python query.py --interactive                   # 交互模式
  python query.py "AI" --top-k 5                  # 返回5条结果
  python query.py --collection my_docs --interactive
        """,
    )
    
    parser.add_argument(
        "question",
        nargs="?",
        default=None,
        help="要查询的问题",
    )
    
    parser.add_argument(
        "--collection",
        default="rag_knowledge",
        help="Chroma 集合名称 (默认: rag_knowledge)",
    )
    
    parser.add_argument(
        "--persist-dir",
        default="./chroma_db",
        help="Chroma 持久化目录 (默认: ./chroma_db)",
    )
    
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="返回的结果数量 (默认: 3)",
    )
    
    parser.add_argument(
        "--no-reranker",
        action="store_true",
        help="禁用重排序器",
    )
    
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="进入交互问答模式",
    )
    
    args = parser.parse_args()
    
    # 如果没有提供问题且不是交互模式，默认进入交互模式
    if args.question is None and not args.interactive:
        args.interactive = True
    
    if args.interactive:
        # 交互模式
        interactive_mode(
            collection_name=args.collection,
            persist_dir=args.persist_dir,
            top_k=args.top_k,
            use_reranker=not args.no_reranker,
        )
    else:
        # 单次查询
        query_knowledge_base(
            question=args.question,
            collection_name=args.collection,
            persist_dir=args.persist_dir,
            top_k=args.top_k,
            use_reranker=not args.no_reranker,
        )


if __name__ == "__main__":
    main()
