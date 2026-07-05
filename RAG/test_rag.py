"""
RAG 模块测试脚本

验证所有组件是否可以正常导入和初始化。
"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_imports():
    """测试模块导入"""
    print("=" * 60)
    print("测试 RAG 模块导入")
    print("=" * 60)
    
    try:
        from RAG.document_loader import DocumentLoader
        print("✅ DocumentLoader 导入成功")
        
        from RAG.text_splitter import TextSplitter
        print("✅ TextSplitter 导入成功")
        
        from RAG.embedding import BGEEmbedding
        print("✅ BGEEmbedding 导入成功")
        
        from RAG.vector_store import ChromaVectorDB
        print("✅ ChromaVectorDB 导入成功")
        
        from RAG.reranker import BGEReranker
        print("✅ BGEReranker 导入成功")
        
        from RAG.retriever import HybridRetriever
        print("✅ HybridRetriever 导入成功")
        
        from RAG.engine import RAGEngine
        print("✅ RAGEngine 导入成功")
        
        print("\n🎉 所有模块导入成功！")
        return True
        
    except ImportError as e:
        print(f"\n❌ 模块导入失败: {e}")
        return False


def test_basic_init():
    """测试基本初始化（不加载模型）"""
    print("\n" + "=" * 60)
    print("测试基本初始化")
    print("=" * 60)
    
    try:
        from RAG.document_loader import DocumentLoader
        loader = DocumentLoader()
        print("✅ DocumentLoader 初始化成功")
        
        from RAG.text_splitter import TextSplitter
        splitter = TextSplitter(chunk_size=512, chunk_overlap=50)
        print("✅ TextSplitter 初始化成功")
        
        print("\n🎉 基本组件初始化成功！")
        return True
        
    except Exception as e:
        print(f"\n❌ 初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    results = []
    
    # 运行测试
    results.append(("模块导入", test_imports()))
    results.append(("基本初始化", test_basic_init()))
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{name}: {status}")
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    print(f"\n总计: {passed}/{total} 通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！可以开始使用 RAG 系统了。")
        sys.exit(0)
    else:
        print("\n⚠️  部分测试失败，请检查错误信息。")
        sys.exit(1)
