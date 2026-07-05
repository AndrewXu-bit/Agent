"""
文档入库脚本 - 将本地文档构建为 RAG 知识库

用法:
    python build_index.py [文档目录路径]

示例:
    python build_index.py                          # 使用默认目录 ./data/documents
    python build_index.py D:\\docs                  # 指定文档目录
    python build_index.py --help                   # 查看帮助
"""

import sys
import argparse
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from RAG.engine import RAGEngine


def _load_rag_config() -> dict:
    """从 config.yaml 读取 RAG 配置，失败时返回空字典。"""
    try:
        import yaml
        config_path = project_root / "config.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data.get("rag", {}) or {}
    except Exception as e:
        print(f"⚠️  读取 config.yaml 失败，将使用默认配置: {e}")
        return {}


def _resolve_path(p: str) -> str:
    """将相对路径解析为基于项目根目录的绝对路径。"""
    if not p:
        return p
    path = Path(p)
    if not path.is_absolute():
        path = project_root / path
    return str(path)


def build_knowledge_base(
    docs_dir: str = "./data/documents",
    collection_name: str = "rag_knowledge",
    persist_dir: str = "./chroma_db",
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    use_reranker: bool = True,
    use_hybrid: bool = True,
    embedding_model_path: str = None,
    reranker_model_path: str = None,
):
    """
    构建知识库索引
    
    Args:
        docs_dir: 文档目录路径
        collection_name: Chroma 集合名称
        persist_dir: Chroma 持久化目录
        chunk_size: 文本块大小
        chunk_overlap: 文本块重叠
        use_reranker: 是否使用重排序
        use_hybrid: 是否使用混合检索
        embedding_model_path: 嵌入模型路径（本地路径或 HF 名称）
        reranker_model_path: 重排序模型路径（本地路径或 HF 名称）
    """
    print("=" * 70)
    print("📚 RAG 知识库 - 文档入库工具")
    print("=" * 70)
    
    # 检查文档目录
    docs_path = Path(docs_dir)
    if not docs_path.exists():
        print(f"\n❌ 错误: 文档目录不存在: {docs_dir}")
        print("\n请先创建目录并放入文档:")
        print(f"   mkdir -p {docs_dir}")
        print("\n支持的格式: PDF, Word, PPT, Excel, TXT, Markdown, HTML")
        return False
    
    # 统计文档数量
    supported_exts = ['.pdf', '.docx', '.doc', '.pptx', '.ppt', 
                      '.xlsx', '.xls', '.txt', '.md', '.html', '.htm']
    files = []
    for ext in supported_exts:
        files.extend(docs_path.glob(f"**/*{ext}"))
    
    if not files:
        print(f"\n⚠️  警告: 在 {docs_dir} 中未找到支持的文档文件")
        print(f"支持的格式: {', '.join(supported_exts)}")
        return False
    
    print(f"\n📁 文档目录: {docs_path.absolute()}")
    print(f"📄 找到 {len(files)} 个文档文件")
    
    # 初始化 RAG 引擎
    print("\n⚙️  初始化 RAG 引擎...")
    print(f"   - 文本块大小: {chunk_size}")
    print(f"   - 文本块重叠: {chunk_overlap}")
    print(f"   - 集合名称: {collection_name}")
    print(f"   - 持久化目录: {persist_dir}")
    print(f"   - 嵌入模型: {embedding_model_path}")
    if use_reranker:
        print(f"   - 重排序模型: {reranker_model_path}")
    print(f"   - 使用重排序: {'是' if use_reranker else '否'}")
    print(f"   - 使用混合检索: {'是' if use_hybrid else '否'}")
    
    try:
        engine = RAGEngine(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embedding_model_name=embedding_model_path,
            chroma_collection=collection_name,
            chroma_persist_dir=persist_dir,
            reranker_model_name=reranker_model_path if use_reranker else "BAAI/bge-reranker-large",
            use_reranker=use_reranker,
            use_hybrid_retrieval=use_hybrid,
        )
        print("✅ RAG 引擎初始化成功")
        
    except Exception as e:
        print(f"\n❌ RAG 引擎初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 构建索引
    print("\n" + "=" * 70)
    print("🔨 开始构建索引...")
    print("=" * 70)
    
    try:
        doc_count = engine.build_index(str(docs_dir), recursive=True)
        
        if doc_count == 0:
            print("\n⚠️  警告: 没有成功索引任何文档")
            return False
        
        print(f"\n✅ 索引构建完成!")
        print(f"   - 成功处理 {doc_count} 个文档片段")
        
        # 显示统计信息
        stats = engine.get_stats()
        print(f"\n📊 知识库统计:")
        print(f"   - 文档片段总数: {stats['document_count']}")
        print(f"   - 向量数据库: {stats['vector_db']['type']}")
        print(f"   - 嵌入模型: {stats['embedding_model']['type']}")
        print(f"   - 存储位置: {persist_dir}")
        
        print("\n" + "=" * 70)
        print("🎉 文档入库成功！")
        print("=" * 70)
        print("\n现在可以查询知识库了:")
        print(f"   python RAG/query.py --collection {collection_name} --persist-dir {persist_dir}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 索引构建失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    # 读取 config.yaml 中的 rag 配置作为默认值
    rag_cfg = _load_rag_config()
    emb_cfg = rag_cfg.get("embedding", {}) or {}
    rer_cfg = rag_cfg.get("reranker", {}) or {}
    chroma_cfg = rag_cfg.get("chroma", {}) or {}

    default_embedding_path = _resolve_path(emb_cfg.get("model_path", "BAAI/bge-m3"))
    default_reranker_path = _resolve_path(rer_cfg.get("model_path", "BAAI/bge-reranker-large"))
    default_docs_dir = _resolve_path(rag_cfg.get("docs_dir", "./data/documents"))

    parser = argparse.ArgumentParser(
        description="RAG 知识库 - 文档入库工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python build_index.py                              # 使用 config.yaml 默认配置
  python build_index.py D:\\docs                      # 指定文档目录
  python build_index.py --chunk-size 1024            # 自定义文本块大小
  python build_index.py --no-reranker                # 禁用重排序（更快）
  python build_index.py --embedding-path D:/models/bge-m3  # 覆盖嵌入模型路径
        """,
    )
    
    parser.add_argument(
        "docs_dir",
        nargs="?",
        default=default_docs_dir,
        help="文档目录路径 (默认: %(default)s)",
    )
    
    parser.add_argument(
        "--collection",
        default=chroma_cfg.get("collection", "rag_knowledge"),
        help="Chroma 集合名称 (默认: %(default)s)",
    )
    
    parser.add_argument(
        "--persist-dir",
        default=chroma_cfg.get("persist_dir", "./chroma_db"),
        help="Chroma 持久化目录 (默认: %(default)s)",
    )
    
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=rag_cfg.get("chunk_size", 512),
        help="文本块大小 (默认: %(default)s)",
    )
    
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=rag_cfg.get("chunk_overlap", 50),
        help="文本块重叠 (默认: %(default)s)",
    )

    parser.add_argument(
        "--embedding-path",
        default=default_embedding_path,
        help="嵌入模型路径（本地目录或 HF 名称，默认: %(default)s）",
    )

    parser.add_argument(
        "--reranker-path",
        default=default_reranker_path,
        help="重排序模型路径（本地目录或 HF 名称，默认: %(default)s）",
    )
    
    parser.add_argument(
        "--no-reranker",
        action="store_true",
        help="禁用重排序器（提高速度，可能降低精度）",
    )
    
    parser.add_argument(
        "--no-hybrid",
        action="store_true",
        help="禁用混合检索（仅使用向量检索）",
    )
    
    args = parser.parse_args()
    
    # 执行构建
    success = build_knowledge_base(
        docs_dir=args.docs_dir,
        collection_name=args.collection,
        persist_dir=args.persist_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        use_reranker=not args.no_reranker,
        use_hybrid=not args.no_hybrid,
        embedding_model_path=args.embedding_path,
        reranker_model_path=args.reranker_path,
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
