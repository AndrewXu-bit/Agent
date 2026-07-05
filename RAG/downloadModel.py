from modelscope import snapshot_download
# snapshot_download('BAAI/bge-m3', cache_dir='./model')
snapshot_download(
    'BAAI/bge-reranker-large',
    cache_dir=r'D:\Code\Agent\RAG\model'   # 使用绝对路径更保险
)