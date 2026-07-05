"""
测试飞书模块是否可以正常导入和初始化
"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """测试模块导入"""
    print("测试模块导入...")
    
    try:
        from feishu import FeishuBot, FeishuClient, FeishuWebhookServer
        print("✅ 模块导入成功")
        return True
    except ImportError as e:
        print(f"❌ 模块导入失败: {e}")
        return False

def test_client_init():
    """测试客户端初始化"""
    print("\n测试 FeishuClient 初始化...")
    
    try:
        from feishu import FeishuClient
        
        client = FeishuClient(
            app_id="test_app_id",
            app_secret="test_app_secret"
        )
        print(f"✅ FeishuClient 初始化成功")
        print(f"   - app_id: {client.app_id}")
        print(f"   - base_url: {client.base_url}")
        return True
    except Exception as e:
        print(f"❌ FeishuClient 初始化失败: {e}")
        return False

def test_bot_init():
    """测试 Bot 初始化（需要配置文件）"""
    print("\n测试 FeishuBot 初始化...")
    
    try:
        from feishu import FeishuBot
        
        # 使用 mock 模式测试
        bot = FeishuBot(
            app_id="test_app_id",
            app_secret="test_app_secret",
            config_path=None  # 使用默认配置
        )
        print(f"✅ FeishuBot 初始化成功")
        print(f"   - Agent 名称: {bot.agent.name}")
        print(f"   - 已挂载工具数: {len(bot.agent.tools)}")
        return True
    except Exception as e:
        print(f"❌ FeishuBot 初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_server_init():
    """测试服务器初始化"""
    print("\n测试 FeishuWebhookServer 初始化...")
    
    try:
        from feishu import FeishuWebhookServer
        
        server = FeishuWebhookServer(
            app_id="test_app_id",
            app_secret="test_app_secret",
            verification_token="test_token",
            host="127.0.0.1",
            port=8000
        )
        print(f"✅ FeishuWebhookServer 初始化成功")
        print(f"   - 监听地址: {server.host}:{server.port}")
        print(f"   - Webhook URL: http://{server.host}:{server.port}/webhook/event")
        return True
    except Exception as e:
        print(f"❌ FeishuWebhookServer 初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """运行所有测试"""
    print("=" * 60)
    print("飞书模块测试")
    print("=" * 60)
    
    results = []
    
    # 运行测试
    results.append(("模块导入", test_imports()))
    results.append(("客户端初始化", test_client_init()))
    results.append(("Bot 初始化", test_bot_init()))
    results.append(("服务器初始化", test_server_init()))
    
    # 打印测试结果
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{name}: {status}")
    
    # 统计
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    print(f"\n总计: {passed}/{total} 通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！")
        return 0
    else:
        print("\n⚠️  部分测试失败，请检查错误信息")
        return 1

if __name__ == "__main__":
    sys.exit(main())
