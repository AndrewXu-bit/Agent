"""
飞书机器人启动脚本

使用方式:
    python feishu_bot.py
"""

import asyncio
import logging
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from feishu.server import FeishuWebhookServer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/feishu_bot.log", encoding="utf-8"),
    ],
)

logger = logging.getLogger(__name__)


def main():
    """启动飞书机器人"""
    logger.info("=" * 60)
    logger.info("启动飞书机器人")
    logger.info("=" * 60)
    
    try:
        # 从配置文件创建服务器
        server = FeishuWebhookServer.create_from_config()
        
        logger.info("\n配置信息:")
        logger.info(f"  Webhook URL: http://0.0.0.0:{server.port}/webhook/event")
        logger.info(f"  健康检查: http://0.0.0.0:{server.port}/health")
        logger.info("\n请在飞书开发者后台配置事件订阅 URL")
        logger.info("=" * 60)
        
        # 启动服务器
        server.run()
        
    except ValueError as e:
        logger.error(f"配置错误: {e}")
        logger.error("\n请确保在 config.yaml 中配置了以下环境变量:")
        logger.error("  - FEISHU_APP_ID")
        logger.error("  - FEISHU_APP_SECRET")
        logger.error("  - FEISHU_VERIFICATION_TOKEN")
        sys.exit(1)
    except Exception as e:
        logger.error(f"启动失败: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
