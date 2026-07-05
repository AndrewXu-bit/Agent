"""
生产力工具集 - 提供时间管理、日程安排等实用功能。
"""

from __future__ import annotations

import datetime

from agent_framework.core.tool_registry import tool


@tool("get_time", "获取当前日期和时间信息")
async def get_time(timezone: str = "local") -> str:
    """返回当前日期时间。
    
    Args:
        timezone: 时区标识（默认本地时区）
        
    Returns:
        格式化的时间字符串
    """
    now = datetime.datetime.now()
    return (
        f"当前时间: {now.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"星期: {now.strftime('%A')}\n"
        f"时间戳: {now.timestamp()}"
    )


@tool("format_date", "格式化日期为指定格式")
async def format_date(
    date_str: str = "today",
    format_str: str = "%Y-%m-%d"
) -> str:
    """将日期字符串格式化为指定格式。
    
    Args:
        date_str: 日期字符串或 'today'
        format_str: 目标格式
        
    Returns:
        格式化后的日期字符串
    """
    if date_str.lower() == "today":
        dt = datetime.datetime.now()
    else:
        try:
            dt = datetime.datetime.fromisoformat(date_str)
        except ValueError:
            return f"无法解析日期: {date_str}，请使用 ISO 格式 (YYYY-MM-DD)"
    
    return dt.strftime(format_str)


@tool("calculate_days", "计算两个日期之间的天数差")
async def calculate_days(
    start_date: str,
    end_date: str = "today"
) -> str:
    """计算两个日期之间相差的天数。
    
    Args:
        start_date: 开始日期 (ISO 格式)
        end_date: 结束日期 (ISO 格式或 'today')
        
    Returns:
        天数差
    """
    try:
        start = datetime.datetime.fromisoformat(start_date)
        if end_date.lower() == "today":
            end = datetime.datetime.now()
        else:
            end = datetime.datetime.fromisoformat(end_date)
        
        delta = end - start
        days = delta.days
        return f"从 {start_date} 到 {end_date} 相差 {days} 天"
    except ValueError as e:
        return f"日期格式错误: {e}"
