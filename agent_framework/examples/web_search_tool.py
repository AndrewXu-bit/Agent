"""
信息获取工具集 - 提供天气查询、URL内容抓取等网络功能。

包括：天气查询、网页内容获取、数学计算等。
"""

from __future__ import annotations

import asyncio
import datetime
import json

import httpx

from agent_framework.core.config import get_config
from agent_framework.core.tool_registry import registry, tool


@tool("get_time", "获取当前日期和时间信息")
async def get_time(timezone: str = "local") -> str:
    """返回当前日期时间。"""
    now = datetime.datetime.now()
    return (
        f"当前时间: {now.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"星期: {now.strftime('%A')}\n"
        f"时间戳: {now.timestamp()}"
    )


@tool(
    "get_weather",
    "获取指定城市的天气信息，支持当天实时天气及未来最多3天的预报。"
    "days 参数说明：0 或 'today' 表示今天（实时+今日预报），"
    "1 或 'tomorrow' 表示明天，2 表示后天；"
    "不传或传 'week' 表示返回今天起未来3天的逐日预报。",
)
async def get_weather(city: str = None, days=None) -> str:
    """查询城市天气，支持当天及未来3天预报。

    Args:
        city: 城市名称（中英文均可，如 Beijing、北京）。不传则使用配置中的默认城市。
        days: 要查询的天数。
              - 0 或 "today"：今天
              - 1 或 "tomorrow"：明天
              - 2：后天
              - "week" 或不传：未来3天逐日预报
    """
    # 从配置获取默认城市
    config = get_config()
    if city is None:
        city = config.get("tools.weather.default_city", "Beijing")

    timeout = config.get("tools.weather.timeout", 15)

    # 解析 days 参数
    day_index = _parse_days(days)

    try:
        # 使用 wttr.in 的 JSON API，可获取当前天气 + 未来3天预报
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(
                f"https://wttr.in/{city}?format=j1&lang=zh",
                headers={"Accept-Language": "zh-CN"},
            )
            resp.raise_for_status()
            data = resp.json()

        # 当前实时天气
        current = data.get("current_condition", [{}])[0]
        current_desc = _safe_get(current, "lang_zh", [{}], "value") or _safe_get(current, "weatherDesc", [{}], "value", default="未知")
        current_info = (
            f"🌡 温度: {current.get('temp_C', '?')}°C (体感 {current.get('FeelsLikeC', '?')}°C)\n"
            f"🌤 天气: {current_desc}\n"
            f"💧 湿度: {current.get('humidity', '?')}%\n"
            f"💨 风速: {current.get('windspeedKmph', '?')} km/h ({_wind_dir(current.get('winddir16Point', ''))})\n"
            f"👁 能见度: {current.get('visibility', '?')} km"
        )

        # 未来预报
        weather_list = data.get("weather", [])
        forecasts = _parse_forecasts(weather_list)

        if day_index == -1:
            # 返回未来3天逐日预报 + 当前天气
            lines = [f"📍 {city} 当前天气：\n{current_info}\n"]
            lines.append(f"📅 未来 {len(forecasts)} 天预报：")
            for fc in forecasts:
                lines.append(f"\n{fc['date']} {_weekday(fc['date'])}")
                lines.append(f"   🌡 {fc['mintemp']}~{fc['maxtemp']}°C  🌤 {_weather_desc_zh(fc.get('hourly', []))}")
                lines.append(f"   🌅 日出 {fc.get('astronomy', {}).get('sunrise', '?')} | 🌇 日落 {fc.get('astronomy', {}).get('sunset', '?')}")
            return "\n".join(lines)

        if day_index >= len(forecasts):
            return f"抱歉，wttr.in 最多只支持未来 {len(forecasts)} 天的预报（0=今天, 1=明天, 2=后天）。"

        fc = forecasts[day_index]
        if day_index == 0:
            header = f"📍 {city} 今天（{fc['date']} {_weekday(fc['date'])}）天气：\n当前：{current_info}\n"
        else:
            header = f"📍 {city} {fc['date']} {_weekday(fc['date'])} 天气预报：\n"

        detail = (
            f"🌡 温度: {fc['mintemp']}~{fc['maxtemp']}°C (平均 {fc['avgtemp']}°C)\n"
            f"🌤 天气: {_weather_desc_zh(fc.get('hourly', []))}\n"
            f"💧 降水概率: {_max_rain_chance(fc.get('hourly', []))}%\n"
            f"🌅 日出 {fc.get('astronomy', {}).get('sunrise', '?')} | 🌇 日落 {fc.get('astronomy', {}).get('sunset', '?')}\n"
            f"⏰ 逐时详情：\n{_hourly_detail(fc.get('hourly', []))}"
        )
        return header + detail

    except httpx.HTTPStatusError as exc:
        return f"获取天气失败（HTTP {exc.response.status_code}）: {exc.response.text[:200]}"
    except Exception as exc:
        return f"获取天气失败: {exc}"


def _parse_days(days) -> int:
    """将 days 参数解析为日期索引。返回 -1 表示返回所有天。"""
    if days is None or days == "" or str(days).lower() in ("week", "all", "全部", "一周"):
        return -1
    if isinstance(days, int):
        return max(0, days)
    s = str(days).strip().lower()
    mapping = {"today": 0, "今天": 0, "0": 0,
               "tomorrow": 1, "明天": 1, "1": 1,
               "后天": 2, "2": 2}
    if s in mapping:
        return mapping[s]
    try:
        return max(0, int(s))
    except ValueError:
        return -1


def _parse_forecasts(weather_list: list) -> list:
    """解析 wttr.in weather 数组为简洁的预报字典列表。"""
    forecasts = []
    for w in weather_list:
        forecasts.append({
            "date": w.get("date", "?"),
            "maxtemp": w.get("maxtempC", "?"),
            "mintemp": w.get("mintempC", "?"),
            "avgtemp": w.get("avgtempC", "?"),
            "astronomy": w.get("astronomy", [{}])[0] if w.get("astronomy") else {},
            "hourly": w.get("hourly", []),
        })
    return forecasts


def _safe_get(obj, key, nested_keys, final_key, default=""):
    """安全地从嵌套结构中取值。"""
    try:
        cur = obj.get(key, None)
        for nk in nested_keys:
            if cur is None:
                return default
            cur = cur[nk] if isinstance(cur, list) else cur.get(nk)
        return cur.get(final_key, default) if isinstance(cur, dict) else default
    except (KeyError, IndexError, TypeError, AttributeError):
        return default


def _weather_desc_zh(hourly: list) -> str:
    """从 hourly 数据中取中文天气描述（取中午时段为代表）。"""
    if not hourly:
        return "未知"
    # 取 12 点时段的描述作为代表
    target = hourly[4] if len(hourly) > 4 else hourly[len(hourly) // 2]
    desc = _safe_get(target, "lang_zh", [{}], "value")
    if not desc:
        desc = _safe_get(target, "weatherDesc", [{}], "value", default="未知")
    return desc


def _max_rain_chance(hourly: list) -> str:
    """取一天中最大降水概率。"""
    if not hourly:
        return "?"
    chances = [int(h.get("chanceofrain", "0") or "0") for h in hourly]
    return str(max(chances)) if chances else "?"


def _wind_dir(code: str) -> str:
    """风向代码转中文。"""
    mapping = {
        "N": "北风", "NNE": "北东北风", "NE": "东北风", "ENE": "东东北风",
        "E": "东风", "ESE": "东东南风", "SE": "东南风", "SSE": "南东南风",
        "S": "南风", "SSW": "南西南风", "SW": "西南风", "WSW": "西西南风",
        "W": "西风", "WNW": "西西北风", "NW": "西北风", "NNW": "北西北风",
    }
    return mapping.get(code, code)


def _weekday(date_str: str) -> str:
    """将 YYYY-MM-DD 转为星期几。"""
    try:
        dt = datetime.datetime.strptime(date_str, "%Y-%m-%d")
        names = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
        return names[dt.weekday()]
    except (ValueError, TypeError):
        return ""


def _hourly_detail(hourly: list) -> str:
    """生成逐时天气摘要（每3小时一个数据点，取关键时段）。"""
    if not hourly:
        return "  无逐时数据"
    # wttr.in 每天有8个时段（0,3,6,9,12,15,18,21点），取4个关键时段
    key_indices = [0, 2, 4, 6]
    lines = []
    for idx in key_indices:
        if idx >= len(hourly):
            break
        h = hourly[idx]
        time = h.get("time", "?").zfill(4)
        time = f"{time[:-2]}:{time[-2:]}" if len(time) == 4 else time
        desc = _safe_get(h, "lang_zh", [{}], "value") or _safe_get(h, "weatherDesc", [{}], "value", default="?")
        temp = h.get("tempC", "?")
        rain = h.get("chanceofrain", "0")
        lines.append(f"   {time}  {desc}  {temp}°C  降水{rain}%")
    return "\n".join(lines)


@tool("fetch_url", "获取一个 URL 的内容摘要")
async def fetch_url(url: str, max_length: int = None) -> str:
    """抓取 URL 内容前 max_length 个字符。"""
    # 从配置获取默认值
    config = get_config()
    if max_length is None:
        max_length = config.get("tools.url_fetcher.max_length", 500)
    
    timeout = config.get("tools.url_fetcher.timeout", 15)
    
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(url, follow_redirects=True)
            resp.raise_for_status()
            text = resp.text
            return text[:max_length]
    except Exception as exc:
        return f"获取 URL 失败: {exc}"


@tool("calculate", "通用计算引擎，支持数学表达式求值")
async def calculate(expression: str) -> str:
    """计算数学表达式，如 '2 + 3 * 4'。"""
    # 从配置获取安全模式设置
    config = get_config()
    safe_mode = config.get("tools.calculator.safe_mode", True)
    
    try:
        if safe_mode:
            # 安全计算：仅允许数学表达式
            allowed = set("0123456789+-*/.()% ")
            if not all(c in allowed for c in expression):
                return "Error: 表达式包含不允许的字符"
            result = eval(expression, {"__builtins__": {}}, {"math": __import__("math")})
        else:
            result = eval(expression)
        return f"{expression} = {result}"
    except Exception as exc:
        return f"计算错误: {exc}"


if __name__ == "__main__":
    import asyncio
    from agent_framework.core.config import get_config
    
    config = get_config()
    print(f"已加载配置，启用工具: {config.get('tools.enabled', [])}")