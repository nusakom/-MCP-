import os
import json
from translate_mcp import TranslateMCP, simple_translate, fake_translate, echo_translate

def handler(event, context):
    req = json.loads(event)
    mcp = TranslateMCP()
    # 支持通过环境变量动态注册通道
    enabled_channels = os.environ.get("MCP_CHANNELS", "simple,fake,echo").split(",")
    for ch in enabled_channels:
        ch = ch.strip()
        if ch == "simple":
            mcp.register_channel("simple", simple_translate)
        elif ch == "fake":
            mcp.register_channel("fake", fake_translate)
        elif ch == "echo":
            mcp.register_channel("echo", echo_translate)
    # 你也可以根据实际需求注册更多通道
    result = mcp.mcp_api(req)
    return json.dumps(result, ensure_ascii=False)