#!/usr/bin/env python3
"""
MCP Client 实现
使用 Ollama 模型（qwen3.5:9b 和 qwen3.5:35b-a3b）动态决定工具调用
"""

import json
import asyncio
import requests
from typing import List, Dict, Any, Optional
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


# 模型配置
MODELS = {
    "qwen3.5:9b": {
        "name": "qwen3.5:9b",
        "api_url": "http://localhost:11434/api/generate",
    },
    "qwen3.5:35b-a3b": {
        "name": "qwen3.5:35b-a3b",
        "api_url": "http://localhost:11434/api/generate",
    }
}


class MCPClient:
    """MCP 客户端，支持双模型对比"""
    
    def __init__(self, model_name: str = "qwen3.5:9b"):
        self.model_name = model_name
        self.model_config = MODELS.get(model_name, MODELS["qwen3.5:9b"])
        self.session: Optional[ClientSession] = None
        self.tools: List[Dict] = []
        
    async def connect_to_server(self):
        """连接到 MCP Server"""
        server_params = StdioServerParameters(
            command="python",
            args=["mcp_server.py"],
            env=None
        )
        
        stdio_transport = await stdio_client(server_params)
        self.stdio, self.write = stdio_transport
        self.session = await ClientSession(self.stdio, self.write)
        await self.session.initialize()
        
        # 动态发现工具
        response = await self.session.list_tools()
        self.tools = response.tools
        
        print(f"\n[{self.model_name}] 已连接到 MCP Server")
        print(f"  发现 {len(self.tools)} 个工具:")
        for tool in self.tools:
            print(f"    - {tool.name}: {tool.description[:50]}...")
    
    def call_ollama(self, prompt: str, system_prompt: str = "") -> str:
        """调用 Ollama 模型"""
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "system": system_prompt,
            "stream": False,
            "options": {"temperature": 0.3}
        }
        
        try:
            response = requests.post(
                self.model_config["api_url"],
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            result = response.json()
            return result.get("response", "")
        except Exception as e:
            return f"[错误] 调用模型失败: {str(e)}"
    
    def decide_tool_calls(self, user_query: str) -> List[Dict]:
        """使用大模型决定调用哪些工具"""
        
        # 构建工具描述
        tools_desc = "\n".join([
            f"{i+1}. {tool.name}: {tool.description}\n   参数: {tool.inputSchema}"
            for i, tool in enumerate(self.tools)
        ])
        
        system_prompt = f"""你是一个智能助手，需要根据用户的查询决定调用哪些工具。

可用工具列表:
{tools_desc}

你的任务是:
1. 分析用户查询的意图
2. 决定需要调用哪些工具（可以多个）
3. 确定每个工具的调用顺序
4. 为每个工具准备正确的参数

请以JSON格式返回工具调用计划:
{{
    "reasoning": "分析过程",
    "tool_calls": [
        {{
            "tool_name": "工具名称",
            "parameters": {{参数对象}},
            "depends_on": "依赖的上一步结果变量名（可选）"
        }}
    ]
}}"""
        
        prompt = f"用户查询: {user_query}\n\n请决定工具调用计划:"
        
        response = self.call_ollama(prompt, system_prompt)
        
        # 提取JSON
        try:
            # 尝试从响应中提取JSON
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0]
            else:
                json_str = response
            
            plan = json.loads(json_str.strip())
            return plan.get("tool_calls", [])
        except Exception as e:
            print(f"[{self.model_name}] 解析工具调用计划失败: {e}")
            print(f"  模型响应: {response[:200]}...")
            return []
    
    async def execute_tool(self, tool_name: str, parameters: Dict) -> str:
        """执行单个工具调用"""
        try:
            result = await self.session.call_tool(tool_name, parameters)
            return result.content[0].text if result.content else ""
        except Exception as e:
            return json.dumps({"error": str(e)})
    
    async def process_query(self, user_query: str) -> Dict[str, Any]:
        """处理用户查询"""
        print(f"\n{'='*60}")
        print(f"[{self.model_name}] 处理查询: {user_query}")
        print(f"{'='*60}")
        
        # 步骤1: 决定工具调用
        print("\n[步骤1] 分析意图并决定工具调用...")
        tool_calls = self.decide_tool_calls(user_query)
        
        if not tool_calls:
            return {
                "success": False,
                "error": "无法决定工具调用",
                "model": self.model_name
            }
        
        print(f"  计划调用 {len(tool_calls)} 个工具:")
        for i, call in enumerate(tool_calls, 1):
            print(f"    {i}. {call['tool_name']}")
        
        # 步骤2: 顺序执行工具调用
        print("\n[步骤2] 执行工具调用...")
        results = {}
        
        for i, call in enumerate(tool_calls):
            tool_name = call["tool_name"]
            parameters = call.get("parameters", {})
            
            # 处理依赖关系
            if "depends_on" in call and call["depends_on"] in results:
                dep_result = results[call["depends_on"]]
                # 将依赖结果注入参数
                for key, value in parameters.items():
                    if value == "{{previous_result}}":
                        parameters[key] = dep_result
            
            print(f"\n  调用 {tool_name}...")
            result = await self.execute_tool(tool_name, parameters)
            results[f"step_{i}"] = result
            print(f"  结果: {result[:100]}...")
        
        # 步骤3: 使用模型生成最终回答
        print("\n[步骤3] 生成最终回答...")
        final_prompt = f"""基于以下工具调用结果，回答用户的原始查询。

用户查询: {user_query}

工具调用结果:
{json.dumps(results, ensure_ascii=False, indent=2)}

请生成一个清晰、完整的回答。"""
        
        final_answer = self.call_ollama(final_prompt, "你是一个专业的HR助手，擅长解释工资和员工数据。")
        
        return {
            "success": True,
            "model": self.model_name,
            "query": user_query,
            "tool_calls": tool_calls,
            "results": results,
            "final_answer": final_answer
        }
    
    async def close(self):
        """关闭连接"""
        if self.session:
            await self.session.close()


async def compare_models(query: str):
    """对比两个模型的工具调用能力"""
    print("\n" + "="*70)
    print("MCP Client 双模型对比测试")
    print("="*70)
    print(f"\n测试查询: {query}")
    
    results = {}
    
    for model_name in MODELS.keys():
        client = MCPClient(model_name)
        try:
            await client.connect_to_server()
            result = await client.process_query(query)
            results[model_name] = result
        except Exception as e:
            print(f"\n[{model_name}] 错误: {e}")
            results[model_name] = {"success": False, "error": str(e)}
        finally:
            await client.close()
    
    # 输出对比结果
    print("\n" + "="*70)
    print("对比结果")
    print("="*70)
    
    for model_name, result in results.items():
        print(f"\n【{model_name}】")
        if result.get("success"):
            print(f"  工具调用: {len(result.get('tool_calls', []))} 个")
            print(f"  最终回答:\n{result.get('final_answer', 'N/A')[:300]}...")
        else:
            print(f"  失败: {result.get('error', 'Unknown')}")
    
    # 保存结果
    with open("mcp_comparison_result.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\n\n✓ 结果已保存到: mcp_comparison_result.json")


async def main():
    """主函数"""
    # 测试查询
    test_queries = [
        "帮我计算所有员工的工资并导出CSV",
        "分析各部门的人员分布情况",
        "计算技术部员工的税后工资",
    ]
    
    print("MCP Client - 使用 Ollama 模型进行工具调用")
    print("支持的模型:", list(MODELS.keys()))
    
    for query in test_queries:
        await compare_models(query)
        print("\n" + "-"*70)


if __name__ == "__main__":
    asyncio.run(main())
