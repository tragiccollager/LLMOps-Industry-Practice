#!/usr/bin/env python3
"""
MCP Client 实现
使用 Ollama 模型（qwen3.5:9b 和 qwen3.5:35b-a3b）动态决定工具调用
"""

import json
import asyncio
import requests
import subprocess
from typing import List, Dict, Any, Optional
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def stop_other_models(current_model: str):
    """停止其他正在运行的模型以释放显存"""
    for model_name in MODELS.keys():
        if model_name != current_model:
            try:
                # 使用 utf-8 编码避免 Windows 中文错误
                result = subprocess.run(
                    ["ollama", "stop", model_name],
                    capture_output=True,
                    encoding='utf-8',
                    errors='ignore',
                    timeout=10
                )
                if result.returncode == 0:
                    print(f"  ✓ 已停止模型: {model_name}")
            except Exception:
                # 忽略错误（如模型未运行或ollama命令失败）
                pass


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
        
        # 使用 async with 上下文管理器
        self._client_ctx = stdio_client(server_params)
        stdio_transport = await self._client_ctx.__aenter__()
        self.stdio, self.write = stdio_transport
        self.session = await ClientSession(self.stdio, self.write).__aenter__()
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
                timeout=300  # 增加到5分钟，防止生成最终回答时超时
            )
            response.raise_for_status()
            result = response.json()
            return result.get("response", "")
        except Exception as e:
            return f"[错误] 调用模型失败: {str(e)}"
    
    def decide_tool_calls(self, user_query: str) -> List[Dict]:
        """使用大模型决定调用哪些工具"""
        
        # 构建工具描述（简化版）
        tools_desc = "\n".join([
            f"{i+1}. {tool.name}: {tool.description[:80]}..."
            for i, tool in enumerate(self.tools)
        ])
        
        system_prompt = f"""你是一个智能助手，需要根据用户的查询决定调用哪些工具。

可用工具列表:
{tools_desc}

工具参数说明:
- get_employee_directory: department (可选，字符串，如"技术部"，不传则返回所有员工)
- calculate_payroll: employee_json (必需，JSON字符串), tax_rate (可选，默认0.1), insurance_rate (可选，默认0.08)
- export_to_csv: data_json (必需，JSON字符串), filename (可选，默认"export.csv")
- analyze_department_summary: employee_json (必需，JSON字符串)

重要规则:
1. 如果查询涉及"所有员工"或"各部门"，get_employee_directory 的 department 参数设为 null 或不传
2. 工具之间有依赖关系时，使用 "output_of_step_X" 作为参数值引用上一步结果
3. 只返回JSON，不要包含其他文字

请以JSON格式返回工具调用计划:
{{
    "tool_calls": [
        {{
            "tool_name": "工具名称",
            "parameters": {{"参数名": "参数值"}}
        }}
    ]
}}"""
        
        prompt = f"用户查询: {user_query}\n\n请决定工具调用计划，只返回JSON:"
        
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
            
            # 清理可能的额外字符
            json_str = json_str.strip()
            if json_str.startswith('"') and json_str.endswith('"'):
                json_str = json_str[1:-1]
            
            plan = json.loads(json_str)
            calls = plan.get("tool_calls", [])
            
            # 处理参数中的特殊值
            for call in calls:
                params = call.get("parameters", {})
                for key, value in params.items():
                    if value == "null" or value == "":
                        params[key] = None
                    elif isinstance(value, str) and value.startswith("output_of_step_"):
                        # 标记需要替换
                        call["_needs_replacement"] = True
                        call["_replacement_key"] = key
                        call["_replacement_step"] = int(value.split("_")[-1])
            
            return calls
        except Exception as e:
            print(f"[{self.model_name}] 解析工具调用计划失败: {e}")
            print(f"  模型响应: {response[:300]}...")
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
            parameters = call.get("parameters", {}).copy()
            
            # 处理依赖关系 - 替换上一步的结果
            if call.get("_needs_replacement"):
                step_idx = call.get("_replacement_step", 0)
                if step_idx <= i:
                    prev_result = results.get(f"step_{step_idx}", "")
                    key = call.get("_replacement_key")
                    if key and prev_result:
                        parameters[key] = prev_result
            
            # 处理其他可能的引用
            for key, value in list(parameters.items()):
                if isinstance(value, str):
                    if value.startswith("output_of_step_"):
                        step_idx = int(value.split("_")[-1])
                        if step_idx <= i:
                            parameters[key] = results.get(f"step_{step_idx}", "")
                    elif value == "null" or value == "":
                        parameters[key] = None
            
            # Fallback: 如果参数为空且看起来像需要JSON，使用前一步结果
            if i > 0:
                json_params = ['employee_json', 'data_json', 'employees', 'data']
                for key in json_params:
                    if key in parameters and (parameters[key] is None or parameters[key] == ""):
                        prev_step_result = results.get(f"step_{i}", "")
                        if prev_step_result:
                            parameters[key] = prev_step_result
                            print(f"    [自动修复] 使用前一步结果填充 {key}")
                            break
            
            print(f"\n  调用 {tool_name}，参数: {parameters}")
            result = await self.execute_tool(tool_name, parameters)
            results[f"step_{i+1}"] = result  # 使用1-based索引，与LLM的output_of_step_X对应
            print(f"  结果: {result[:150]}...")
        
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
        try:
            if hasattr(self, 'session') and self.session:
                await self.session.__aexit__(None, None, None)
            if hasattr(self, '_client_ctx') and self._client_ctx:
                await self._client_ctx.__aexit__(None, None, None)
        except Exception as e:
            print(f"[{self.model_name}] 关闭连接时出错: {e}")


async def compare_models(query: str, models: list = None):
    """对比两个模型的工具调用能力"""
    print("\n" + "="*70)
    print("MCP Client 双模型对比测试")
    print("="*70)
    print(f"\n测试查询: {query}")
    
    results = {}
    test_models = models if models else list(MODELS.keys())
    
    for model_name in test_models:
        # 停止其他模型以释放显存
        stop_other_models(model_name)
        
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
            # 测试完当前模型后也停止它，为下一个模型释放显存
            stop_other_models(None)  # None会停止所有模型
    
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
    print("\n提示: 如果 35b 模型报 500 错误，可能是显存不足或模型未加载")
    print("      可以先只测试 9b 模型")
    
    # 选择测试模式：只测试9b，或双模型对比
    # 单模型测试（推荐）
    models_to_test = ["qwen3.5:9b"]
    
    # 双模型测试（取消下行注释）
    # models_to_test = list(MODELS.keys())
    
    for query in test_queries:
        await compare_models(query, models=models_to_test)
        print("\n" + "-"*70)


if __name__ == "__main__":
    asyncio.run(main())
