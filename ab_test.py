#!/usr/bin/env python3
"""
AB测试脚本：对比 qwen3.5:9b 和 qwen3.5:35b-a3b 两个模型的性能
"""

import time
import csv
import json
import sys
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('ab_test.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


class ErrorType(Enum):
    """错误类型枚举"""
    CONNECTION_ERROR = "connection_error"
    TIMEOUT_ERROR = "timeout_error"
    HTTP_ERROR = "http_error"
    JSON_ERROR = "json_error"
    MODEL_NOT_FOUND = "model_not_found"
    UNKNOWN_ERROR = "unknown_error"


@dataclass
class TestError:
    """测试错误信息"""
    error_type: ErrorType
    message: str
    details: Optional[str] = None
    
    def __str__(self):
        if self.details:
            return f"[{self.error_type.value}] {self.message}: {self.details}"
        return f"[{self.error_type.value}] {self.message}"


class ModelAPIException(Exception):
    """模型API调用异常"""
    def __init__(self, error: TestError):
        self.error = error
        super().__init__(str(error))


# 模型配置
MODELS = {
    "qwen3.5:9b": {
        "name": "qwen3.5:9b",
        "api_url": "http://localhost:11434/api/generate",  # Ollama 默认地址
    },
    "qwen3.5:35b-a3b": {
        "name": "qwen3.5:35b-a3b",
        "api_url": "http://localhost:11434/api/generate",
    }
}

# 测试配置
CONFIG = {
    "max_retries": 3,              # 最大重试次数
    "retry_delay": 2,              # 重试间隔（秒）
    "request_timeout": 300,        # 请求超时（秒）
    "connection_timeout": 10,      # 连接超时（秒）
    "backoff_factor": 1,           # 重试退避因子
}

# 测试提示词列表（短、中、长三种）
TEST_PROMPTS = {
    "short": [
        "你好",
        "什么是AI？",
        "1+1等于几？",
    ],
    "medium": [
        "请解释一下什么是机器学习，并举例说明。",
        "简述Python和JavaScript的主要区别。",
        "什么是RESTful API？它有哪些设计原则？",
    ],
    "long": [
        """请详细解释深度学习的工作原理，包括神经网络的基本结构、前向传播、反向传播算法，
        以及常用的优化器如Adam和SGD的区别。同时请举例说明CNN和RNN在不同场景下的应用。""",
        """请写一篇关于人工智能发展历程的简要综述，从图灵测试开始，
        历经专家系统、机器学习时代，一直到现在的深度学习和大语言模型时代。
        要求涵盖每个阶段的关键技术突破和代表性成果。""",
        """设计一个电商推荐系统的架构方案，需要包括：1）数据收集与处理模块；
        2）特征工程方案；3）推荐算法选择（协同过滤、内容推荐、深度学习模型）；
        4）在线 serving 架构；5）A/B测试与效果评估方案。请详细说明每个模块的设计思路。""",
    ]
}


def create_session_with_retries() -> requests.Session:
    """创建带重试机制的请求会话"""
    session = requests.Session()
    retry_strategy = Retry(
        total=CONFIG["max_retries"],
        backoff_factor=CONFIG["backoff_factor"],
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["POST", "GET"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def classify_error(exception: Exception, response: Optional[requests.Response] = None) -> TestError:
    """
    分类错误类型
    
    Args:
        exception: 异常对象
        response: HTTP响应对象（可选）
        
    Returns:
        分类后的错误信息
    """
    if isinstance(exception, requests.exceptions.Timeout):
        return TestError(
            ErrorType.TIMEOUT_ERROR,
            "请求超时",
            str(exception)
        )
    elif isinstance(exception, requests.exceptions.ConnectionError):
        return TestError(
            ErrorType.CONNECTION_ERROR,
            "连接错误，请检查Ollama服务是否运行",
            str(exception)
        )
    elif isinstance(exception, requests.exceptions.HTTPError):
        return TestError(
            ErrorType.HTTP_ERROR,
            f"HTTP错误: {exception.response.status_code if hasattr(exception, 'response') else 'Unknown'}",
            str(exception)
        )
    elif isinstance(exception, json.JSONDecodeError):
        return TestError(
            ErrorType.JSON_ERROR,
            "JSON解析错误",
            str(exception)
        )
    elif response and response.status_code == 404:
        return TestError(
            ErrorType.MODEL_NOT_FOUND,
            "模型未找到，请确认模型已下载",
            response.text
        )
    else:
        return TestError(
            ErrorType.UNKNOWN_ERROR,
            "未知错误",
            str(exception)
        )


def call_ollama_model(
    model_name: str, 
    prompt: str, 
    api_url: str = "http://localhost:11434/api/generate",
    session: Optional[requests.Session] = None
) -> Dict[str, Any]:
    """
    调用 Ollama API 获取模型响应（带重试机制）
    
    Args:
        model_name: 模型名称
        prompt: 提示词
        api_url: Ollama API 地址
        session: 请求会话（可选）
        
    Returns:
        包含响应内容和性能指标的字典
    """
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False
    }
    
    if session is None:
        session = create_session_with_retries()
    
    last_error = None
    
    for attempt in range(CONFIG["max_retries"]):
        try:
            logger.debug(f"调用模型 {model_name}, 尝试 {attempt + 1}/{CONFIG['max_retries']}")
            
            start_time = time.time()
            response = session.post(
                api_url, 
                json=payload, 
                timeout=(CONFIG["connection_timeout"], CONFIG["request_timeout"])
            )
            end_time = time.time()
            
            # 检查HTTP错误
            response.raise_for_status()
            
            if response.status_code == 200:
                try:
                    result = response.json()
                    logger.debug(f"模型 {model_name} 调用成功")
                    return {
                        "success": True,
                        "response": result.get("response", ""),
                        "total_tokens": result.get("eval_count", 0),
                        "prompt_tokens": result.get("prompt_eval_count", 0),
                        "total_time": end_time - start_time,
                        "error": None
                    }
                except json.JSONDecodeError as e:
                    error = classify_error(e)
                    logger.warning(f"JSON解析错误: {error}")
                    raise ModelAPIException(error)
            else:
                error = TestError(
                    ErrorType.HTTP_ERROR,
                    f"HTTP {response.status_code}",
                    response.text[:200]
                )
                raise ModelAPIException(error)
                
        except requests.exceptions.RequestException as e:
            last_error = classify_error(e)
            logger.warning(f"请求失败 (尝试 {attempt + 1}/{CONFIG['max_retries']}): {last_error}")
            
            if attempt < CONFIG["max_retries"] - 1:
                wait_time = CONFIG["retry_delay"] * (2 ** attempt)  # 指数退避
                logger.info(f"等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
            else:
                logger.error(f"达到最大重试次数，放弃请求")
        
        except ModelAPIException:
            raise
        
        except Exception as e:
            last_error = classify_error(e)
            logger.error(f"未预期的错误: {last_error}")
            break
    
    # 所有重试都失败了
    return {
        "success": False,
        "response": "",
        "total_tokens": 0,
        "prompt_tokens": 0,
        "total_time": 0,
        "error": str(last_error) if last_error else "未知错误"
    }


def call_model_streaming(
    model_name: str, 
    prompt: str, 
    api_url: str = "http://localhost:11434/api/generate",
    session: Optional[requests.Session] = None
) -> Dict[str, Any]:
    """
    使用流式调用获取 TTFT (Time To First Token) 和总延迟（带重试机制）
    
    Args:
        model_name: 模型名称
        prompt: 提示词
        api_url: Ollama API 地址
        session: 请求会话（可选）
        
    Returns:
        包含 TTFT、总延迟和其他指标的字典
    """
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": True
    }
    
    if session is None:
        session = create_session_with_retries()
    
    last_error = None
    
    for attempt in range(CONFIG["max_retries"]):
        full_response = []
        first_token_time = None
        start_time = time.time()
        total_tokens = 0
        json_errors = 0
        max_json_errors = 10  # 允许的最大JSON解析错误数
        
        try:
            logger.debug(f"流式调用模型 {model_name}, 尝试 {attempt + 1}/{CONFIG['max_retries']}")
            
            response = session.post(
                api_url, 
                json=payload, 
                stream=True, 
                timeout=(CONFIG["connection_timeout"], CONFIG["request_timeout"])
            )
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line.decode('utf-8'))
                        
                        # 检查错误字段
                        if "error" in data:
                            error_msg = data["error"]
                            if "not found" in error_msg.lower():
                                raise ModelAPIException(TestError(
                                    ErrorType.MODEL_NOT_FOUND,
                                    f"模型 '{model_name}' 未找到",
                                    error_msg
                                ))
                            raise ModelAPIException(TestError(
                                ErrorType.UNKNOWN_ERROR,
                                "模型返回错误",
                                error_msg
                            ))
                        
                        # 记录第一个token到达的时间
                        if first_token_time is None and data.get("response"):
                            first_token_time = time.time()
                        
                        if data.get("response"):
                            full_response.append(data["response"])
                        
                        # 检查是否完成
                        if data.get("done"):
                            total_tokens = data.get("eval_count", 0)
                            break
                            
                    except json.JSONDecodeError:
                        json_errors += 1
                        if json_errors > max_json_errors:
                            raise ModelAPIException(TestError(
                                ErrorType.JSON_ERROR,
                                f"JSON解析错误次数超过限制 ({max_json_errors})"
                            ))
                        continue
            
            end_time = time.time()
            
            # 验证是否收到了响应
            if not full_response and not total_tokens:
                logger.warning(f"模型 {model_name} 返回空响应")
            
            logger.debug(f"流式调用成功，生成 {total_tokens} tokens")
            
            return {
                "success": True,
                "response": "".join(full_response),
                "ttft": first_token_time - start_time if first_token_time else 0,
                "total_latency": end_time - start_time,
                "total_tokens": total_tokens,
                "tokens_per_second": total_tokens / (end_time - start_time) if (end_time - start_time) > 0 else 0,
                "error": None
            }
            
        except requests.exceptions.RequestException as e:
            last_error = classify_error(e)
            logger.warning(f"流式请求失败 (尝试 {attempt + 1}/{CONFIG['max_retries']}): {last_error}")
            
            if attempt < CONFIG["max_retries"] - 1:
                wait_time = CONFIG["retry_delay"] * (2 ** attempt)
                logger.info(f"等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
            else:
                logger.error(f"达到最大重试次数，放弃请求")
        
        except ModelAPIException:
            raise
        
        except Exception as e:
            end_time = time.time()
            last_error = classify_error(e)
            logger.error(f"流式调用未预期的错误: {last_error}")
            break
    
    # 所有重试都失败了
    return {
        "success": False,
        "response": "",
        "ttft": 0,
        "total_latency": 0,
        "total_tokens": 0,
        "tokens_per_second": 0,
        "error": str(last_error) if last_error else "未知错误"
    }


def check_model_available(model_name: str, api_url: str) -> Tuple[bool, Optional[str]]:
    """
    检查模型是否可用
    
    Args:
        model_name: 模型名称
        api_url: API基础地址
        
    Returns:
        (是否可用, 错误信息)
    """
    try:
        # 尝试获取模型列表
        base_url = api_url.rsplit('/', 1)[0]  # 移除 /generate
        tags_url = f"{base_url}/tags"
        
        response = requests.get(tags_url, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        models = [m.get("name", "") for m in data.get("models", [])]
        
        if model_name in models:
            logger.info(f"模型 '{model_name}' 已就绪")
            return True, None
        else:
            available = ", ".join(models[:5]) + ("..." if len(models) > 5 else "")
            return False, f"模型未找到。可用模型: {available}"
            
    except Exception as e:
        return False, f"无法检查模型状态: {str(e)}"


def run_benchmark(model_name: str, model_config: Dict[str, str]) -> List[Dict[str, Any]]:
    """
    对单个模型运行所有测试用例（带异常处理）
    
    Args:
        model_name: 模型名称
        model_config: 模型配置
        
    Returns:
        测试结果列表
    """
    results = []
    api_url = model_config.get("api_url", "http://localhost:11434/api/generate")
    
    print(f"\n{'='*60}")
    print(f"正在测试模型: {model_name}")
    print(f"{'='*60}")
    
    # 检查模型是否可用
    is_available, error_msg = check_model_available(model_name, api_url)
    if not is_available:
        logger.error(f"模型检查失败: {error_msg}")
        print(f"⚠ 警告: {error_msg}")
        print("将继续尝试测试，但可能会失败...")
    
    # 创建会话（复用连接）
    session = create_session_with_retries()
    
    # 统计信息
    success_count = 0
    fail_count = 0
    error_types = {}
    
    for prompt_type, prompts in TEST_PROMPTS.items():
        print(f"\n[{prompt_type.upper()} 提示词测试]")
        
        for i, prompt in enumerate(prompts, 1):
            print(f"  测试 {i}/{len(prompts)}...", end=" ", flush=True)
            
            try:
                # 使用流式调用获取 TTFT
                metrics = call_model_streaming(model_name, prompt, api_url, session)
                
                if metrics["success"]:
                    print(f"✓ TTFT: {metrics['ttft']:.3f}s, 总延迟: {metrics['total_latency']:.3f}s")
                    success_count += 1
                else:
                    print(f"✗ 错误: {metrics['error']}")
                    fail_count += 1
                    # 记录错误类型
                    error_key = metrics['error'].split(']')[0] + "]" if '[' in metrics['error'] else "Unknown"
                    error_types[error_key] = error_types.get(error_key, 0) + 1
                
                results.append({
                    "timestamp": datetime.now().isoformat(),
                    "model": model_name,
                    "prompt_type": prompt_type,
                    "prompt_index": i,
                    "prompt_length": len(prompt),
                    "ttft_seconds": round(metrics["ttft"], 4),
                    "total_latency_seconds": round(metrics["total_latency"], 4),
                    "total_tokens": metrics["total_tokens"],
                    "tokens_per_second": round(metrics["tokens_per_second"], 2),
                    "success": metrics["success"],
                    "error": metrics["error"] if not metrics["success"] else ""
                })
                
            except ModelAPIException as e:
                logger.error(f"API异常: {e.error}")
                print(f"✗ API错误: {e.error.message}")
                fail_count += 1
                error_types[e.error.error_type.value] = error_types.get(e.error.error_type.value, 0) + 1
                
                results.append({
                    "timestamp": datetime.now().isoformat(),
                    "model": model_name,
                    "prompt_type": prompt_type,
                    "prompt_index": i,
                    "prompt_length": len(prompt),
                    "ttft_seconds": 0,
                    "total_latency_seconds": 0,
                    "total_tokens": 0,
                    "tokens_per_second": 0,
                    "success": False,
                    "error": str(e.error)
                })
                
            except Exception as e:
                logger.exception(f"未预期的异常: {e}")
                print(f"✗ 异常: {str(e)[:50]}")
                fail_count += 1
                error_types["unexpected"] = error_types.get("unexpected", 0) + 1
                
                results.append({
                    "timestamp": datetime.now().isoformat(),
                    "model": model_name,
                    "prompt_type": prompt_type,
                    "prompt_index": i,
                    "prompt_length": len(prompt),
                    "ttft_seconds": 0,
                    "total_latency_seconds": 0,
                    "total_tokens": 0,
                    "tokens_per_second": 0,
                    "success": False,
                    "error": f"[unexpected] {str(e)}"
                })
    
    # 打印测试摘要
    print(f"\n[{model_name} 测试摘要]")
    print(f"  成功: {success_count}, 失败: {fail_count}")
    if error_types:
        print(f"  错误类型分布: {error_types}")
    
    # 关闭会话
    session.close()
    
    return results


def save_results_to_csv(results: List[Dict[str, Any]], filename: str = "performance_metrics.csv"):
    """
    将测试结果保存到 CSV 文件
    
    Args:
        results: 测试结果列表
        filename: CSV 文件名
    """
    if not results:
        print("没有结果可保存")
        return
    
    fieldnames = [
        "timestamp", "model", "prompt_type", "prompt_index",
        "prompt_length", "ttft_seconds", "total_latency_seconds",
        "total_tokens", "tokens_per_second", "success", "error"
    ]
    
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\n✓ 结果已保存到: {filename}")


def print_comparison_table(results: List[Dict[str, Any]]):
    """
    打印对比表格
    
    Args:
        results: 测试结果列表
    """
    print(f"\n{'='*80}")
    print("性能对比结果")
    print(f"{'='*80}")
    
    # 按模型和提示词类型分组统计
    from collections import defaultdict
    
    stats = defaultdict(lambda: defaultdict(list))
    
    for r in results:
        if r["success"]:
            model = r["model"]
            ptype = r["prompt_type"]
            stats[model][ptype].append({
                "ttft": r["ttft_seconds"],
                "latency": r["total_latency_seconds"],
                "tps": r["tokens_per_second"]
            })
    
    # 打印详细对比
    print("\n【各模型各提示词类型平均性能】")
    print("-" * 80)
    print(f"{'模型':<20} {'提示词类型':<10} {'平均TTFT(s)':<15} {'平均总延迟(s)':<15} {'平均TPS':<10}")
    print("-" * 80)
    
    for model in MODELS.keys():
        for ptype in ["short", "medium", "long"]:
            if ptype in stats[model] and stats[model][ptype]:
                data = stats[model][ptype]
                avg_ttft = sum(d["ttft"] for d in data) / len(data)
                avg_latency = sum(d["latency"] for d in data) / len(data)
                avg_tps = sum(d["tps"] for d in data) / len(data)
                
                print(f"{model:<20} {ptype:<10} {avg_ttft:<15.3f} {avg_latency:<15.3f} {avg_tps:<10.2f}")
    
    # 打印总体对比
    print("\n" + "=" * 80)
    print("【总体性能对比】")
    print("-" * 80)
    print(f"{'模型':<25} {'平均TTFT(s)':<15} {'平均总延迟(s)':<15} {'平均TPS':<10}")
    print("-" * 80)
    
    for model in MODELS.keys():
        all_data = []
        for ptype_data in stats[model].values():
            all_data.extend(ptype_data)
        
        if all_data:
            avg_ttft = sum(d["ttft"] for d in all_data) / len(all_data)
            avg_latency = sum(d["latency"] for d in all_data) / len(all_data)
            avg_tps = sum(d["tps"] for d in all_data) / len(all_data)
            
            print(f"{model:<25} {avg_ttft:<15.3f} {avg_latency:<15.3f} {avg_tps:<10.2f}")
    
    print("=" * 80)


def signal_handler(sig, frame):
    """处理中断信号"""
    print("\n\n⚠ 测试被用户中断")
    sys.exit(0)


def main():
    """主函数（带全局异常处理）"""
    import signal
    signal.signal(signal.SIGINT, signal_handler)
    
    print("="*60)
    print("LLM AB 性能测试")
    print("对比模型: qwen3.5:9b vs qwen3.5:35b-a3b")
    print("="*60)
    
    logger.info("测试开始")
    start_time = time.time()
    all_results = []
    
    try:
        # 测试每个模型
        for model_key, model_config in MODELS.items():
            try:
                results = run_benchmark(model_key, model_config)
                all_results.extend(results)
            except Exception as e:
                logger.exception(f"测试模型 {model_key} 时发生严重错误")
                print(f"\n✗ 模型 {model_key} 测试失败: {str(e)}")
                continue
        
        # 保存结果到 CSV
        save_results_to_csv(all_results, "performance_metrics.csv")
        
        # 输出对比表格
        print_comparison_table(all_results)
        
        # 统计总体成功率
        total_tests = len(all_results)
        successful_tests = sum(1 for r in all_results if r["success"])
        
        elapsed = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"测试完成!")
        print(f"总耗时: {elapsed:.1f}秒")
        print(f"成功率: {successful_tests}/{total_tests} ({100*successful_tests/total_tests:.1f}%)")
        print(f"{'='*60}")
        logger.info(f"测试完成，成功率: {successful_tests}/{total_tests}")
        
    except KeyboardInterrupt:
        print("\n\n⚠ 测试被用户中断")
        logger.warning("测试被用户中断")
        
        # 保存已收集的结果
        if all_results:
            save_results_to_csv(all_results, "performance_metrics_partial.csv")
            print(f"已保存部分结果 ({len(all_results)} 条)")
        
    except Exception as e:
        logger.exception(f"测试过程中发生严重错误: {e}")
        print(f"\n✗ 测试失败: {str(e)}")
        
        # 保存已收集的结果
        if all_results:
            save_results_to_csv(all_results, "performance_metrics_partial.csv")
            print(f"已保存部分结果 ({len(all_results)} 条)")
        
        sys.exit(1)


if __name__ == "__main__":
    main()
