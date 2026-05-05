#!/usr/bin/env python3
"""
LLM 性能监控 (Observability & Performance)

核心指标：
- TTFT (Time To First Token): 首Token时间
- TPOT (Time Per Output Token): 每Token生成时间
- 端到端延迟 (End-to-End Latency)
- 吞吐量 (Throughput)

输出：performance_metrics.csv
"""

import json
import time
import csv
import subprocess
import requests
from typing import Dict, List, Any
from dataclasses import dataclass
from datetime import datetime
import statistics


@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    timestamp: str
    model_name: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    ttft_ms: float  # Time To First Token (毫秒)
    tpot_ms: float  # Time Per Output Token (毫秒)
    latency_ms: float  # 端到端延迟 (毫秒)
    throughput_tps: float  # 吞吐量 (tokens/second)
    input_length: int
    output_length: int
    status: str
    error_message: str = ""


class LLMPerformanceMonitor:
    """LLM性能监控器"""
    
    def __init__(self, api_url: str = "http://localhost:11434/api/generate"):
        self.api_url = api_url
        self.metrics_history: List[PerformanceMetrics] = []
        
    def call_model_streaming(self, model: str, prompt: str) -> Dict[str, Any]:
        """调用模型并测量详细性能指标"""
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": {"temperature": 0.3}
        }
        
        try:
            start_time = time.perf_counter()
            first_token_time = None
            tokens_received = 0
            full_response = ""
            
            response = requests.post(
                self.api_url,
                json=payload,
                stream=True,
                timeout=300
            )
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        if first_token_time is None and data.get("response"):
                            first_token_time = time.perf_counter()
                        if data.get("response"):
                            tokens_received += 1
                            full_response += data.get("response", "")
                        if data.get("done"):
                            break
                    except json.JSONDecodeError:
                        continue
            
            end_time = time.perf_counter()
            
            total_duration = (end_time - start_time) * 1000
            ttft = (first_token_time - start_time) * 1000 if first_token_time else total_duration
            
            prompt_tokens = len(prompt) // 4
            completion_tokens = len(full_response) // 4
            
            if tokens_received > 1:
                generation_time = (end_time - first_token_time) * 1000
                tpot = generation_time / (tokens_received - 1)
            else:
                tpot = 0
            
            throughput = (completion_tokens / total_duration) * 1000 if total_duration > 0 else 0
            
            return {
                "success": True,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "ttft_ms": ttft,
                "tpot_ms": tpot,
                "latency_ms": total_duration,
                "throughput_tps": throughput,
                "input_length": len(prompt),
                "output_length": len(full_response)
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def benchmark_single(self, model: str, prompt: str, test_name: str = "") -> PerformanceMetrics:
        """对单个请求进行性能基准测试"""
        print(f"  测试 [{test_name or '未命名'}]...", end=" ")
        
        result = self.call_model_streaming(model, prompt)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if result.get("success"):
            metrics = PerformanceMetrics(
                timestamp=timestamp,
                model_name=model,
                prompt_tokens=result.get("prompt_tokens", 0),
                completion_tokens=result.get("completion_tokens", 0),
                total_tokens=result.get("total_tokens", 0),
                ttft_ms=result.get("ttft_ms", 0),
                tpot_ms=result.get("tpot_ms", 0),
                latency_ms=result.get("latency_ms", 0),
                throughput_tps=result.get("throughput_tps", 0),
                input_length=result.get("input_length", 0),
                output_length=result.get("output_length", 0),
                status="SUCCESS"
            )
            print(f"✓ TTFT={metrics.ttft_ms:.1f}ms, TPOT={metrics.tpot_ms:.1f}ms")
        else:
            metrics = PerformanceMetrics(
                timestamp=timestamp,
                model_name=model,
                prompt_tokens=0, completion_tokens=0, total_tokens=0,
                ttft_ms=0, tpot_ms=0, latency_ms=0, throughput_tps=0,
                input_length=len(prompt), output_length=0,
                status="FAILED",
                error_message=result.get("error", "Unknown error")[:50]
            )
            print(f"✗ 失败")
        
        self.metrics_history.append(metrics)
        return metrics
    
    def benchmark_model(self, model: str, test_prompts: List[Dict[str, str]], warmup: int = 1) -> Dict[str, Any]:
        """对模型进行全面基准测试"""
        print(f"\n{'='*60}")
        print(f"性能基准测试: {model}")
        print(f"{'='*60}")
        
        if warmup > 0:
            print(f"\n[预热] {warmup} 次...")
            for _ in range(warmup):
                self.call_model_streaming(model, "Hello")
                time.sleep(0.5)
        
        print(f"\n[正式测试] {len(test_prompts)} 个用例\n")
        
        results = []
        for test in test_prompts:
            metrics = self.benchmark_single(model, test["prompt"], test.get("name", ""))
            results.append(metrics)
            time.sleep(1)
        
        # 统计
        success_results = [r for r in results if r.status == "SUCCESS"]
        if success_results:
            summary = {
                "model": model,
                "total_tests": len(results),
                "success_rate": len(success_results) / len(results) * 100,
                "avg_ttft_ms": statistics.mean([r.ttft_ms for r in success_results]),
                "avg_tpot_ms": statistics.mean([r.tpot_ms for r in success_results]),
                "avg_latency_ms": statistics.mean([r.latency_ms for r in success_results]),
                "avg_throughput_tps": statistics.mean([r.throughput_tps for r in success_results]),
                "max_ttft_ms": max([r.ttft_ms for r in success_results]),
                "min_ttft_ms": min([r.ttft_ms for r in success_results])
            }
            
            print(f"\n{'='*60}")
            print(f"测试结果汇总: {model}")
            print(f"{'='*60}")
            print(f"  成功率: {summary['success_rate']:.1f}%")
            print(f"  平均TTFT: {summary['avg_ttft_ms']:.2f} ms")
            print(f"  平均TPOT: {summary['avg_tpot_ms']:.2f} ms")
            print(f"  平均延迟: {summary['avg_latency_ms']:.2f} ms")
            print(f"  平均吞吐量: {summary['avg_throughput_tps']:.2f} tokens/s")
            print(f"  TTFT范围: {summary['min_ttft_ms']:.2f} - {summary['max_ttft_ms']:.2f} ms")
        
        return {"model": model, "results": results}
    
    def save_to_csv(self, filename: str = "performance_metrics.csv"):
        """保存指标到CSV文件"""
        if not self.metrics_history:
            print("没有数据可保存")
            return
        
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'model_name', 'prompt_tokens', 'completion_tokens',
                'total_tokens', 'ttft_ms', 'tpot_ms', 'latency_ms', 'throughput_tps',
                'input_length', 'output_length', 'status', 'error_message'
            ])
            
            for m in self.metrics_history:
                writer.writerow([
                    m.timestamp, m.model_name, m.prompt_tokens, m.completion_tokens,
                    m.total_tokens, m.ttft_ms, m.tpot_ms, m.latency_ms, m.throughput_tps,
                    m.input_length, m.output_length, m.status, m.error_message
                ])
        
        print(f"\n✓ 性能指标已保存到: {filename}")


def main():
    """主函数"""
    print("="*60)
    print("LLM 性能监控工具")
    print("="*60)
    
    # 测试用例
    test_prompts = [
        {"name": "短文本", "prompt": "什么是Python？"},
        {"name": "中等文本", "prompt": "解释Python中的列表推导式，并给出3个使用示例。"},
        {"name": "长文本", "prompt": "详细解释Python的GIL（全局解释器锁）机制，包括它的工作原理、优缺点以及如何在多线程编程中处理它。"}
    ]
    
    monitor = LLMPerformanceMonitor()
    
    # 测试9b模型
    monitor.benchmark_model("qwen3.5:9b", test_prompts, warmup=1)
    
    # 测试35b模型（带显存管理）
    print("\n" + "-"*60)
    print("\n准备测试35b模型...")
    
    # 先停止9b释放显存
    try:
        import subprocess
        print("  🔄 停止9b模型，释放显存...")
        subprocess.run(["ollama", "stop", "qwen3.5:9b"], capture_output=True, timeout=10)
        time.sleep(3)
    except Exception:
        pass
    
    # 尝试加载35b
    try:
        print("  🔄 尝试加载35b模型...")
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "qwen3.5:35b-a3b",
                "prompt": "hi",
                "stream": False,
                "options": {"num_predict": 5}
            },
            timeout=60
        )
        
        if response.status_code == 200:
            print("  ✅ 35b模型加载成功，开始测试...")
            monitor.benchmark_model("qwen3.5:35b-a3b", test_prompts, warmup=1)
        else:
            print(f"  ❌ 35b模型加载失败 (HTTP {response.status_code})")
            print("     可能原因: 显存不足(需要约22GB)")
    except Exception as e:
        print(f"  ❌ 35b模型不可用: {str(e)[:80]}")
        print("     提示: 请确保已下载模型 'ollama pull qwen3.5:35b-a3b'")
    
    # 保存结果
    monitor.save_to_csv("performance_metrics.csv")
    
    # 程序结束前停止所有模型
    print("\n  🧹 清理显存，停止所有模型...")
    for model in ["qwen3.5:9b", "qwen3.5:35b-a3b"]:
        try:
            subprocess.run(["ollama", "stop", model], capture_output=True, timeout=10)
        except Exception:
            pass
    time.sleep(2)
    
    print("\n" + "="*60)
    print("性能监控完成")
    print("="*60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ 用户中断，正在清理...")
        for model in ["qwen3.5:9b", "qwen3.5:35b-a3b"]:
            try:
                subprocess.run(["ollama", "stop", model], capture_output=True, timeout=10)
            except Exception:
                pass
        print("✓ 已停止所有模型")
    except Exception as e:
        print(f"\n\n❌ 程序异常: {e}")
        for model in ["qwen3.5:9b", "qwen3.5:35b-a3b"]:
            try:
                subprocess.run(["ollama", "stop", model], capture_output=True, timeout=10)
            except Exception:
                pass
        print("✓ 已停止所有模型")
