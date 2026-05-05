#!/usr/bin/env python3
"""
LLM 质量评估 (Observability & Quality)

多维度评估：
- 回答相关性 (Relevance)
- 事实准确性 (Accuracy)
- 回复完整性 (Completeness)
- 语言流畅性 (Fluency)

复用 LLM-as-a-Judge 逻辑
输出：quality_metrics.csv
"""

import json
import time
import csv
import subprocess
import requests
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from datetime import datetime


@dataclass
class QualityMetrics:
    """质量评估指标数据类"""
    timestamp: str
    model_name: str
    question: str
    reference_answer: str
    model_answer: str
    
    # 质量评分 (0-10分)
    relevance_score: float  # 回答相关性
    accuracy_score: float   # 事实准确性
    completeness_score: float  # 回复完整性
    fluency_score: float    # 语言流畅性
    
    total_score: float      # 总分
    judge_model: str        # 裁判模型
    evaluation_feedback: str  # 评估反馈
    status: str             # 评估状态


class LLMQualityEvaluator:
    """LLM质量评估器 - 使用LLM-as-a-Judge模式"""
    
    def __init__(
        self,
        judge_model: str = "qwen3.5:9b",
        api_url: str = "http://localhost:11434/api/generate"
    ):
        self.judge_model = judge_model
        self.api_url = api_url
        self.evaluation_history: List[QualityMetrics] = []
    
    def _extract_scores_from_text(self, text: str) -> Dict[str, float]:
        """从非JSON文本中提取评分"""
        scores = {}
        
        # 匹配模式：相关性: 8.5 或 相关性 8.5 或 Relevance: 8.5
        patterns = [
            (r'相关性[:\s]*(\d+(?:\.\d+)?)', '相关性'),
            (r'准确性[:\s]*(\d+(?:\.\d+)?)', '准确性'),
            (r'完整性[:\s]*(\d+(?:\.\d+)?)', '完整性'),
            (r'流畅性[:\s]*(\d+(?:\.\d+)?)', '流畅性'),
            (r'[Rr]elevance[:\s]*(\d+(?:\.\d+)?)', '相关性'),
            (r'[Aa]ccuracy[:\s]*(\d+(?:\.\d+)?)', '准确性'),
            (r'[Cc]ompleteness[:\s]*(\d+(?:\.\d+)?)', '完整性'),
            (r'[Ff]luency[:\s]*(\d+(?:\.\d+)?)', '流畅性'),
        ]
        
        import re
        for pattern, key in patterns:
            match = re.search(pattern, text)
            if match and key not in scores:
                try:
                    scores[key] = float(match.group(1))
                except:
                    pass
        
        return scores

    def call_model(self, model: str, prompt: str, system: str = "", max_tokens: int = 800) -> str:
        """调用Ollama模型 (使用Generate API，与llm_judge.py一致)"""
        gen_url = "http://localhost:11434/api/generate"
        
        payload = {
            "model": model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "num_predict": max_tokens
            }
        }
        
        try:
            response = requests.post(
                gen_url,
                json=payload,
                timeout=300
            )
            response.raise_for_status()
            result = response.json()
            
            # 获取 response 字段
            resp_text = result.get("response", "").strip()
            
            # 如果 response 为空，从 thinking 提取
            if not resp_text and "thinking" in result:
                thinking = result.get("thinking", "")
                # Qwen thinking 格式通常在 "Thinking Process:" 后
                # 实际回答在最后，找 "Response:" 或取最后一段
                for marker in ["Response:", "Final Answer:", "Answer:", "回复："]:
                    if marker in thinking:
                        parts = thinking.split(marker)
                        if len(parts) > 1:
                            resp_text = parts[-1].strip()
                            break
                
                # 如果没有找到标记，尝试从最后非空行提取
                if not resp_text:
                    lines = thinking.split('\n')
                    for line in reversed(lines):
                        line = line.strip()
                        if line and not line.startswith('Thinking') and not line.startswith('*') and not line.startswith('1.'):
                            resp_text = line
                            break
                
                # 如果仍然没有，用 thinking 内容（截取）
                if not resp_text:
                    resp_text = thinking.strip()
            
            return resp_text
        except Exception as e:
            return f"[错误] 调用失败: {str(e)}"
    
    def get_model_answer(self, model: str, question: str) -> str:
        """获取被测模型的回答"""
        print(f"    [{model}] 生成回答...", end=" ")
        
        system_prompt = "你是一个专业的助手。请简洁准确地回答问题，控制在200字以内。"
        answer = ""  # 初始化变量，避免未绑定警告
        
        # 尝试多次获取回答
        for attempt in range(2):
            answer = self.call_model(model, question, system_prompt, max_tokens=500)
            
            if answer and not answer.startswith("[错误]"):
                print(f"OK ({len(answer)} 字符)")
                return answer
            
            if attempt == 0:
                print("重试...", end=" ")
                time.sleep(1)
        
        print(f"FAIL")
        return answer if answer else "[错误] 模型未返回有效回答"
    
    def evaluate_answer(
        self,
        target_model: str,
        question: str,
        reference_answer: str,
        model_answer: str
    ) -> QualityMetrics:
        """使用裁判模型评估回答质量"""
        print(f"    [{self.judge_model}] 质量评估...", end=" ")
        
        prompt = f"""作为质量评估专家，请对以下回答进行多维度评分。

问题: {question}
参考答案: {reference_answer}
模型回答: {model_answer}

请从4个维度评分(0-10分):
1. 相关性: 回答是否紧扣问题
2. 准确性: 内容是否正确，有无事实错误
3. 完整性: 是否涵盖问题的关键要点
4. 流畅性: 语言表达是否通顺自然

返回JSON格式:
{{"scores": [{{"dimension": "相关性", "score": 8.5}}, {{"dimension": "准确性", "score": 7.0}}, {{"dimension": "完整性", "score": 8.0}}, {{"dimension": "流畅性", "score": 9.0}}], "total_score": 8.1, "feedback": "改进建议"}}

只返回JSON:"""

        system_prompt = "你是质量评估专家。必须只返回JSON格式，不要任何其他文字。"
        
        response = self.call_model(self.judge_model, prompt, system_prompt, max_tokens=600)
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 解析JSON
        try:
            # 清理响应
            response_clean = response.strip()
            if response_clean.startswith('"') and response_clean.endswith('"'):
                response_clean = response_clean[1:-1]
            
            # 提取JSON
            if "```json" in response_clean:
                json_str = response_clean.split("```json")[1].split("```")[0]
            elif "```" in response_clean:
                json_str = response_clean.split("```")[1].split("```")[0]
            else:
                start_idx = response_clean.find('{')
                end_idx = response_clean.rfind('}')
                if start_idx != -1 and end_idx != -1:
                    json_str = response_clean[start_idx:end_idx+1]
                else:
                    json_str = response_clean
            
            result = json.loads(json_str.strip())
            
            scores = {item["dimension"]: float(item["score"]) for item in result.get("scores", [])}
            total_score = result.get("total_score", sum(scores.values()) / len(scores) if scores else 0)
            feedback = result.get("feedback", "")
            
            metrics = QualityMetrics(
                timestamp=timestamp,
                model_name=target_model,
                question=question,
                reference_answer=reference_answer,
                model_answer=model_answer[:200] + "..." if len(model_answer) > 200 else model_answer,
                relevance_score=scores.get("相关性", 0),
                accuracy_score=scores.get("准确性", 0),
                completeness_score=scores.get("完整性", 0),
                fluency_score=scores.get("流畅性", 0),
                total_score=float(total_score),
                judge_model=self.judge_model,
                evaluation_feedback=feedback,
                status="SUCCESS"
            )
            print(f"OK 总分={metrics.total_score:.1f}")
            
        except Exception as e:
            # JSON 解析失败，尝试从文本提取评分
            scores = self._extract_scores_from_text(response)
            if scores:
                total_score = sum(scores.values()) / len(scores)
                metrics = QualityMetrics(
                    timestamp=timestamp,
                    model_name=target_model,
                    question=question,
                    reference_answer=reference_answer,
                    model_answer=model_answer[:200] + "..." if len(model_answer) > 200 else model_answer,
                    relevance_score=scores.get("相关性", scores.get("relevance", 0)),
                    accuracy_score=scores.get("准确性", scores.get("accuracy", 0)),
                    completeness_score=scores.get("完整性", scores.get("completeness", 0)),
                    fluency_score=scores.get("流畅性", scores.get("fluency", 0)),
                    total_score=total_score,
                    judge_model=self.judge_model,
                    evaluation_feedback="从文本提取",
                    status="SUCCESS"
                )
                print(f"OK 总分={metrics.total_score:.1f} (文本提取)")
            else:
                print(f"FAIL 解析失败: {str(e)[:30]}")
                metrics = QualityMetrics(
                    timestamp=timestamp,
                    model_name=target_model,
                    question=question,
                    reference_answer=reference_answer,
                    model_answer=model_answer[:100],
                    relevance_score=0,
                    accuracy_score=0,
                    completeness_score=0,
                    fluency_score=0,
                    total_score=0,
                    judge_model=self.judge_model,
                    evaluation_feedback=f"解析失败: {str(e)[:50]}",
                    status="FAILED"
                )
        
        self.evaluation_history.append(metrics)
        return metrics
    
    def warmup_model(self, model: str):
        """预热模型，确保它已加载到显存"""
        print(f"  [预热] 模型 {model}...", end=" ")
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": model,
                    "prompt": "你好",
                    "stream": False,
                    "options": {"num_predict": 10}
                },
                timeout=120
            )
            if response.status_code == 200:
                print("OK")
                return True
            else:
                print(f"FAIL (HTTP {response.status_code})")
                return False
        except Exception as e:
            print(f"FAIL ({str(e)[:30]})")
            return False

    def evaluate_model(
        self,
        target_model: str,
        test_cases: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """对模型进行全面质量评估"""
        print(f"\n{'='*60}")
        print(f"质量评估: {target_model}")
        print(f"裁判模型: {self.judge_model}")
        print(f"{'='*60}")
        
        # 预热被测模型
        if not self.warmup_model(target_model):
            print(f"[WARN] 模型 {target_model} 预热失败，跳过测试")
            return {"model": target_model, "results": []}
        
        results = []
        
        for i, test in enumerate(test_cases, 1):
            print(f"\n[{i}/{len(test_cases)}] {test['question'][:40]}...")
            
            # 获取模型回答
            answer = self.get_model_answer(target_model, test["question"])
            
            # 评估质量
            metrics = self.evaluate_answer(
                target_model,
                test["question"],
                test["reference"],
                answer
            )
            results.append(metrics)
            
            time.sleep(1)
        
        # 统计汇总
        success_results = [r for r in results if r.status == "SUCCESS"]
        if success_results:
            summary = {
                "model": target_model,
                "total_tests": len(test_cases),
                "success_rate": len(success_results) / len(test_cases) * 100,
                "avg_relevance": sum(r.relevance_score for r in success_results) / len(success_results),
                "avg_accuracy": sum(r.accuracy_score for r in success_results) / len(success_results),
                "avg_completeness": sum(r.completeness_score for r in success_results) / len(success_results),
                "avg_fluency": sum(r.fluency_score for r in success_results) / len(success_results),
                "avg_total": sum(r.total_score for r in success_results) / len(success_results)
            }
            
            print(f"\n{'='*60}")
            print(f"质量评估汇总: {target_model}")
            print(f"{'='*60}")
            print(f"  成功率: {summary['success_rate']:.1f}%")
            print(f"  平均相关性: {summary['avg_relevance']:.2f}/10")
            print(f"  平均准确性: {summary['avg_accuracy']:.2f}/10")
            print(f"  平均完整性: {summary['avg_completeness']:.2f}/10")
            print(f"  平均流畅性: {summary['avg_fluency']:.2f}/10")
            print(f"  综合得分: {summary['avg_total']:.2f}/10")
        
        return {"model": target_model, "results": results}
    
    def save_to_csv(self, filename: str = "quality_metrics.csv"):
        """保存质量评估结果到CSV"""
        if not self.evaluation_history:
            print("没有数据可保存")
            return
        
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'model_name', 'question', 'reference_answer', 'model_answer',
                'relevance_score', 'accuracy_score', 'completeness_score', 'fluency_score',
                'total_score', 'judge_model', 'evaluation_feedback', 'status'
            ])
            
            for m in self.evaluation_history:
                writer.writerow([
                    m.timestamp, m.model_name, m.question, m.reference_answer, m.model_answer,
                    m.relevance_score, m.accuracy_score, m.completeness_score, m.fluency_score,
                    m.total_score, m.judge_model, m.evaluation_feedback, m.status
                ])
        
        print(f"\n[OK] 质量评估结果已保存到: {filename}")


def stop_model(model_name: str):
    """停止模型释放显存"""
    try:
        subprocess.run(
            ["ollama", "stop", model_name],
            capture_output=True,
            encoding='utf-8',
            errors='ignore',
            timeout=10
        )
    except Exception:
        pass


def stop_all_models():
    """停止所有模型"""
    print("  [清理] 清理显存，停止所有模型...")
    for model in ["qwen3.5:9b", "qwen3.5:35b-a3b"]:
        stop_model(model)
    time.sleep(2)


def main():
    """主函数"""
    print("="*60)
    print("LLM 质量评估工具 (LLM-as-a-Judge)")
    print("="*60)
    
    # 测试用例
    test_cases = [
        {
            "question": "什么是机器学习？",
            "reference": "机器学习是人工智能的一个分支，通过算法让计算机从数据中学习规律。"
        },
        {
            "question": "Python的装饰器是什么？",
            "reference": "装饰器是用@语法修饰函数的设计模式，可在不修改原函数的情况下添加功能。"
        },
        {
            "question": "如何优化数据库查询？",
            "reference": "添加索引、优化SQL语句、使用缓存、分表分库等方法。"
        }
    ]
    
    evaluator = LLMQualityEvaluator(judge_model="qwen3.5:9b")
    
    # 测试9b模型
    evaluator.evaluate_model("qwen3.5:9b", test_cases)
    
    # 测试35b模型（带显存管理）
    print("\n" + "-"*60)
    print("\n准备测试35b模型...")
    
    # 停止9b释放显存
    print("  [切换] 停止9b模型...")
    stop_model("qwen3.5:9b")
    time.sleep(3)
    
    # 尝试加载35b
    try:
        print("  [切换] 尝试加载35b模型...")
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
            print("  [OK] 35b模型加载成功，开始测试...")
            evaluator.evaluate_model("qwen3.5:35b-a3b", test_cases)
        else:
            print(f"  [FAIL] 35b模型加载失败 (HTTP {response.status_code})")
    except Exception as e:
        print(f"  [FAIL] 35b模型不可用: {str(e)[:80]}")
    
    # 保存结果
    evaluator.save_to_csv("quality_metrics.csv")
    
    # 程序结束前停止所有模型
    stop_all_models()
    
    print("\n" + "="*60)
    print("质量评估完成")
    print("="*60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[WARN] 用户中断，正在清理...")
        stop_all_models()
        print("[OK] 已停止所有模型")
    except Exception as e:
        print(f"\n\n[ERROR] 程序异常: {e}")
        stop_all_models()
        print("[OK] 已停止所有模型")
