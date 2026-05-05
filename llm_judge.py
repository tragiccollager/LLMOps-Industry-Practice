#!/usr/bin/env python3
"""
LLM Judge - 大模型交叉裁判评分系统（带安全护栏）

功能：
1. 安全护栏：敏感词检测、输入验证、越狱攻击防护
2. 交叉裁判：9b和35b模型互相评分，避免"自己评自己"
3. 显存管理：确保两个大模型不会同时运行，自动释放显存
4. 评估结果：终端实时输出 + JSON文件保存

评分维度：
- 准确性 (Accuracy): 回答内容是否正确
- 完整性 (Completeness): 是否涵盖问题的所有要点  
- 简洁性 (Conciseness): 是否简洁明了，无冗余（权重0.8）
- 相关性 (Relevance): 是否紧扣问题，无跑题

使用方法：
  python llm_judge.py              # 默认：交叉评估（9b↔35b互相裁判）
  python llm_judge.py --single     # 单模型评测（仅用于测试）

显存管理：
  - 生成回答前：停止裁判模型，为被测模型释放显存
  - 评分前：停止被测模型，为裁判模型释放显存
  - 确保任何时刻只有一个模型在显存中

输出：
  - 终端：实时显示评估报告、各维度分数、改进建议
  - 文件：llm_judge_cross_latest.json（完整详细结果）
"""

import json
import asyncio
import requests
import re
import subprocess
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime


class SafetyLevel(Enum):
    """安全等级"""
    SAFE = "安全"
    WARNING = "警告"
    DANGEROUS = "危险"
    BLOCKED = "已拦截"


@dataclass
class SafetyCheckResult:
    """安全检查结果"""
    is_safe: bool
    level: SafetyLevel
    violations: List[str]
    sanitized_input: Optional[str]


@dataclass
class TestCase:
    """测试用例"""
    question: str
    reference_answer: str
    category: str = "general"
    difficulty: str = "medium"  # easy, medium, hard


@dataclass
class ScoreResult:
    """评分结果"""
    dimension: str
    score: float
    explanation: str
    weight: float = 1.0  # 权重，用于加权总分


@dataclass
class EvaluationResult:
    """完整评估结果"""
    test_case: TestCase
    model_answer: str
    scores: List[ScoreResult]
    total_score: float
    weighted_score: float
    judge_model: str
    safety_result: SafetyCheckResult
    evaluation_time: str
    feedback: str  # 裁判的改进建议


# ==================== 显存管理 ====================

def stop_model(model_name: str):
    """停止指定模型以释放显存"""
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
    for model in ["qwen3.5:9b", "qwen3.5:35b-a3b"]:
        stop_model(model)
        time.sleep(1)  # 给一点时间释放显存


# ==================== 安全护栏配置 ====================

# 敏感词列表（中文 + 英文）
SENSITIVE_WORDS = {
    "violence": ["暴力", "杀人", "炸弹", "terrorist", "kill", "bomb"],
    "porn": ["色情", "淫秽", "porn", "sex"],
    "discrimination": ["种族歧视", "性别歧视", "歧视", "racist", "sexist"],
    "illegal": ["毒品", "黑客", "攻击", "破解", "drug", "hack", "attack"],
    "privacy": ["密码", "身份证号", "银行卡", "password", "credit card"],
}

# 输入长度限制
MAX_INPUT_LENGTH = 2000
MIN_INPUT_LENGTH = 3

# 危险提示词（用于检测越狱攻击）
JAILBREAK_PATTERNS = [
    r"忽略.*指令",
    r"忽略.*规则",
    r"你是.*现在",
    r"DAN\s*mode",
    r"jailbreak",
    r"绕过.*限制",
]


class SafetyGuardrail:
    """安全护栏 - 敏感词检测和输入验证"""
    
    def __init__(self):
        self.sensitive_words = SENSITIVE_WORDS
        self.max_length = MAX_INPUT_LENGTH
        self.min_length = MIN_INPUT_LENGTH
        self.jailbreak_patterns = [re.compile(p, re.IGNORECASE) for p in JAILBREAK_PATTERNS]
    
    def check_input(self, text: str, check_type: str = "question") -> SafetyCheckResult:
        """
        检查输入是否安全
        
        Args:
            text: 待检查的文本
            check_type: 检查类型 (question/answer)
        
        Returns:
            SafetyCheckResult: 检查结果
        """
        violations = []
        
        # 1. 长度检查
        if len(text) > self.max_length:
            violations.append(f"输入过长: {len(text)} 字符 (最大 {self.max_length})")
        
        if len(text) < self.min_length:
            violations.append(f"输入过短: {len(text)} 字符 (最小 {self.min_length})")
        
        # 2. 敏感词检测
        text_lower = text.lower()
        for category, words in self.sensitive_words.items():
            for word in words:
                if word.lower() in text_lower:
                    violations.append(f"敏感词 [{category}]: {word}")
        
        # 3. 越狱攻击检测（仅对问题检查）
        if check_type == "question":
            for pattern in self.jailbreak_patterns:
                if pattern.search(text):
                    violations.append(f"潜在越狱攻击模式: {pattern.pattern}")
        
        # 4. 特殊字符检测（防止注入）
        dangerous_chars = ["<script>", "javascript:", "onerror=", "onload="]
        for char in dangerous_chars:
            if char in text_lower:
                violations.append(f"危险字符: {char}")
        
        # 确定安全等级
        if not violations:
            return SafetyCheckResult(
                is_safe=True,
                level=SafetyLevel.SAFE,
                violations=[],
                sanitized_input=text
            )
        
        # 危险程度判断
        critical_categories = ["violence", "illegal", "privacy"]
        is_critical = any(cat in v for cat in critical_categories for v in violations)
        jailbreak_detected = any("越狱" in v for v in violations)
        
        if is_critical or jailbreak_detected or len(violations) >= 3:
            level = SafetyLevel.BLOCKED
            sanitized = None  # 危险内容，不返回净化版本
        elif len(violations) >= 2:
            level = SafetyLevel.DANGEROUS
            sanitized = self._sanitize(text)
        else:
            level = SafetyLevel.WARNING
            sanitized = self._sanitize(text)
        
        return SafetyCheckResult(
            is_safe=False,
            level=level,
            violations=violations,
            sanitized_input=sanitized
        )
    
    def _sanitize(self, text: str) -> str:
        """净化文本 - 替换敏感词"""
        sanitized = text
        for category, words in self.sensitive_words.items():
            for word in words:
                sanitized = sanitized.replace(word, "*" * len(word))
                sanitized = sanitized.replace(word.lower(), "*" * len(word))
        return sanitized
    
    def validate_question(self, question: str) -> Tuple[bool, Optional[str], SafetyCheckResult]:
        """
        验证问题是否可处理
        
        Returns:
            (是否通过, 错误信息, 检查结果)
        """
        result = self.check_input(question, "question")
        
        if result.level == SafetyLevel.BLOCKED:
            return False, f"输入被拦截: {', '.join(result.violations)}", result
        
        if result.level == SafetyLevel.DANGEROUS:
            # 危险但仍可处理，使用净化后的输入
            return True, None, result
        
        return True, None, result


# ==================== 测试数据集（精简版） ====================

TEST_DATASET: List[TestCase] = [
    TestCase(
        question="什么是机器学习？用一句话解释。",
        reference_answer="机器学习是让计算机从数据中学习规律而无需显式编程的技术。",
        category="技术概念",
        difficulty="easy"
    ),
    TestCase(
        question="Python装饰器是什么？简要说明。",
        reference_answer="装饰器是用@语法修饰函数的设计模式，可在不修改原函数的情况下添加功能。",
        category="编程",
        difficulty="medium"
    ),
    TestCase(
        question="数据库查询优化的两个关键方法？",
        reference_answer="添加索引和优化SQL语句。",
        category="数据库",
        difficulty="medium"
    ),
    TestCase(
        question="什么是RESTful API？一句话概括。",
        reference_answer="RESTful API是基于HTTP方法操作资源的接口设计规范。",
        category="Web开发",
        difficulty="easy"
    ),
    TestCase(
        question="Docker容器相比虚拟机的主要优势？",
        reference_answer="更轻量、启动更快，共享主机内核。",
        category="运维",
        difficulty="medium"
    ),
]


# ==================== LLM Judge 核心类 ====================

class LLMJudge:
    """LLM裁判评分器（带安全护栏）"""
    
    def __init__(
        self,
        judge_model: str = "qwen3.5:9b",
        target_model: str = "qwen3.5:9b",
        api_url: str = "http://localhost:11434/api/generate",
        enable_safety_guard: bool = True
    ):
        self.judge_model = judge_model
        self.target_model = target_model
        self.api_url = api_url
        self.guardrail = SafetyGuardrail() if enable_safety_guard else None
        self.evaluation_log: List[EvaluationResult] = []
    
    def call_model(self, model: str, prompt: str, system: str = "", timeout: int = 300, retries: int = 2, max_tokens: int = 800) -> str:
        """调用Ollama模型（带重试，限制生成长度）"""
        payload = {
            "model": model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "num_predict": max_tokens  # 限制最大生成token数，减少显存负担
            }
        }
        
        for attempt in range(retries):
            try:
                response = requests.post(
                    self.api_url,
                    json=payload,
                    timeout=timeout
                )
                response.raise_for_status()
                result = response.json().get("response", "")
                if result and not result.startswith("[错误]"):
                    return result
            except Exception as e:
                error_msg = str(e)
                if attempt < retries - 1:
                    print(f"    [重试 {attempt+1}/{retries}] 调用失败: {error_msg[:50]}...")
                    time.sleep(3)  # 等待后重试
                else:
                    return f"[错误] 调用模型 {model} 失败: {error_msg[:100]}"
        
        return f"[错误] 调用模型 {model} 失败，已重试{retries}次"
    
    def get_model_answer(self, question: str) -> Tuple[str, SafetyCheckResult]:
        """获取被测模型的回答（带安全检查，带显存管理）"""
        print(f"  正在获取 [{self.target_model}] 的回答...")
        
        # 先停止裁判模型，为被测模型释放显存（只有不同模型时才切换）
        if self.judge_model != self.target_model:
            print(f"  🔄 释放 [{self.judge_model}] 显存...")
            stop_model(self.judge_model)
            time.sleep(5)  # 增加等待时间到5秒
            print(f"  ⏳ 等待 [{self.target_model}] 加载...")
        else:
            print(f"  ℹ️  裁判和被测是同一模型，无需切换显存")
        
        # 检查问题安全
        safety_result = SafetyCheckResult(True, SafetyLevel.SAFE, [], question)
        if self.guardrail:
            is_valid, error_msg, check_result = self.guardrail.validate_question(question)
            if not is_valid:
                print(f"  ⚠️ 安全检查未通过: {error_msg}")
                return f"[被拦截] {error_msg}", check_result
            
            if check_result.level in [SafetyLevel.WARNING, SafetyLevel.DANGEROUS]:
                print(f"  ⚠️ 检测到潜在风险: {', '.join(check_result.violations)}")
                question = check_result.sanitized_input or question
            safety_result = check_result
        
        # 系统提示：要求简洁回答，控制长度
        system_prompt = "你是一个专业的助手。请简洁准确地回答问题，控制在300字以内，突出重点即可。"
        
        answer = self.call_model(
            self.target_model,
            question,
            system_prompt,
            max_tokens=500  # 限制生成长度，减轻显存负担
        )
        
        # 检查回答安全（只检查敏感词，不检查长度）
        if self.guardrail:
            # 简化版安全检查 - 只检查敏感词，不检查长度
            has_sensitive = False
            for category, words in self.guardrail.sensitive_words.items():
                for word in words:
                    if word.lower() in answer.lower():
                        has_sensitive = True
                        break
            if has_sensitive:
                print(f"  ⚠️ 回答包含敏感词，已净化")
                answer = self.guardrail._sanitize(answer)
        
        print(f"  回答长度: {len(answer)} 字符")
        
        # 完成后停止被测模型，为裁判模型释放显存
        if self.judge_model != self.target_model:
            print(f"  🔄 释放 [{self.target_model}] 显存...")
            stop_model(self.target_model)
            time.sleep(2)
        
        return answer, safety_result
    
    def judge_answer(
        self,
        test_case: TestCase,
        model_answer: str
    ) -> Tuple[List[ScoreResult], str]:
        """
        裁判模型评分（带显存管理）
        
        Returns:
            (评分列表, 改进建议)
        """
        print(f"  [{self.judge_model}] 正在进行评分...")
        
        # 加载裁判模型前，确保被测模型已停止（只有不同模型时才切换）
        if self.judge_model != self.target_model:
            print(f"  🔄 释放 [{self.target_model}] 显存，加载 [{self.judge_model}]...")
            stop_model(self.target_model)
            time.sleep(5)  # 增加等待时间到5秒
            print(f"  ⏳ 等待 {self.judge_model} 加载...")
        else:
            print(f"  ℹ️  使用同一模型进行自评，无需切换")
        
        # 截断过长的回答，避免prompt太大
        max_answer_len = 1500
        if len(model_answer) > max_answer_len:
            model_answer = model_answer[:max_answer_len] + "..."
        
        # 简化版提示，更容易被模型理解和返回
        prompt = f"""请对以下回答进行评分。

问题: {test_case.question}
参考答案: {test_case.reference_answer}
模型回答: {model_answer}

请从4个维度评分(0-10分):
1. 准确性: 内容是否正确
2. 完整性: 是否涵盖要点  
3. 简洁性: 是否简洁(权重0.8)
4. 相关性: 是否紧扣问题

必须返回以下格式的JSON:
{{"scores": [{{"dimension": "准确性", "score": 8.5, "explanation": "基本正确", "weight": 1.0}}, {{"dimension": "完整性", "score": 7.0, "explanation": "涵盖主要点", "weight": 1.0}}, {{"dimension": "简洁性", "score": 8.0, "explanation": "较简洁", "weight": 0.8}}, {{"dimension": "相关性", "score": 9.0, "explanation": "很相关", "weight": 1.0}}], "feedback": "改进建议"}}

只返回JSON:"""
        
        system_prompt = "你是裁判。必须只返回JSON格式，不要任何其他文字。评分0-10分。"
        
        response = self.call_model(
            self.judge_model, 
            prompt, 
            system_prompt,
            max_tokens=800  # 给裁判模型更多空间
        )
        
        # 调试输出
        if not response or response.startswith("["):
            print(f"    [调试] 裁判模型返回异常: {response[:100] if response else '空'}")
        
        # 解析JSON
        try:
            # 清理响应
            response_clean = response.strip()
            
            # 如果响应被引号包裹，去掉外层引号
            if response_clean.startswith('"') and response_clean.endswith('"'):
                response_clean = response_clean[1:-1]
                response_clean = response_clean.replace('\\n', '\n').replace('\\"', '"')
            
            # 提取JSON
            if "```json" in response_clean:
                json_str = response_clean.split("```json")[1].split("```")[0]
            elif "```" in response_clean:
                json_str = response_clean.split("```")[1].split("```")[0]
            else:
                # 尝试直接找JSON对象
                start_idx = response_clean.find('{')
                end_idx = response_clean.rfind('}')
                if start_idx != -1 and end_idx != -1:
                    json_str = response_clean[start_idx:end_idx+1]
                else:
                    json_str = response_clean
            
            json_str = json_str.strip()
            result = json.loads(json_str)
            
            scores = []
            for item in result.get("scores", []):
                scores.append(ScoreResult(
                    dimension=item.get("dimension", ""),
                    score=float(item.get("score", 0)),
                    explanation=item.get("explanation", ""),
                    weight=float(item.get("weight", 1.0))
                ))
            
            feedback = result.get("feedback", "")
            return scores, feedback
            
        except Exception as e:
            print(f"    [警告] 解析评分结果失败: {e}")
            print(f"    [调试] 原始响应前200字符: {response[:200] if response else '空'}")
            # 返回默认评分
            return [
                ScoreResult("准确性", 5.0, f"解析失败: {str(e)[:30]}", 1.0),
                ScoreResult("完整性", 5.0, "解析失败，默认评分", 1.0),
                ScoreResult("简洁性", 5.0, "解析失败，默认评分", 0.8),
                ScoreResult("相关性", 5.0, "解析失败，默认评分", 1.0),
            ], "解析失败，无法提供改进建议"
    
    def evaluate_single(self, test_case: TestCase) -> EvaluationResult:
        """评估单个测试用例"""
        print(f"\n{'='*60}")
        print(f"问题: {test_case.question}")
        print(f"类别: {test_case.category} | 难度: {test_case.difficulty}")
        print(f"{'='*60}")
        
        start_time = datetime.now()
        
        # 1. 获取被测模型回答（带安全检查）
        model_answer, safety_result = self.get_model_answer(test_case.question)
        
        # 2. 裁判模型评分
        scores, feedback = self.judge_answer(test_case, model_answer)
        
        # 计算总分
        total_score = sum(s.score for s in scores) / len(scores) if scores else 0
        weighted_score = sum(s.score * s.weight for s in scores) / sum(s.weight for s in scores) if scores else 0
        
        end_time = datetime.now()
        eval_time = str(end_time - start_time)
        
        # 显示评分结果
        print(f"\n  评分结果:")
        for s in scores:
            print(f"    {s.dimension} (权重{s.weight}): {s.score}/10 - {s.explanation}")
        print(f"    总分: {total_score:.1f}/10 | 加权总分: {weighted_score:.1f}/10")
        print(f"    安全状态: {safety_result.level.value}")
        
        result = EvaluationResult(
            test_case=test_case,
            model_answer=model_answer,
            scores=scores,
            total_score=total_score,
            weighted_score=weighted_score,
            judge_model=self.judge_model,
            safety_result=safety_result,
            evaluation_time=eval_time,
            feedback=feedback
        )
        
        self.evaluation_log.append(result)
        return result
    
    def evaluate_all(self) -> Dict[str, Any]:
        """评估所有测试用例"""
        print("\n" + "="*70)
        print("LLM Judge - 大模型自动评分系统（带安全护栏）")
        print("="*70)
        print(f"\n配置:")
        print(f"  被测模型: {self.target_model}")
        print(f"  裁判模型: {self.judge_model}")
        print(f"  测试用例数: {len(TEST_DATASET)}")
        print(f"  安全护栏: {'启用' if self.guardrail else '禁用'}")
        
        results = []
        category_scores: Dict[str, List[float]] = {}
        difficulty_scores: Dict[str, List[float]] = {}
        safety_stats = {"SAFE": 0, "WARNING": 0, "DANGEROUS": 0, "BLOCKED": 0}
        
        for test_case in TEST_DATASET:
            result = self.evaluate_single(test_case)
            results.append(result)
            
            # 按类别统计
            cat = test_case.category
            if cat not in category_scores:
                category_scores[cat] = []
            category_scores[cat].append(result.weighted_score)
            
            # 按难度统计
            diff = test_case.difficulty
            if diff not in difficulty_scores:
                difficulty_scores[diff] = []
            difficulty_scores[diff].append(result.weighted_score)
            
            # 安全统计
            safety_stats[result.safety_result.level.name] += 1
        
        # 汇总报告
        print("\n" + "="*70)
        print("评估报告汇总")
        print("="*70)
        
        # 整体统计
        all_scores = [r.weighted_score for r in results]
        avg_score = sum(all_scores) / len(all_scores) if all_scores else 0
        
        print(f"\n【整体表现】")
        print(f"  平均加权总分: {avg_score:.2f}/10")
        print(f"  最高分: {max(all_scores):.2f}")
        print(f"  最低分: {min(all_scores):.2f}")
        print(f"  标准差: {self._calc_std(all_scores):.2f}")
        
        # 安全统计
        print(f"\n【安全检查统计】")
        for level, count in safety_stats.items():
            print(f"  {level}: {count} 次")
        
        # 分类别统计
        print(f"\n【分类别表现】")
        for cat, scores in category_scores.items():
            cat_avg = sum(scores) / len(scores)
            print(f"  {cat}: {cat_avg:.2f}/10 (共{len(scores)}题)")
        
        # 按难度统计
        print(f"\n【按难度表现】")
        for diff in ["easy", "medium", "hard"]:
            if diff in difficulty_scores:
                diff_avg = sum(difficulty_scores[diff]) / len(difficulty_scores[diff])
                print(f"  {diff}: {diff_avg:.2f}/10 (共{len(difficulty_scores[diff])}题)")
        
        # 各维度平均分
        print(f"\n【各维度平均分】")
        dimension_totals: Dict[str, List[float]] = {}
        for r in results:
            for s in r.scores:
                if s.dimension not in dimension_totals:
                    dimension_totals[s.dimension] = []
                dimension_totals[s.dimension].append(s.score)
        
        for dim, scores in dimension_totals.items():
            dim_avg = sum(scores) / len(scores)
            print(f"  {dim}: {dim_avg:.2f}/10")
        
        # 详细结果
        print(f"\n【详细评分】")
        for i, r in enumerate(results, 1):
            print(f"\n  [{i}] {r.test_case.question}")
            print(f"      总分: {r.total_score:.1f}/10 | 加权: {r.weighted_score:.1f}/10 | 安全: {r.safety_result.level.value}")
        
        # 保存完整结果
        return self._save_results(results, avg_score, category_scores, dimension_totals, safety_stats)
    
    def _calc_std(self, scores: List[float]) -> float:
        """计算标准差"""
        if len(scores) < 2:
            return 0.0
        mean = sum(scores) / len(scores)
        variance = sum((x - mean) ** 2 for x in scores) / len(scores)
        return variance ** 0.5
    
    def _save_results(
        self,
        results: List[EvaluationResult],
        avg_score: float,
        category_scores: Dict[str, List[float]],
        dimension_totals: Dict[str, List[float]],
        safety_stats: Dict[str, int]
    ) -> Dict[str, Any]:
        """保存完整的评估结果"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        output = {
            "metadata": {
                "timestamp": timestamp,
                "config": {
                    "target_model": self.target_model,
                    "judge_model": self.judge_model,
                    "total_cases": len(TEST_DATASET),
                    "safety_guard_enabled": self.guardrail is not None
                }
            },
            "summary": {
                "average_score": round(avg_score, 2),
                "max_score": max([r.weighted_score for r in results]),
                "min_score": min([r.weighted_score for r in results]),
                "std_deviation": round(self._calc_std([r.weighted_score for r in results]), 2),
                "category_avg": {k: round(sum(v)/len(v), 2) for k, v in category_scores.items()},
                "dimension_avg": {k: round(sum(v)/len(v), 2) for k, v in dimension_totals.items()},
                "safety_statistics": safety_stats
            },
            "detailed_results": [
                {
                    "question_id": i + 1,
                    "question": r.test_case.question,
                    "category": r.test_case.category,
                    "difficulty": r.test_case.difficulty,
                    "reference_answer": r.test_case.reference_answer,
                    "model_answer": r.model_answer,
                    "total_score": r.total_score,
                    "weighted_score": r.weighted_score,
                    "scores": [
                        {
                            "dimension": s.dimension,
                            "score": s.score,
                            "weight": s.weight,
                            "explanation": s.explanation
                        }
                        for s in r.scores
                    ],
                    "safety_check": {
                        "is_safe": r.safety_result.is_safe,
                        "level": r.safety_result.level.value,
                        "violations": r.safety_result.violations,
                        "sanitized_input": r.safety_result.sanitized_input
                    },
                    "feedback": r.feedback,
                    "evaluation_time": r.evaluation_time
                }
                for i, r in enumerate(results)
            ]
        }
        
        # 保存主结果文件
        filename = f"llm_judge_results_{timestamp}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        
        # 同时保存最新结果
        with open("llm_judge_results_latest.json", "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        
        print(f"\n✓ 详细结果已保存到: {filename}")
        print(f"✓ 最新结果已保存到: llm_judge_results_latest.json")
        
        return output


# ==================== 双模型对比 ====================

async def compare_models():
    """对比两个模型的表现"""
    models = ["qwen3.5:9b", "qwen3.5:35b-a3b"]
    judge = "qwen3.5:9b"
    
    print("\n" + "="*70)
    print("LLM Judge - 双模型对比评测")
    print("="*70)
    
    all_results = {}
    
    for model in models:
        print(f"\n\n{'#'*70}")
        print(f"# 正在评测模型: {model}")
        print(f"{'#'*70}")
        
        evaluator = LLMJudge(
            judge_model=judge,
            target_model=model,
            enable_safety_guard=True
        )
        result = evaluator.evaluate_all()
        all_results[model] = result
    
    # 对比报告
    print("\n\n" + "="*70)
    print("双模型对比报告")
    print("="*70)
    
    for model, result in all_results.items():
        avg = result["summary"]["average_score"]
        print(f"\n  {model}:")
        print(f"    平均分: {avg:.2f}/10")
        print(f"    安全拦截: {result['summary']['safety_statistics'].get('BLOCKED', 0)} 次")
    
    # 找出表现更好的模型
    scores = {m: r["summary"]["average_score"] for m, r in all_results.items()}
    best_model = max(scores, key=scores.get)
    score_diff = scores[best_model] - scores[min(scores, key=scores.get)]
    
    print(f"\n  🏆 表现更好: {best_model}")
    print(f"     平均分: {scores[best_model]:.2f}/10")
    print(f"     领先: {score_diff:.2f} 分")
    
    # 保存对比结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_file = f"llm_judge_comparison_{timestamp}.json"
    with open(comparison_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    with open("llm_judge_comparison_latest.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ 对比结果已保存到: {comparison_file}")
    print(f"✓ 最新对比结果: llm_judge_comparison_latest.json")


def print_terminal_report(results_data: Dict[str, Any], is_comparison: bool = False):
    """在终端打印详细的评估报告"""
    
    if is_comparison:
        print("\n" + "="*70)
        print("📊 LLM JUDGE - 双模型对比评估报告")
        print("="*70)
        
        for model_name, data in results_data.items():
            print(f"\n{'─'*70}")
            print(f"🤖 模型: {model_name}")
            print(f"{'─'*70}")
            _print_single_model_report(data)
        
        # 对比总结
        print(f"\n{'='*70}")
        print("🏆 对比总结")
        print(f"{'='*70}")
        
        model_scores = {}
        for model_name, data in results_data.items():
            avg_score = data["summary"]["average_score"]
            model_scores[model_name] = avg_score
            print(f"\n  {model_name}:")
            print(f"    平均加权分: {avg_score:.2f}/10")
            print(f"    安全拦截次数: {data['summary']['safety_statistics'].get('BLOCKED', 0)}")
            print(f"    标准差: {data['summary'].get('std_deviation', 0):.2f}")
        
        # 确定获胜者
        if len(model_scores) >= 2:
            best_model = max(model_scores, key=model_scores.get)
            worst_model = min(model_scores, key=model_scores.get)
            diff = model_scores[best_model] - model_scores[worst_model]
            
            print(f"\n  🥇 最佳模型: {best_model} ({model_scores[best_model]:.2f}分)")
            print(f"  🥈 落后模型: {worst_model} ({model_scores[worst_model]:.2f}分)")
            print(f"  📈 性能差距: {diff:.2f} 分 ({diff/model_scores[worst_model]*100:.1f}%)")
            
            if diff >= 2:
                print(f"  ✅ 结论: {best_model} 明显优于 {worst_model}")
            elif diff >= 1:
                print(f"  ⚡ 结论: {best_model} 小幅领先 {worst_model}")
            else:
                print(f"  ⚖️ 结论: 两个模型表现相当")
    else:
        print("\n" + "="*70)
        print("📊 LLM JUDGE - 单模型评估报告")
        print(f"{'='*70}")
        _print_single_model_report(results_data)


def _print_single_model_report(data: Dict[str, Any]):
    """打印单个模型的详细报告"""
    summary = data["summary"]
    detailed = data["detailed_results"]
    
    # 整体统计
    print(f"\n【📈 整体表现】")
    print(f"  平均加权分: {summary['average_score']:.2f}/10")
    print(f"  最高分: {summary['max_score']:.2f} | 最低分: {summary['min_score']:.2f}")
    print(f"  标准差: {summary.get('std_deviation', 0):.2f}")
    
    # 安全统计
    print(f"\n【🛡️ 安全检查统计】")
    safety = summary.get("safety_statistics", {})
    for level, count in safety.items():
        icon = "✅" if level == "SAFE" else "⚠️" if level == "WARNING" else "🚫"
        print(f"  {icon} {level}: {count} 次")
    
    # 分类别统计
    print(f"\n【📂 分类别表现】")
    for cat, score in summary.get("category_avg", {}).items():
        bar = "█" * int(score) + "░" * (10 - int(score))
        print(f"  {cat:12s}: [{bar}] {score:.1f}/10")
    
    # 各维度平均分
    print(f"\n【📊 各维度平均分】")
    for dim, score in summary.get("dimension_avg", {}).items():
        bar = "█" * int(score) + "░" * (10 - int(score))
        print(f"  {dim:8s}: [{bar}] {score:.1f}/10")
    
    # 每个测试用例的详细结果
    print(f"\n【📝 详细评估结果】")
    for i, item in enumerate(detailed, 1):
        print(f"\n  {'─'*66}")
        print(f"  [{i}] {item['question']}")
        print(f"      类别: {item['category']} | 难度: {item['difficulty']}")
        print(f"      加权分: {item['weighted_score']:.1f}/10 | 安全: {item['safety_check']['level']}")
        
        # 各维度分数
        print(f"      维度得分:", end="")
        for s in item['scores']:
            print(f" {s['dimension']}={s['score']:.1f}", end="")
        print()
        
        # 改进建议
        if item.get('feedback'):
            print(f"      💡 改进建议: {item['feedback'][:80]}...")


def check_model_available(model_name: str) -> bool:
    """检查模型是否可用"""
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model_name,
                "prompt": "hi",
                "stream": False
            },
            timeout=30
        )
        return response.status_code == 200
    except Exception:
        return False


def run_cross_evaluation():
    """交叉评估：两个模型相互裁判，避免同时运行"""
    print("\n" + "="*70)
    print("🚀 LLM Judge - 交叉评估模式")
    print("="*70)
    
    # 检查模型可用性
    print("\n正在检查模型可用性...")
    models_available = {
        "qwen3.5:9b": check_model_available("qwen3.5:9b"),
        "qwen3.5:35b-a3b": check_model_available("qwen3.5:35b-a3b")
    }
    
    for model, available in models_available.items():
        status = "✅ 可用" if available else "❌ 不可用"
        print(f"  {model}: {status}")
    
    # 如果35b不可用，尝试预加载
    if not models_available["qwen3.5:35b-a3b"]:
        print("\n⚠️ 35b模型未加载，尝试预加载...")
        print("   这可能需要一些时间，请耐心等待...")
        
        # 先确保9b停止，释放显存
        stop_all_models()
        time.sleep(3)
        
        # 尝试加载35b（使用轻量级请求）
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "qwen3.5:35b-a3b",
                    "prompt": "hi",
                    "stream": False,
                    "options": {"num_predict": 10}  # 最小化生成
                },
                timeout=60  # 给足够时间加载
            )
            if response.status_code == 200:
                print("   ✅ 35b模型加载成功！")
                models_available["qwen3.5:35b-a3b"] = True
            else:
                print(f"   ❌ 加载失败: HTTP {response.status_code}")
        except Exception as e:
            print(f"   ❌ 加载失败: {str(e)[:100]}")
            print("   可能原因: 显存不足(需要约22GB)")
    
    # 如果35b仍然不可用，只用9b自己评自己
    if not models_available["qwen3.5:35b-a3b"]:
        print("\n⚠️ 35b模型不可用，切换到单模型评测模式")
        print("   提示: 请运行 'ollama run qwen3.5:35b-a3b' 手动加载模型")
        
        judge = LLMJudge(
            judge_model="qwen3.5:9b",
            target_model="qwen3.5:9b",
            enable_safety_guard=True
        )
        result = judge.evaluate_all()
        print_terminal_report(result, is_comparison=False)
        stop_all_models()
        return
    
    print("\n功能模块:")
    print("  ✓ 安全护栏: 敏感词检测、输入验证、越狱攻击防护")
    print("  ✓ 相互裁判: 9b和35b模型互相评分")
    print("  ✓ 显存管理: 确保不同时加载两个大模型")
    print("  ✓ 实时反馈: 终端直接输出评估结果和改进建议")
    print("\n评估配置:")
    print("  • 9b 生成回答 → 35b 担任裁判")
    print("  • 35b 生成回答 → 9b 担任裁判")
    print()
    
    # 配置交叉评估
    evaluations = [
        {"target": "qwen3.5:9b", "judge": "qwen3.5:35b-a3b", "name": "9b(被测) ← 35b(裁判)"},
        {"target": "qwen3.5:35b-a3b", "judge": "qwen3.5:9b", "name": "35b(被测) ← 9b(裁判)"},
    ]
    
    all_results = {}
    
    for config in evaluations:
        target = config["target"]
        judge = config["judge"]
        name = config["name"]
        
        print(f"\n{'#'*70}")
        print(f"# 正在评测: {name}")
        print(f"{'#'*70}")
        
        # 开始前清理显存
        stop_all_models()
        time.sleep(3)
        
        evaluator = LLMJudge(
            judge_model=judge,
            target_model=target,
            enable_safety_guard=True
        )
        result = evaluator.evaluate_all()
        all_results[name] = result
    
    # 在终端打印详细报告
    print_terminal_report(all_results, is_comparison=True)
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_file = f"llm_judge_cross_{timestamp}.json"
    with open(comparison_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    with open("llm_judge_cross_latest.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*70}")
    print("✅ 交叉评估完成！")
    print(f"📁 结果已保存到: {comparison_file}")
    print(f"📁 最新结果: llm_judge_cross_latest.json")
    print(f"{'='*70}\n")
    
    # 最后清理显存
    stop_all_models()


def main():
    """主函数"""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--single":
        # 单模型评测（自己评自己，仅用于测试）
        print("\n" + "="*70)
        print("🚀 LLM Judge - 单模型评测模式")
        print("="*70)
        print("⚠️ 注意：此模式下裁判和被测是同一个模型")
        
        judge = LLMJudge(
            judge_model="qwen3.5:9b",
            target_model="qwen3.5:9b",
            enable_safety_guard=True
        )
        result = judge.evaluate_all()
        print_terminal_report(result, is_comparison=False)
        stop_all_models()
        
    else:
        # 默认：交叉评估（相互裁判）
        run_cross_evaluation()


if __name__ == "__main__":
    main()
