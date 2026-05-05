#!/usr/bin/env python3
"""
LLM 综合仪表盘 (Observability Dashboard)

整合三个模块的数据：
- 性能指标 (performance_metrics.csv)
- 质量评估结果 (quality_metrics.csv)
- 安全审计结果 (security_audit.json)

输出：observability_report.json

使用方法：
  python observability_dashboard.py
"""

import json
import csv
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class PerformanceSummary:
    """性能指标汇总"""
    model: str
    avg_ttft: float
    avg_tpot: float
    avg_throughput: float
    total_requests: int
    success_rate: float


@dataclass
class QualitySummary:
    """质量评估汇总"""
    model: str
    avg_relevance: float
    avg_accuracy: float
    avg_completeness: float
    avg_fluency: float
    avg_total_score: float
    total_tests: int
    success_rate: float


@dataclass
class SecuritySummary:
    """安全审计汇总"""
    total_checks: int
    safe_count: int
    blocked_count: int
    risk_distribution: Dict[str, int]


@dataclass
class ModelScorecard:
    """模型评分卡"""
    model: str
    performance_score: float
    quality_score: float
    security_score: float
    overall_score: float
    grade: str


class ObservabilityDashboard:
    """综合仪表盘"""
    
    def __init__(self):
        self.performance_data: List[Dict] = []
        self.quality_data: List[Dict] = []
        self.security_data: Optional[Dict] = None
        self.report: Dict[str, Any] = {}
    
    def load_performance_metrics(self, filename: str = "performance_metrics.csv") -> bool:
        """加载性能指标"""
        if not os.path.exists(filename):
            print(f"[WARN] 性能指标文件不存在: {filename}")
            return False
        
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                self.performance_data = list(reader)
            print(f"[OK] 加载性能指标: {len(self.performance_data)} 条记录")
            return True
        except Exception as e:
            print(f"[ERROR] 加载性能指标失败: {e}")
            return False
    
    def load_quality_metrics(self, filename: str = "quality_metrics.csv") -> bool:
        """加载质量评估结果"""
        if not os.path.exists(filename):
            print(f"[WARN] 质量评估文件不存在: {filename}")
            return False
        
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                self.quality_data = list(reader)
            print(f"[OK] 加载质量评估: {len(self.quality_data)} 条记录")
            return True
        except Exception as e:
            print(f"[ERROR] 加载质量评估失败: {e}")
            return False
    
    def load_security_audit(self, filename: str = "security_audit.json") -> bool:
        """加载安全审计结果"""
        if not os.path.exists(filename):
            print(f"[WARN] 安全审计文件不存在: {filename}")
            return False
        
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                self.security_data = json.load(f)
            print(f"[OK] 加载安全审计: {self.security_data.get('total_monitored', 0)} 条记录")
            return True
        except Exception as e:
            print(f"[ERROR] 加载安全审计失败: {e}")
            return False
    
    def analyze_performance(self) -> List[PerformanceSummary]:
        """分析性能指标"""
        if not self.performance_data:
            return []
        
        model_stats: Dict[str, Dict] = {}
        
        for row in self.performance_data:
            model = row.get('model', 'unknown')
            if model not in model_stats:
                model_stats[model] = {
                    'ttft_sum': 0,
                    'tpot_sum': 0,
                    'throughput_sum': 0,
                    'count': 0,
                    'success': 0
                }
            
            try:
                model_stats[model]['ttft_sum'] += float(row.get('ttft_seconds', 0))
                model_stats[model]['tpot_sum'] += float(row.get('tpot_seconds', 0))
                model_stats[model]['throughput_sum'] += float(row.get('throughput_chars_per_sec', 0))
                model_stats[model]['count'] += 1
                if row.get('status') == 'SUCCESS':
                    model_stats[model]['success'] += 1
            except:
                pass
        
        summaries = []
        for model, stats in model_stats.items():
            if stats['count'] > 0:
                summaries.append(PerformanceSummary(
                    model=model,
                    avg_ttft=stats['ttft_sum'] / stats['count'],
                    avg_tpot=stats['tpot_sum'] / stats['count'],
                    avg_throughput=stats['throughput_sum'] / stats['count'],
                    total_requests=stats['count'],
                    success_rate=(stats['success'] / stats['count']) * 100
                ))
        
        return summaries
    
    def analyze_quality(self) -> List[QualitySummary]:
        """分析质量评估"""
        if not self.quality_data:
            return []
        
        model_stats: Dict[str, Dict] = {}
        
        for row in self.quality_data:
            model = row.get('model_name', 'unknown')
            if model not in model_stats:
                model_stats[model] = {
                    'relevance_sum': 0,
                    'accuracy_sum': 0,
                    'completeness_sum': 0,
                    'fluency_sum': 0,
                    'total_score_sum': 0,
                    'count': 0,
                    'success': 0
                }
            
            try:
                if row.get('status') == 'SUCCESS':
                    model_stats[model]['relevance_sum'] += float(row.get('relevance_score', 0))
                    model_stats[model]['accuracy_sum'] += float(row.get('accuracy_score', 0))
                    model_stats[model]['completeness_sum'] += float(row.get('completeness_score', 0))
                    model_stats[model]['fluency_sum'] += float(row.get('fluency_score', 0))
                    model_stats[model]['total_score_sum'] += float(row.get('total_score', 0))
                    model_stats[model]['success'] += 1
                model_stats[model]['count'] += 1
            except:
                pass
        
        summaries = []
        for model, stats in model_stats.items():
            if stats['count'] > 0:
                success_count = stats['success']
                summaries.append(QualitySummary(
                    model=model,
                    avg_relevance=stats['relevance_sum'] / success_count if success_count > 0 else 0,
                    avg_accuracy=stats['accuracy_sum'] / success_count if success_count > 0 else 0,
                    avg_completeness=stats['completeness_sum'] / success_count if success_count > 0 else 0,
                    avg_fluency=stats['fluency_sum'] / success_count if success_count > 0 else 0,
                    avg_total_score=stats['total_score_sum'] / success_count if success_count > 0 else 0,
                    total_tests=stats['count'],
                    success_rate=(stats['success'] / stats['count']) * 100
                ))
        
        return summaries
    
    def analyze_security(self) -> Optional[SecuritySummary]:
        """分析安全审计"""
        if not self.security_data:
            return None
        
        results = self.security_data.get('results', [])
        safe_count = sum(1 for r in results if r.get('is_safe', False))
        blocked_count = len(results) - safe_count
        
        # 统计风险分布
        risk_dist = {"安全": 0, "低风险": 0, "中风险": 0, "高风险": 0, "严重": 0}
        for r in results:
            risk = r.get('security_check', {}).get('risk_level', '安全')
            risk_dist[risk] = risk_dist.get(risk, 0) + 1
        
        return SecuritySummary(
            total_checks=len(results),
            safe_count=safe_count,
            blocked_count=blocked_count,
            risk_distribution=risk_dist
        )
    
    def calculate_scorecard(self, 
                          perf: Optional[PerformanceSummary],
                          qual: Optional[QualitySummary],
                          sec: Optional[SecuritySummary]) -> Optional[ModelScorecard]:
        """计算模型评分卡"""
        if not perf and not qual:
            return None
        
        model = perf.model if perf else (qual.model if qual else "unknown")
        
        # 性能评分 (0-100)
        perf_score = 0
        if perf:
            # TTFT 越低越好，TPOT 越低越好，throughput 越高越好
            ttft_score = max(0, 100 - perf.avg_ttft * 10)  # TTFT < 0.1s 得满分
            tpot_score = max(0, 100 - perf.avg_tpot * 100)  # TPOT < 0.01s 得满分
            throughput_score = min(100, perf.avg_throughput / 10)  # 1000 chars/s 得满分
            perf_score = (ttft_score * 0.3 + tpot_score * 0.3 + throughput_score * 0.4)
            perf_score = perf_score * (perf.success_rate / 100)  # 乘以成功率
        
        # 质量评分 (0-100)
        qual_score = 0
        if qual:
            qual_score = qual.avg_total_score * 10  # 10分制转100分制
            qual_score = qual_score * (qual.success_rate / 100)
        
        # 安全评分 (0-100)
        sec_score = 100
        if sec and sec.total_checks > 0:
            sec_score = (sec.safe_count / sec.total_checks) * 100
        
        # 综合评分 (加权)
        overall = perf_score * 0.3 + qual_score * 0.4 + sec_score * 0.3
        
        # 等级评定
        if overall >= 90:
            grade = "A+"
        elif overall >= 80:
            grade = "A"
        elif overall >= 70:
            grade = "B"
        elif overall >= 60:
            grade = "C"
        else:
            grade = "D"
        
        return ModelScorecard(
            model=model,
            performance_score=round(perf_score, 2),
            quality_score=round(qual_score, 2),
            security_score=round(sec_score, 2),
            overall_score=round(overall, 2),
            grade=grade
        )
    
    def generate_report(self) -> Dict[str, Any]:
        """生成综合报告"""
        print("\n" + "="*60)
        print("正在生成综合仪表盘报告...")
        print("="*60)
        
        # 分析各维度
        perf_summaries = self.analyze_performance()
        qual_summaries = self.analyze_quality()
        sec_summary = self.analyze_security()
        
        # 生成评分卡
        models = set()
        for p in perf_summaries:
            models.add(p.model)
        for q in qual_summaries:
            models.add(q.model)
        
        scorecards = []
        for model in models:
            perf = next((p for p in perf_summaries if p.model == model), None)
            qual = next((q for q in qual_summaries if q.model == model), None)
            card = self.calculate_scorecard(perf, qual, sec_summary)
            if card:
                scorecards.append(card)
        
        # 构建报告
        self.report = {
            "report_id": f"OBS_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "summary": {
                "total_models": len(scorecards),
                "models_analyzed": [s.model for s in scorecards],
                "overall_status": self._determine_overall_status(scorecards, sec_summary)
            },
            "performance": {
                "available": len(perf_summaries) > 0,
                "models": [asdict(p) for p in perf_summaries]
            },
            "quality": {
                "available": len(qual_summaries) > 0,
                "models": [asdict(q) for q in qual_summaries]
            },
            "security": {
                "available": sec_summary is not None,
                "summary": asdict(sec_summary) if sec_summary else None
            },
            "scorecards": [asdict(s) for s in scorecards],
            "recommendations": self._generate_recommendations(scorecards, sec_summary)
        }
        
        return self.report
    
    def _determine_overall_status(self, scorecards: List[ModelScorecard], 
                                  sec: Optional[SecuritySummary]) -> str:
        """确定整体状态"""
        if not scorecards:
            return "NO_DATA"
        
        avg_score = sum(s.overall_score for s in scorecards) / len(scorecards)
        
        if sec and sec.blocked_count > 0:
            if avg_score >= 80:
                return "HEALTHY_WITH_WARNINGS"
            return "NEEDS_ATTENTION"
        
        if avg_score >= 90:
            return "EXCELLENT"
        elif avg_score >= 80:
            return "HEALTHY"
        elif avg_score >= 60:
            return "ACCEPTABLE"
        else:
            return "CRITICAL"
    
    def _generate_recommendations(self, scorecards: List[ModelScorecard],
                                   sec: Optional[SecuritySummary]) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        if not scorecards:
            recommendations.append("暂无数据，建议运行性能、质量和安全测试")
            return recommendations
        
        # 性能建议
        low_perf = [s for s in scorecards if s.performance_score < 60]
        if low_perf:
            recommendations.append(f"模型 {', '.join(m.model for m in low_perf)} 性能评分较低，建议优化推理速度")
        
        # 质量建议
        low_qual = [s for s in scorecards if s.quality_score < 60]
        if low_qual:
            recommendations.append(f"模型 {', '.join(m.model for m in low_qual)} 质量评分较低，建议检查模型输出准确性")
        
        # 安全建议
        if sec and sec.blocked_count > 0:
            recommendations.append(f"检测到 {sec.blocked_count} 个安全风险，建议加强输入过滤和内容审核")
        
        # 综合建议
        best_model = max(scorecards, key=lambda x: x.overall_score)
        worst_model = min(scorecards, key=lambda x: x.overall_score)
        if best_model.overall_score - worst_model.overall_score > 20:
            recommendations.append(f"模型间差异较大，{best_model.model} 表现优于 {worst_model.model}")
        
        if not recommendations:
            recommendations.append("系统运行良好，继续保持监控")
        
        return recommendations
    
    def save_report(self, filename: str = "observability_report.json"):
        """保存报告"""
        if not self.report:
            self.generate_report()
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.report, f, ensure_ascii=False, indent=2)
            print(f"\n[OK] 综合报告已保存: {filename}")
        except Exception as e:
            print(f"\n[ERROR] 保存报告失败: {e}")
    
    def print_dashboard(self):
        """打印仪表盘到终端"""
        if not self.report:
            self.generate_report()
        
        print("\n" + "="*60)
        print("LLM 综合仪表盘")
        print("="*60)
        print(f"报告ID: {self.report['report_id']}")
        print(f"生成时间: {self.report['generated_at']}")
        print(f"整体状态: {self.report['summary']['overall_status']}")
        
        # 评分卡
        print("\n" + "-"*60)
        print("模型评分卡")
        print("-"*60)
        for card in self.report['scorecards']:
            print(f"\n模型: {card['model']}")
            print(f"  性能评分: {card['performance_score']:.1f}")
            print(f"  质量评分: {card['quality_score']:.1f}")
            print(f"  安全评分: {card['security_score']:.1f}")
            print(f"  综合评分: {card['overall_score']:.1f} (等级: {card['grade']})")
        
        # 性能指标
        if self.report['performance']['available']:
            print("\n" + "-"*60)
            print("性能指标")
            print("-"*60)
            for p in self.report['performance']['models']:
                print(f"\n{p['model']}:")
                print(f"  平均TTFT: {p['avg_ttft']:.3f}s")
                print(f"  平均TPOT: {p['avg_tpot']:.3f}s")
                print(f"  吞吐量: {p['avg_throughput']:.1f} chars/s")
                print(f"  成功率: {p['success_rate']:.1f}%")
        
        # 质量指标
        if self.report['quality']['available']:
            print("\n" + "-"*60)
            print("质量指标")
            print("-"*60)
            for q in self.report['quality']['models']:
                print(f"\n{q['model']}:")
                print(f"  相关性: {q['avg_relevance']:.2f}/10")
                print(f"  准确性: {q['avg_accuracy']:.2f}/10")
                print(f"  完整性: {q['avg_completeness']:.2f}/10")
                print(f"  流畅性: {q['avg_fluency']:.2f}/10")
                print(f"  综合得分: {q['avg_total_score']:.2f}/10")
        
        # 安全指标
        if self.report['security']['available'] and self.report['security']['summary']:
            print("\n" + "-"*60)
            print("安全审计")
            print("-"*60)
            sec = self.report['security']['summary']
            print(f"  总检查数: {sec['total_checks']}")
            print(f"  安全: {sec['safe_count']}")
            print(f"  风险: {sec['blocked_count']}")
            print(f"  风险分布: {sec['risk_distribution']}")
        
        # 改进建议
        print("\n" + "-"*60)
        print("改进建议")
        print("-"*60)
        for i, rec in enumerate(self.report['recommendations'], 1):
            print(f"  {i}. {rec}")
        
        print("\n" + "="*60)


def main():
    """主函数"""
    print("="*60)
    print("LLM 综合仪表盘 (Observability Dashboard)")
    print("="*60)
    
    dashboard = ObservabilityDashboard()
    
    # 加载数据
    dashboard.load_performance_metrics("performance_metrics.csv")
    dashboard.load_quality_metrics("quality_metrics.csv")
    dashboard.load_security_audit("security_audit.json")
    
    # 生成报告
    dashboard.generate_report()
    
    # 打印仪表盘
    dashboard.print_dashboard()
    
    # 保存报告
    dashboard.save_report("observability_report.json")
    
    print("\n[OK] 综合仪表盘生成完成")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[ERROR] 程序异常: {e}")
        import traceback
        traceback.print_exc()
