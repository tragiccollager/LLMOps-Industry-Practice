#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
传统方式工具调用示例
直接按顺序硬编码调用工具函数
展示传统开发的"硬编码"特性：开发者必须规定死输入输出格式，
一旦需求变更（例如用户不想导出文件），这套流水线就失效了。
"""

import json
import csv
import time
import sys
import io
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional

# 修复 Windows 控制台编码问题
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


# ============ 模拟工具函数（返回JSON字符串，模拟API响应） ============

def get_employee_directory(department: Optional[str] = None) -> str:
    """
    获取员工目录信息
    返回JSON字符串（模拟API响应）
    """
    time.sleep(0.5)  # 模拟系统耗时
    
    # 模拟员工数据
    all_employees = [
        {"id": "E001", "name": "张三", "department": "技术部", "level": "P5", "base_salary": 15000},
        {"id": "E002", "name": "李四", "department": "技术部", "level": "P6", "base_salary": 20000},
        {"id": "E003", "name": "王五", "department": "销售部", "level": "M3", "base_salary": 12000},
        {"id": "E004", "name": "赵六", "department": "销售部", "level": "M2", "base_salary": 8000},
        {"id": "E005", "name": "钱七", "department": "财务部", "level": "P4", "base_salary": 10000},
    ]
    
    if department:
        employees = [e for e in all_employees if e["department"] == department]
    else:
        employees = all_employees
    
    print(f"[get_employee_directory] 获取到 {len(employees)} 名员工")
    
    return json.dumps({
        "success": True,
        "count": len(employees),
        "employees": employees
    })


def calculate_payroll_and_tax(employee_json: str, 
                               tax_rate: float = 0.1,
                               insurance_rate: float = 0.08) -> str:
    """
    计算工资和税费
    接收JSON字符串，返回JSON字符串
    """
    time.sleep(0.5)  # 模拟系统耗时
    
    try:
        employee_data = json.loads(employee_json)
        employees = employee_data.get("employees", [])
        payroll_records = []
        
        for emp in employees:
            base_salary = emp["base_salary"]
            tax = base_salary * tax_rate
            insurance = base_salary * insurance_rate
            net_salary = base_salary - tax - insurance
            
            payroll_records.append({
                "employee_id": emp["id"],
                "name": emp["name"],
                "department": emp["department"],
                "level": emp["level"],
                "应发工资": base_salary,
                "五险一金扣除": round(insurance, 2),
                "个税扣除": round(tax, 2),
                "实发工资": round(net_salary, 2)
            })
        
        total_payroll = sum(r["实发工资"] for r in payroll_records)
        
        print(f"[calculate_payroll_and_tax] 计算完成，总工资支出: ¥{total_payroll:,.2f}")
        
        return json.dumps({
            "success": True,
            "month": datetime.now().strftime("%Y-%m"),
            "total_payroll": round(total_payroll, 2),
            "records": payroll_records
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


def export_payroll_csv(payroll_json: str, filename: str = "payroll.csv") -> str:
    """
    导出工资单为CSV文件
    接收JSON字符串，返回JSON字符串
    """
    time.sleep(0.3)  # 模拟系统耗时
    
    try:
        payroll_data = json.loads(payroll_json)
        records = payroll_data.get("records", [])
        
        if not records:
            return json.dumps({"error": "没有工资记录可导出"})
        
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)
        
        print(f"[export_payroll_csv] 成功导出到 {filename}")
        
        return json.dumps({
            "success": True,
            "file_path": filename,
            "record_count": len(records)
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


# ============ 传统 SaaS 后端接口：硬编码的流水线，高度耦合 ============

def saas_generate_payroll_api(department: Optional[str] = None) -> Tuple[List[List[Any]], str]:
    """
    传统 SaaS 后端接口：硬编码的流水线，高度耦合
    
    特点：
    - 步骤被严格固化，依次执行：查数据 -> 算工资 -> 导出
    - 必须针对前端 UI 的表格组件进行特定的二维数组清洗
    - 严格返回元组，供前端强绑定渲染
    - 一旦需求变更（例如用户不想导出文件），这套流水线就失效了
    """
    print("\n" + "="*60)
    print("传统 SaaS 后端：硬编码流水线")
    print("="*60)
    
    try:
        time.sleep(1)  # 模拟系统耗时
        
        # 步骤被严格固化，依次执行：查数据 -> 算工资 -> 导出
        emp_str = get_employee_directory(department)
        emp_data = json.loads(emp_str)
        if "error" in emp_data:
            raise Exception(emp_str)
            
        payroll_str = calculate_payroll_and_tax(emp_str)
        if "error" in payroll_str:
            raise Exception(payroll_str)
            
        export_result_str = export_payroll_csv(payroll_str)
        export_result = json.loads(export_result_str)
        if "error" in export_result:
            raise Exception(export_result["error"])
        
        # 必须针对前端 UI 的表格组件进行特定的二维数组清洗
        # 这种格式是硬编码的，前端必须严格按照这个格式渲染
        payroll_data = json.loads(payroll_str)
        table_data = [
            [d["name"], d["level"], d["应发工资"], d["五险一金扣除"], d["个税扣除"], d["实发工资"]] 
            for d in payroll_data.get("records", [])
        ]
        
        # 严格返回元组，供前端强绑定渲染
        # 格式: (表格数据, 文件路径)
        print("✓ 硬编码流水线执行完成")
        print(f"  - 员工数: {len(table_data)}")
        print(f"  - 导出文件: {export_result.get('file_path')}")
        
        return table_data, export_result.get("file_path")
        
    except Exception as e:
        print(f"❌ SaaS 执行失败: {e}")
        # 即使出错也要返回固定的格式，前端必须处理这种格式
        return [[str(e), "", "", "", "", ""]], ""


# ============ 展示传统方式的局限性 ============

def demonstrate_limitations():
    """展示传统硬编码方式的局限性"""
    print("\n" + "="*60)
    print("传统方式的局限性")
    print("="*60)
    
    limitations = [
        "1. 调用顺序硬编码：必须按照 查数据->算工资->导出 的顺序执行",
        "2. 参数格式固定：每个函数的输入输出格式必须严格匹配",
        "3. 无法动态调整：如果用户只想查询不想导出，代码无法复用",
        "4. 前端强耦合：返回的二维数组格式是固定的，前端必须按此渲染",
        "5. 难以扩展：新增一个步骤需要修改整个流水线",
        "6. 错误处理复杂：每个步骤的错误都需要单独处理",
    ]
    
    for limitation in limitations:
        print(f"  {limitation}")
    
    print("\n对比MCP方式：")
    print("  - 工具可以独立发现和调用")
    print("  - 调用顺序由模型动态决定")
    print("  - 参数格式由Schema定义，自动验证")
    print("  - 前端只需展示结果，无需强绑定格式")


def main():
    """主函数"""
    try:
        print("传统工具调用方式示例")
        print("="*60)
        print("特点：\n  - 工具函数直接调用\n  - 调用顺序硬编码\n  - 参数传递固定\n  - 无法动态发现和组合\n  - 前端强耦合")
        
        # 示例1：处理所有部门
        print("\n\n示例1: 处理所有部门")
        try:
            table_data, file_path = saas_generate_payroll_api()
            print(f"\n返回的表格数据（供前端渲染）:")
            print(f"  表头: ['姓名', '级别', '应发工资', '五险一金', '个税', '实发工资']")
            for row in table_data[:3]:  # 只显示前3行
                print(f"  数据: {row}")
            print(f"  导出文件路径: {file_path}")
        except Exception as e:
            print(f"[错误] 示例1执行失败: {e}")
        
        # 示例2：只处理技术部
        print("\n\n" + "-"*60)
        print("示例2: 只处理技术部")
        try:
            table_data, file_path = saas_generate_payroll_api(department="技术部")
            print(f"\n返回的表格数据:")
            for row in table_data:
                print(f"  {row}")
        except Exception as e:
            print(f"[错误] 示例2执行失败: {e}")
        
        # 展示局限性
        demonstrate_limitations()
        
        print("\n" + "="*60)
        print("对比请运行: mcp_server.py 和 mcp_client.py")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n[信息] 用户中断执行")
    except Exception as e:
        print(f"\n[错误] 程序执行失败: {e}")


if __name__ == "__main__":
    main()
