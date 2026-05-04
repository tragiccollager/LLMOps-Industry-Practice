#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCP Server 实现
使用 FastMCP 框架将工具函数注册为 MCP Tools
暴露在本地端口上，供任何支持 MCP 的客户端调用
"""

import json
import csv
import time
import sys
import io
from datetime import datetime
from typing import List, Dict, Any, Optional
from mcp.server.fastmcp import FastMCP

# 修复 Windows 控制台编码问题
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


# 创建 MCP 服务器实例
mcp = FastMCP("payroll_service")


# ============ 模拟数据存储 ============
EMPLOYEES_DB = [
    {"id": "E001", "name": "张三", "department": "技术部", "level": "P5", "base_salary": 15000},
    {"id": "E002", "name": "李四", "department": "技术部", "level": "P6", "base_salary": 20000},
    {"id": "E003", "name": "王五", "department": "销售部", "level": "M3", "base_salary": 12000},
    {"id": "E004", "name": "赵六", "department": "销售部", "level": "M2", "base_salary": 8000},
    {"id": "E005", "name": "钱七", "department": "财务部", "level": "P4", "base_salary": 10000},
]


# ============ MCP Tools 定义 ============

@mcp.tool()
def get_employee_directory(department: str = None) -> str:
    """
    获取员工目录信息
    
    Args:
        department: 部门名称（可选），如"技术部"、"销售部"、"财务部"
    
    Returns:
        JSON字符串，包含员工列表
    """
    time.sleep(0.3)  # 模拟系统耗时
    
    if department:
        employees = [e for e in EMPLOYEES_DB if e["department"] == department]
    else:
        employees = EMPLOYEES_DB
    
    print(f"[MCP Tool] get_employee_directory 被调用，返回 {len(employees)} 名员工")
    
    return json.dumps({
        "success": True,
        "count": len(employees),
        "employees": employees
    }, ensure_ascii=False)


@mcp.tool()
def calculate_payroll(employee_json: str, 
                      tax_rate: float = 0.1,
                      insurance_rate: float = 0.08) -> str:
    """
    计算员工工资和税费
    
    Args:
        employee_json: 员工数据JSON字符串（来自get_employee_directory）
        tax_rate: 税率，默认0.1（10%）
        insurance_rate: 保险费率，默认0.08（8%）
    
    Returns:
        JSON字符串，包含计算后的工资单
    """
    time.sleep(0.3)  # 模拟系统耗时
    
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
                "base_salary": base_salary,
                "tax": round(tax, 2),
                "insurance": round(insurance, 2),
                "net_salary": round(net_salary, 2)
            })
        
        total_payroll = sum(r["net_salary"] for r in payroll_records)
        
        print(f"[MCP Tool] calculate_payroll 被调用，计算 {len(payroll_records)} 名员工工资")
        
        return json.dumps({
            "success": True,
            "month": datetime.now().strftime("%Y-%m"),
            "total_payroll": round(total_payroll, 2),
            "record_count": len(payroll_records),
            "records": payroll_records
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def export_to_csv(data_json: str, filename: str = "export.csv") -> str:
    """
    将数据导出为CSV文件
    
    Args:
        data_json: 数据JSON字符串，必须包含records字段
        filename: 输出文件名，默认"export.csv"
    
    Returns:
        JSON字符串，包含导出结果
    """
    time.sleep(0.2)  # 模拟系统耗时
    
    try:
        data = json.loads(data_json)
        records = data.get("records", [])
        
        if not records:
            return json.dumps({"error": "没有记录可导出"})
        
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)
        
        print(f"[MCP Tool] export_to_csv 被调用，导出 {len(records)} 条记录到 {filename}")
        
        return json.dumps({
            "success": True,
            "file_path": filename,
            "record_count": len(records),
            "message": f"成功导出到 {filename}"
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def analyze_department_summary(employee_json: str) -> str:
    """
    分析部门统计信息
    
    Args:
        employee_json: 员工数据JSON字符串
    
    Returns:
        JSON字符串，包含部门统计
    """
    time.sleep(0.2)
    
    try:
        data = json.loads(employee_json)
        employees = data.get("employees", [])
        
        # 按部门统计
        dept_stats = {}
        for emp in employees:
            dept = emp["department"]
            if dept not in dept_stats:
                dept_stats[dept] = {"count": 0, "total_salary": 0}
            dept_stats[dept]["count"] += 1
            dept_stats[dept]["total_salary"] += emp["base_salary"]
        
        print(f"[MCP Tool] analyze_department_summary 被调用，分析 {len(dept_stats)} 个部门")
        
        return json.dumps({
            "success": True,
            "department_count": len(dept_stats),
            "total_employees": len(employees),
            "departments": dept_stats
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)})


# ============ MCP Server 启动 ============

def main():
    """启动 MCP Server"""
    try:
        print("="*60)
        print("MCP Server 启动")
        print("="*60)
        print("\n已注册的工具:")
        print("  1. get_employee_directory - 获取员工目录")
        print("  2. calculate_payroll - 计算工资和税费")
        print("  3. export_to_csv - 导出CSV文件")
        print("  4. analyze_department_summary - 部门统计分析")
        print("\n服务器正在运行，等待客户端连接...")
        print("按 Ctrl+C 停止服务器")
        print("="*60)
        
        # 启动服务器（stdio 模式）
        mcp.run(transport='stdio')
    except KeyboardInterrupt:
        print("\n[信息] 服务器被用户停止")
    except Exception as e:
        print(f"\n[错误] 服务器运行失败: {e}")
        raise


if __name__ == "__main__":
    main()
