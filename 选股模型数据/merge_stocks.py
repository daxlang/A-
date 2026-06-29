#!/usr/bin/env python3
"""合并四个行业的财报分析CSV为统一格式"""

import pandas as pd
import numpy as np
import os

BASE = r"C:\Users\daxlang\Desktop\杂七杂八的笔记\选股武器库\真实财报"

def load_steel():
    """钢铁: 使用聚焦评估五层级数据 (16家最完整的核心公司)"""
    path = os.path.join(BASE, r"Q1_2026_钢铁\聚焦评估\聚焦评估五层级数据.csv")
    df = pd.read_csv(path, encoding="utf-8-sig")
    
    result = pd.DataFrame()
    result["code"] = df["公司代码"]
    result["name"] = df["公司名称"]
    result["industry"] = "钢铁"
    result["report_period"] = "2026Q1"
    
    # 财务指标 (2026Q1)
    result["rev"] = df["2026Q1_营业总收入"] / 1e8  # 转为亿
    result["gross_margin"] = df["毛利率"] * 100   # 转百分比
    result["roa"] = df["回报层_ROA"] * 100
    result["asset_turnover"] = df["效率层_资产周转"]
    result["inventory_turnover"] = df["效率层_存货周转"]
    result["debt_ratio"] = df["健康层_资产负债率"] * 100
    result["operating_cf_ratio"] = df["生存层_现金生成"] * 100  # 销售现金比率
    result["net_profit_yi"] = df["回报层_净利润_亿"]
    result["net_margin"] = df["回报层_净利率"] * 100
    
    # 估值
    result["pe"] = df["估值层_PE"]
    result["pb"] = df["估值层_PB"]
    result["ps"] = df["估值层_PS"]
    result["dividend_yield"] = df["估值层_股息率"]
    result["market_cap"] = df["估值层_总市值"] / 1e8
    
    # 2025年对比数据 (注意: 2025_营业收入已经是以亿为单位)
    result["revenue_2025"] = df["2025_营业收入"]
    result["gross_margin_2025"] = df["2025_毛利率"] * 100
    result["roic_2025_raw"] = df["2025_ROIC"]
    result["revenue_yoy"] = df["营业收入_同比变化"]
    
    # 现有评分
    result["quality_score"] = pd.to_numeric(df["三扇门综合评分"], errors="coerce")
    result["quality_rank"] = pd.to_numeric(df["三扇门排名"], errors="coerce")
    
    result = result.dropna(subset=["code", "name"])
    return result


def load_baijiu():
    """白酒: 2026Q1分析"""
    path = os.path.join(BASE, r"Q1_2026_白酒\白酒行业2026Q1分析.csv")
    df = pd.read_csv(path, encoding="utf-8-sig")
    
    result = pd.DataFrame()
    result["code"] = df["code"]
    result["name"] = df["name"]
    result["industry"] = "白酒"
    result["report_period"] = "2026Q1"
    
    result["rev"] = df["rev"]
    result["gross_margin"] = df["gm"]
    result["net_margin"] = df["netm"]
    result["roa"] = df["roa"]
    result["asset_turnover"] = df["turnover"]
    result["debt_ratio"] = df["debt"]
    result["cash_ratio"] = df["cashratio"]  # 这里可能是现金/资产比
    
    result["pe"] = df["pe"]
    result["pb"] = df["pb"]
    result["ps"] = df["ps"]
    result["dividend_yield"] = df["div"]
    result["market_cap"] = df["mktcap"]
    
    result["quality_score"] = df["质量分"]
    result["valuation_score"] = df["温度计"]
    result["recommendation_score"] = df["推荐分"]
    result["valuation_zone"] = df["区域"]
    
    result = result.dropna(subset=["code", "name"])
    return result


def load_bank():
    """银行: 2026Q1分析"""
    path = os.path.join(BASE, r"Q1_2026_银行\银行行业2026Q1分析.csv")
    df = pd.read_csv(path, encoding="utf-8-sig")
    
    result = pd.DataFrame()
    result["code"] = df["code"]
    result["name"] = df["name"]
    result["industry"] = "银行"
    result["report_period"] = "2026Q1"
    
    result["rev"] = df["rev"]
    result["net_profit"] = df["net"]
    result["roe"] = df["roe"]
    result["roa"] = df["roa"]
    
    result["pe"] = df["pe"]
    result["pb"] = df["pb"]
    result["ps"] = df["ps"]
    result["dividend_yield"] = df["div"]
    result["market_cap"] = df["mktcap"]
    
    result["quality_score"] = df["质量分"]
    result["valuation_score"] = df["温度计"]
    result["recommendation_score"] = df["推荐分"]
    result["valuation_zone"] = df["区域"]
    
    result = result.dropna(subset=["code", "name"])
    return result


def load_game():
    """游戏: 2026Q1分析"""
    path = os.path.join(BASE, r"Q1_2026_游戏\游戏行业2026Q1分析.csv")
    df = pd.read_csv(path, encoding="utf-8-sig")
    
    result = pd.DataFrame()
    result["code"] = df["code"]
    result["name"] = df["name"]
    result["industry"] = "游戏"
    result["report_period"] = "2026Q1"
    
    result["rev"] = df["rev"]
    result["gross_margin"] = df["gm"]
    result["roa"] = df["roa"]
    result["revenue_growth"] = df["growth"]
    result["debt_ratio"] = df["debt"]
    result["cash_ratio"] = df["cashratio"]
    
    result["pe"] = df["pe"]
    result["pb"] = df["pb"]
    result["ps"] = df["ps"]
    result["dividend_yield"] = df["div"]
    result["market_cap"] = df["mktcap"]
    
    result["quality_score"] = df["质量分"]
    result["valuation_score"] = df["温度计"]
    result["recommendation_score"] = df["推荐分"]
    result["valuation_zone"] = df["区域"]
    
    result = result.dropna(subset=["code", "name"])
    return result


def main():
    steel = load_steel()
    baijiu = load_baijiu()
    bank = load_bank()
    game = load_game()
    
    # 合并
    all_stocks = pd.concat([steel, baijiu, bank, game], ignore_index=True)
    
    # 清洗
    all_stocks = all_stocks.replace([np.inf, -np.inf], np.nan)
    
    # 统计
    print("=" * 60)
    print("数据整合完成")
    print("=" * 60)
    for ind in all_stocks["industry"].unique():
        subset = all_stocks[all_stocks["industry"] == ind]
        print(f"  {ind}: {len(subset)} 家公司")
    print(f"  合计: {len(all_stocks)} 家公司")
    print()
    
    # 列统计
    print("字段覆盖情况 (非空率):")
    for col in sorted(all_stocks.columns):
        if col in ("code", "name", "industry", "report_period"):
            continue
        coverage = all_stocks[col].notna().sum()
        pct = coverage / len(all_stocks) * 100
        stars = "*" * int(pct / 5)
        print(f"  {col:<25s} {coverage:3d}/{len(all_stocks)} ({pct:5.1f}%) {stars}")
    
    # 保存
    outpath = r"C:\Users\daxlang\Desktop\杂七杂八的笔记\选股模型数据\all_stocks_q1_2026.csv"
    all_stocks.to_csv(outpath, index=False, encoding="utf-8-sig")
    print(f"\n保存至: {outpath}")
    
    # 分行业保存
    for ind in all_stocks["industry"].unique():
        subset = all_stocks[all_stocks["industry"] == ind]
        outpath_ind = rf"C:\Users\daxlang\Desktop\杂七杂八的笔记\选股模型数据\stocks_{ind}_q1_2026.csv"
        subset.to_csv(outpath_ind, index=False, encoding="utf-8-sig")
        print(f"分行业: {outpath_ind}")
    
    return all_stocks


if __name__ == "__main__":
    df = main()
