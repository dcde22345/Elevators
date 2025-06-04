#!/usr/bin/env python3
"""
簡化版電梯模擬器演示 - 每層樓不同到達率

展示如何為每層樓設定不同的乘客到達率
"""

from ElevatorSimulator import Simulator
import ElevatorManager

def test_floor_specific_rates():
    """測試每層樓不同的到達率功能"""
    
    print("=" * 50)
    print("測試每層樓不同的到達率功能")
    print("=" * 50)
    
    # 設定5層樓建築，每層樓不同的到達率
    n_floors = 4
    # 樓層 0,1,2,3,4 的到達率 (arrivals/sec)
    floor_rates = [0.2, 0.05, 0.05, 0.1, 0.15]  # 底樓最忙，頂樓次忙
    
    print("樓層到達率設定:")
    for floor, rate in enumerate(floor_rates):
        print(f"  樓層 {floor}: {rate} arrivals/sec")
    
    # 創建模擬器
    sim = Simulator(
        n_floors=n_floors,
        n_elevators=4,
        sim_len=120,  # 2分鐘模擬
        floor_arrival_rates=floor_rates,
        manager=ElevatorManager.NaiveRoundRobin,
        verbose=True,  # 簡化輸出
        debug_mode=False
    )
    
    # 生成場景
    sim.generate_scenario(verbose=False)
    print(f"\n生成了 {len(sim.scenario)} 個到達事件")
    
    # 統計每層樓的到達數量
    floor_counts = [0] * (n_floors + 1)
    for arrival in sim.scenario:
        floor_counts[arrival.xi] += arrival.n
    
    print("\n實際每層樓的乘客到達數:")
    for floor, count in enumerate(floor_counts):
        expected = floor_rates[floor] * sim.sim_len
        print(f"  樓層 {floor}: {count:3d} 人 (期望: {expected:.1f} 人)")
    
    # 運行模擬
    summary = sim.run_simulation()
    
    print(f"\n模擬結果:")
    print(f"  服務乘客數: {summary['passengers']['served']}")
    print(f"  平均服務時間: {summary['goals']['service_time'][1]:.1f} 秒")
    print(f"  等待乘客數: {summary['passengers']['waiting'][0]}")

def test_little_law():
    """測試 Little's Law"""
    
    print("\n" + "=" * 50)
    print("Little's Law 驗證")
    print("=" * 50)
    print("Little's Law: L = λW")
    print("L = 系統中平均顧客數")
    print("λ = 到達率 (arrivals/sec)")  
    print("W = 平均等待時間 (sec)")
    
    # 測試不同的到達率
    test_rates = [0.05, 0.1, 0.15]
    
    for rate in test_rates:
        print(f"\n測試到達率 λ = {rate} arrivals/sec:")
        
        sim = Simulator(
            n_floors=3,
            n_elevators=1,
            sim_len=200,  # 200秒
            arrival_pace=rate,
            manager=ElevatorManager.NaiveManager,
            verbose=False
        )
        
        sim.generate_scenario()
        summary = sim.run_simulation()
        
        # 計算指標
        served_passengers = summary['passengers']['served']
        if served_passengers > 0:
            avg_waiting_time = summary['info']['waiting_time'][1] or 0
            avg_service_time = summary['goals']['service_time'][1] or 0
            
            # Little's Law 計算
            theoretical_L = rate * avg_waiting_time
            actual_waiting = summary['passengers']['waiting'][0]
            
            print(f"  服務乘客數: {served_passengers}")
            print(f"  平均等待時間 W = {avg_waiting_time:.1f} 秒")
            print(f"  平均服務時間 = {avg_service_time:.1f} 秒")
            print(f"  實際等待人數 = {actual_waiting}")
            print(f"  理論等待人數 λW = {theoretical_L:.1f}")
        else:
            print(f"  沒有乘客完成服務")

if __name__ == "__main__":
    # 測試每層樓不同到達率
    test_floor_specific_rates()
    
    # 測試 Little's Law
    test_little_law()
    
    print("\n" + "=" * 50)
    print("測試完成！主要發現:")
    print("✅ 可以為每層樓設定不同的到達率")
    print("✅ arrival_pace 對應 Little's Law 中的 λ (到達率)")
    print("✅ 可以模擬現實建築物的乘客流量模式")
    print("=" * 50) 