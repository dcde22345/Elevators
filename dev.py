import sys
import numpy as np
import os
import pandas as pd
from MyTools import *
from algorithm.DDSAlgorithm import DDSManager
from algorithm.LookAlgorithm import Look
from ElevatorSimulator import Simulator
from AnalysisPlotter import AnalysisPlotter
import matplotlib.pyplot as plt

if __name__ == "__main__":
    sim_pace = 100 if 'verbose' in sys.argv else None

    data = [
        [0, 0, 0, 0, 0, 0, 0, 0.5, 0, 0, 0, 0.5, 1],                        # B1f
        [1, 0, 0, 16.5, 16.5, 11, 30.5, 28.5, 20.5, 9.5, 3.5, 2.5, 4],      # 1f
        [0, 0, 0, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0],                          # 2f
        [0, 4, 0, 0, 0, 0, 0, 0, 0, 4.5, 0, 0, 0],                          # 3f
        [0, 13.5, 0.5, 0, 0, 0, 0.5, 0.5, 0.5, 1, 0.5, 0, 0],               # 4f
        [0, 7.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],                          # 5f
        [0.5, 12.5, 0.5, 0, 0.5, 0, 0, 0, 0.5, 0, 0, 0.5, 0.5],             # 6f
        [0.5, 7.5, 1.5, 0, 0, 0, 0, 0, 0, 0, 0.5, 0.5, 0.5],                # 7f
        [0, 6.5, 0.5, 0.5, 1, 0, 0, 0, 0, 0, 0, 0, 0],                      # 8f
        [0, 6, 0, 0.5, 1, 0.5, 0.5, 1, 0, 0, 0, 0, 0],                      # 9f
        [0, 4, 0, 0, 0.5, 0, 0.5, 1.5, 0, 0, 0.5, 0, 0],                    # 10f
        [0, 1, 0, 0, 0, 1, 0, 0.5, 0, 0, 0, 0, 0],                          # 11f
        [0.5, 4, 0, 0, 0.5, 0, 0.5, 0, 0, 0, 0, 0, 0]                       # 12f
    ]

    new_data = [
        [0, 0, 0, 0, 0, 0, 0, 0.5, 0, 0, 0, 0.5, 1],
        [1, 0, 19, 35.5, 16.5, 11, 30.5, 28.5, 20.5, 9.5, 3.5, 2.5, 4],
        [0, 19, 0, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0],
        [0, 23, 0, 0, 0, 0, 0, 0, 0, 4.5, 0, 0, 0],
        [0, 13.5, 0.5, 0, 0, 0, 0.5, 0.5, 0.5, 1, 0.5, 0, 0],
        [0, 7.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0.5, 12.5, 0.5, 0, 0.5, 0, 0, 0, 0.5, 0, 0, 0.5, 0.5],
        [0.5, 7.5, 1.5, 0, 0, 0, 0, 0, 0, 0, 0.5, 0.5, 0.5],
        [0, 6.5, 0.5, 0.5, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 6, 0, 0.5, 1, 0.5, 0.5, 1, 0, 0, 0, 0, 0],
        [0, 4, 0, 0, 0.5, 0, 0.5, 1.5, 0, 0, 0.5, 0, 0],
        [0, 1, 0, 0, 0, 1, 0, 0.5, 0, 0, 0, 0, 0],
        [0.5, 4, 0, 0, 0.5, 0, 0.5, 0, 0, 0, 0, 0, 0]
    ]

    data_60_at_six_floor = [
        [0, 0, 0, 0, 0, 0, 0, 0.5, 0, 0, 0, 0.5, 1],
        [1, 0, 19, 35.5, 16.5, 11, 60, 28.5, 20.5, 9.5, 3.5, 2.5, 4],
        [0, 19, 0, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0],
        [0, 23, 0, 0, 0, 0, 0, 0, 0, 4.5, 0, 0, 0],
        [0, 13.5, 0.5, 0, 0, 0, 0.5, 0.5, 0.5, 1, 0.5, 0, 0],
        [0, 7.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0.5, 12.5, 0.5, 0, 0.5, 0, 0, 0, 0.5, 0, 0, 0.5, 0.5],
        [0.5, 7.5, 1.5, 0, 0, 0, 0, 0, 0, 0, 0.5, 0.5, 0.5],
        [0, 6.5, 0.5, 0.5, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 6, 0, 0.5, 1, 0.5, 0.5, 1, 0, 0, 0, 0, 0],
        [0, 4, 0, 0, 0.5, 0, 0.5, 1.5, 0, 0, 0.5, 0, 0],
        [0, 1, 0, 0, 0, 1, 0, 0.5, 0, 0, 0, 0, 0],
        [0.5, 4, 0, 0, 0.5, 0, 0.5, 0, 0, 0, 0, 0, 0]
    ]

    data_120_at_six_floor = [
        [0, 0, 0, 0, 0, 0, 0, 0.5, 0, 0, 0, 0.5, 1],
        [1, 0, 19, 35.5, 16.5, 11, 120, 28.5, 20.5, 9.5, 3.5, 2.5, 4],
        [0, 19, 0, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0],
        [0, 23, 0, 0, 0, 0, 0, 0, 0, 4.5, 0, 0, 0],
        [0, 13.5, 0.5, 0, 0, 0, 0.5, 0.5, 0.5, 1, 0.5, 0, 0],
        [0, 7.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0.5, 12.5, 0.5, 0, 0.5, 0, 0, 0, 0.5, 0, 0, 0.5, 0.5],
        [0.5, 7.5, 1.5, 0, 0, 0, 0, 0, 0, 0, 0.5, 0.5, 0.5],
        [0, 6.5, 0.5, 0.5, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 6, 0, 0.5, 1, 0.5, 0.5, 1, 0, 0, 0, 0, 0],
        [0, 4, 0, 0, 0.5, 0, 0.5, 1.5, 0, 0, 0.5, 0, 0],
        [0, 1, 0, 0, 0, 1, 0, 0.5, 0, 0, 0, 0, 0],
        [0.5, 4, 0, 0, 0.5, 0, 0.5, 0, 0, 0, 0, 0, 0]
    ]

    data_180_at_six_floor = [
        [0, 0, 0, 0, 0, 0, 0, 0.5, 0, 0, 0, 0.5, 1],
        [1, 0, 19, 35.5, 16.5, 11, 180, 28.5, 20.5, 9.5, 3.5, 2.5, 4],
        [0, 19, 0, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0],
        [0, 23, 0, 0, 0, 0, 0, 0, 0, 4.5, 0, 0, 0],
        [0, 13.5, 0.5, 0, 0, 0, 0.5, 0.5, 0.5, 1, 0.5, 0, 0],
        [0, 7.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0.5, 12.5, 0.5, 0, 0.5, 0, 0, 0, 0.5, 0, 0, 0.5, 0.5],
        [0.5, 7.5, 1.5, 0, 0, 0, 0, 0, 0, 0, 0.5, 0.5, 0.5],
        [0, 6.5, 0.5, 0.5, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 6, 0, 0.5, 1, 0.5, 0.5, 1, 0, 0, 0, 0, 0],
        [0, 4, 0, 0, 0.5, 0, 0.5, 1.5, 0, 0, 0.5, 0, 0],
        [0, 1, 0, 0, 0, 1, 0, 0.5, 0, 0, 0, 0, 0],
        [0.5, 4, 0, 0, 0.5, 0, 0.5, 0, 0, 0, 0, 0, 0]
    ]

    data_with_faculty_in = [
        [0, 0, 0, 0, 0, 0, 0, 0.5, 0, 0, 0, 0.5, 1],                        # B1f
        [1, 0, 0, 16.5, 16.5, 11, 30.5, 28.5, 20.5, 35.5, 57.5, 56.5, 63],      # 1f
        [0, 0, 0, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0],                          # 2f
        [0, 4, 0, 0, 0, 0, 0, 0, 0, 4.5, 0, 0, 0],                          # 3f
        [0, 13.5, 0.5, 0, 0, 0, 0.5, 0.5, 0.5, 1, 0.5, 0, 0],               # 4f
        [0, 7.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],                          # 5f
        [0.5, 12.5, 0.5, 0, 0.5, 0, 0, 0, 0.5, 0, 0, 0.5, 0.5],             # 6f
        [0.5, 7.5, 1.5, 0, 0, 0, 0, 0, 0, 0, 0.5, 0.5, 0.5],                # 7f
        [0, 6.5, 0.5, 0.5, 1, 0, 0, 0, 0, 0, 0, 0, 0],                      # 8f
        [0, 32, 0, 0.5, 1, 0.5, 0.5, 1, 0, 0, 0, 0, 0],                      # 9f
        [0, 58, 0, 0, 0.5, 0, 0.5, 1.5, 0, 0, 0.5, 0, 0],                    # 10f
        [0, 55, 0, 0, 0, 1, 0, 0.5, 0, 0, 0, 0, 0],                          # 11f
        [0.5, 63, 0, 0, 0.5, 0, 0.5, 0, 0, 0, 0, 0, 0]                       # 12f
    ]

    limitations = [
        None,                       # 能夠到達B1-12樓
        [2, 3, 4, 5],               # 能夠到達B1, 1樓, 6樓-12樓
        [0, 2, 3, 5, 7, 9, 11],     # 能夠到達4, 6, 8, 10, 12樓
        [0, 2, 4, 6, 8, 10, 12]     # 能夠到達3, 5, 7, 9, 11樓
    ]

    privilege_limitations = [
        [2, 3, 4, 5, 6, 7, 8],      # 能夠到 1, 9, 10, 11, 12樓
        None,                       # 能夠到 B1-12樓
        None,                       # 能夠到 1-12樓
        None                        # 能夠到 1-12樓
    ]

    def run_elevator_simulation(data, sim_pace=None, manager=Look, limitations=None, seed=1, prefix=None):
        # Calculate arrival rates based on row sums of data matrix
        # Each row represents passengers starting from that floor
        # Data is for 20 minutes, so convert to arrivals per second
        floor_arrival_rates = [sum(row)/(20*60) for row in data]  # 10 minutes = 600 seconds

        # Calculate destination probability matrix
        # destination_probabilities[i][j] = probability of going from floor i to floor j
        destination_probabilities = []
        for dest_floor in range(len(data[0])):  # for each destination floor
            col_sum = sum(data[i][dest_floor] for i in range(len(data)))  # sum of column
            if col_sum > 0:
                col_probs = [data[i][dest_floor] / col_sum for i in range(len(data))]
            else:
                col_probs = [0] * len(data)
            destination_probabilities.append(col_probs)

        # Transpose to get destination_probabilities[from_floor][to_floor]
        destination_probabilities = [[destination_probabilities[j][i] for j in range(len(destination_probabilities))] 
                                        for i in range(len(data))]

        x = Simulator(
            n_floors=13,  # （B1-f12，共13個樓層）
            n_elevators=4,
            sim_len=10*60,  # 10 minutes = 600 seconds
            sim_pace=sim_pace,
            capacity=10,    # 電梯容量10人
            speed=0.2,    # 5秒一層樓 -> 0.2樓/秒
            open_time=5.5,   # 開門時間5秒
            delay=1,       # 一個人進入的時間3秒
            floor_arrival_rates=floor_arrival_rates,  # 使用樓層特定的到達率
            destination_probabilities=destination_probabilities,  # 使用機率矩陣
            manager=manager,
            limitations=limitations,
            debug_mode=False,
            seed=seed
        )

        arrivals_per_minute = x.generate_scenario(prefix=prefix)
        summary, all_passengers_data, quartile_stats = x.run_simulation(prefix=prefix)
        
        # 輸出統計結果
        served_passengers = summary['passengers']['served']
        print(f"\n=== Elevator system performance statistics ===")
        print(f"Served passengers: {served_passengers}")
        print(f"Total waiting time: {summary['info']['total_waiting_time']:.2f}s")
        print(f"Total service time: {summary['info']['total_inside_time']:.2f}s")
        
        if served_passengers > 0:
            total_waiting_people = summary['info']['waiting_time'][0]
            total_inside_people = summary['info']['inside_time'][0]
            total_service_people = summary['goals']['service_time'][0]

            avg_waiting_time = summary['info']['waiting_time'][1]  # 平均等待時間
            avg_inside_time = summary['info']['inside_time'][1]    # 平均乘坐時間
            avg_service_time = summary['goals']['service_time'][1] # 平均服務時間
            
            print(f"Average waiting time: {avg_waiting_time:.2f}s")
            print(f"Average service time: {avg_inside_time:.2f}s")
        
            print(f"\nWaiting time distribution: ")
            print(f"min: {summary['info']['waiting_time'][2]}")
            print(f"median: {summary['info']['waiting_time'][3]}")
            print(f"max: {summary['info']['waiting_time'][5]}")

            print(f"\nInside time distribution: ")
            print(f"min: {summary['info']['inside_time'][2]}")
            print(f"median: {summary['info']['inside_time'][3]}")
            print(f"max: {summary['info']['inside_time'][5]}")

            print(f"\nService time distribution: ")
            print(f"min: {summary['goals']['service_time'][2]}")
            print(f"median: {summary['goals']['service_time'][3]}")
            print(f"max: {summary['goals']['service_time'][5]}")

            print(f"\nAverage time in the system time: {avg_service_time:.2f}s")
            print(f"Verification: {avg_service_time:.2f} ≈ {avg_waiting_time:.2f} + {avg_inside_time:.2f} = {avg_waiting_time + avg_inside_time:.2f}")
        else:
            print("沒有乘客完成服務")
            
        return summary, all_passengers_data, quartile_stats, arrivals_per_minute
        
    def calculate_monte_carlo_averages(summaries, all_passengers_data_list, arrivals_per_minute_list, scenario_name):
        """
        计算蒙特卡罗模拟结果的平均值
        
        Args:
            summaries: 多次模拟的summary结果列表
            all_passengers_data_list: 多次模拟的all_passengers_data结果列表
            arrivals_per_minute_list: 多次模拟的arrivals_per_minute结果列表
            scenario_name: 场景名称
        
        Returns:
            dict: 包含平均值的统计结果
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from collections import defaultdict
        
        n_simulations = len(summaries)
        if n_simulations == 0:
            return {}
        
        print(f"\n=== Monte Carlo Analysis for {scenario_name} ({n_simulations} simulations) ===")
        
        # 计算基本统计的平均值
        avg_served = np.mean([s['passengers']['served'] for s in summaries])
        avg_waiting_time = np.mean([s['info']['waiting_time'][1] for s in summaries])  # 平均等待时间
        avg_inside_time = np.mean([s['info']['inside_time'][1] for s in summaries])    # 平均在电梯内时间
        avg_service_time = np.mean([s['goals']['service_time'][1] for s in summaries]) # 平均服务时间
        
        # 计算楼层等待时间变异性的平均值
        floor_stats_aggregated = defaultdict(lambda: {'counts': [], 'means': [], 'stds': [], 'cvs': []})
        
        for summary in summaries:
            if 'floor_waiting_stats' in summary['info']:
                floor_waiting_stats = summary['info']['floor_waiting_stats']
                for floor, stats in floor_waiting_stats.items():
                    floor_stats_aggregated[floor]['counts'].append(stats['count'])
                    floor_stats_aggregated[floor]['means'].append(stats['mean'])
                    floor_stats_aggregated[floor]['stds'].append(stats['std'])
                    cv = stats['std'] / stats['mean'] if stats['mean'] > 0 else 0
                    floor_stats_aggregated[floor]['cvs'].append(cv)
        
        # 计算每个楼层的平均统计
        floor_avg_stats = {}
        for floor, stats_lists in floor_stats_aggregated.items():
            if stats_lists['counts']:  # 确保有数据
                floor_avg_stats[floor] = {
                    'avg_count': np.mean(stats_lists['counts']),
                    'avg_mean': np.mean(stats_lists['means']),
                    'avg_std': np.mean(stats_lists['stds']),
                    'avg_cv': np.mean(stats_lists['cvs']),
                    'std_of_means': np.std(stats_lists['means']),  # 平均值的标准差
                    'std_of_stds': np.std(stats_lists['stds'])     # 标准差的标准差
                }
        
        # 计算平均 arrival rate 并可视化
        if arrivals_per_minute_list:
            combined_arrivals_per_minute = _calculate_average_arrivals_per_minute(arrivals_per_minute_list, scenario_name)
        
        # 打印结果
        print(f"Average served passengers: {avg_served:.1f}")
        print(f"Average waiting time: {avg_waiting_time:.1f}s")
        print(f"Average inside time: {avg_inside_time:.1f}s") 
        print(f"Average service time: {avg_service_time:.1f}s")
        
        print(f"\n=== Monte Carlo Floor Waiting Time Variability Analysis ===")
        print(f"{'Floor':<6} {'Avg Count':<10} {'Avg Mean':<10} {'Avg Std':<10} {'Avg CV':<10} {'Std of Means':<12} {'Std of Stds':<12}")
        print("-" * 80)
        
        for floor in sorted(floor_avg_stats.keys()):
            stats = floor_avg_stats[floor]
            floor_name = f"{floor}F" if floor > 0 else "B1"
            print(f"{floor_name:<6} {stats['avg_count']:<10.1f} {stats['avg_mean']:<10.1f} {stats['avg_std']:<10.1f} "
                  f"{stats['avg_cv']:<10.3f} {stats['std_of_means']:<12.1f} {stats['std_of_stds']:<12.1f}")
        
        # 分析变异性最大的楼层
        if floor_avg_stats:
            max_avg_cv_floor = max(floor_avg_stats.items(), key=lambda x: x[1]['avg_cv'])
            max_std_of_means_floor = max(floor_avg_stats.items(), key=lambda x: x[1]['std_of_means'])
            
            print(f"\nMonte Carlo Variability Analysis:")
            print(f"Floor with highest average CV: {max_avg_cv_floor[0]}F (avg CV: {max_avg_cv_floor[1]['avg_cv']:.3f})")
            print(f"Floor with highest variability in means: {max_std_of_means_floor[0]}F (std of means: {max_std_of_means_floor[1]['std_of_means']:.1f}s)")
        
        # 返回结果用于进一步分析
        monte_carlo_results = {
            'scenario_name': scenario_name,
            'n_simulations': n_simulations,
            'avg_served': avg_served,
            'avg_waiting_time': avg_waiting_time,
            'avg_inside_time': avg_inside_time,
            'avg_service_time': avg_service_time,
            'floor_avg_stats': floor_avg_stats,
            'combined_arrivals_per_minute': combined_arrivals_per_minute if arrivals_per_minute_list else None
        }
        
        return monte_carlo_results

    def _calculate_average_arrivals_per_minute(arrivals_per_minute_list, scenario_name):
        """
        计算多次模拟的平均 arrivals per minute 并绘制图表
        
        Args:
            arrivals_per_minute_list: 多次模拟的arrivals_per_minute结果列表
            scenario_name: 场景名称
            
        Returns:
            dict: 平均arrivals per minute数据
        """

        
        print(f"\n=== Average Arrival Rate Analysis for {scenario_name} ===")
        
        # 找到所有分钟的最大值，确保所有模拟有相同的时间范围
        max_minutes = 0
        for arrivals_dict in arrivals_per_minute_list:
            if arrivals_dict:
                max_minutes = max(max_minutes, max(arrivals_dict.keys()) + 1)
        
        if max_minutes == 0:
            print("No arrival data found in simulations")
            return {}
        
        # 计算每分钟的平均到达人数和标准差
        combined_arrivals = {}
        arrival_matrix = []  # 用于计算标准差
        
        for minute in range(max_minutes):
            arrivals_for_minute = []
            for arrivals_dict in arrivals_per_minute_list:
                arrivals_for_minute.append(arrivals_dict.get(minute, 0))
            
            arrival_matrix.append(arrivals_for_minute)
            combined_arrivals[minute] = {
                'mean': np.mean(arrivals_for_minute),
                'std': np.std(arrivals_for_minute),
                'min': np.min(arrivals_for_minute),
                'max': np.max(arrivals_for_minute)
            }
        
        # 准备绘图数据
        minutes = list(range(max_minutes))
        mean_arrivals = [combined_arrivals[m]['mean'] for m in minutes]
        std_arrivals = [combined_arrivals[m]['std'] for m in minutes]
        
        # 设置中文字体支持
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 创建图表
        plt.figure(figsize=(14, 8))
        
        # 绘制平均值线
        plt.plot(minutes, mean_arrivals, 'b-', linewidth=2, label=f'平均值 (n={len(arrivals_per_minute_list)})')
        
        # 绘制置信区间 (均值 ± 标准差)
        upper_bound = [mean_arrivals[i] + std_arrivals[i] for i in range(len(minutes))]
        lower_bound = [max(0, mean_arrivals[i] - std_arrivals[i]) for i in range(len(minutes))]
        plt.fill_between(minutes, lower_bound, upper_bound, alpha=0.3, color='lightblue', label='±1 標準差')
        
        # 添加散点显示数据密度
        plt.scatter(minutes, mean_arrivals, c='red', s=30, alpha=0.7, zorder=5)
        
        # 设置图表样式
        plt.title(f'Monte Carlo Average Arrival Rate - {scenario_name}', fontsize=16, fontweight='bold')
        plt.xlabel('Time (minutes)', fontsize=14)
        plt.ylabel('Average Number of Arriving Passengers', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # 添加统计信息
        total_avg_passengers = sum(mean_arrivals)
        overall_avg_per_minute = total_avg_passengers / len(minutes) if minutes else 0
        max_avg_passengers = max(mean_arrivals) if mean_arrivals else 0
        min_avg_passengers = min(mean_arrivals) if mean_arrivals else 0
        
        stats_text = f'總平均乘客數: {total_avg_passengers:.1f}\n每分鐘平均: {overall_avg_per_minute:.1f}\n最高平均值: {max_avg_passengers:.1f}\n最低平均值: {min_avg_passengers:.1f}'
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # 保存图表
        save_path = f"images/monte_carlo_average_arrivals_{scenario_name}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Average arrival rate chart saved to: {save_path}")
        plt.close()  # 关闭图表以节省内存
        
        # 打印详细统计
        print(f"\n每分鐘平均到達人數統計:")
        print(f"{'分鐘':<6} {'平均值':<8} {'標準差':<8} {'最小值':<6} {'最大值':<6}")
        print("-" * 40)
        for minute in range(min(10, max_minutes)):  # 只显示前10分钟
            stats = combined_arrivals[minute]
            print(f"{minute:<6} {stats['mean']:<8.1f} {stats['std']:<8.1f} {stats['min']:<6} {stats['max']:<6}")
        
        if max_minutes > 10:
            print("... (showing first 10 minutes only)")
        
        print(f"\n整體統計:")
        print(f"總模擬時長: {max_minutes} 分鐘")
        print(f"總平均乘客數: {total_avg_passengers:.1f}")
        print(f"每分鐘平均到達率: {overall_avg_per_minute:.1f} 人/分鐘")
        print(f"峰值平均到達率: {max_avg_passengers:.1f} 人/分鐘")
        
        return combined_arrivals
    
    data_60_at_six_floor_summary = []
    data_60_at_six_floor_all_passengers_data = []
    data_60_at_six_floor_floor_quartile_stats = []
    data_60_at_six_floor_total_quartile_stats = []
    data_60_at_six_floor_arrivals_per_minute = []

    limit_data_60_at_six_floor_summary = []
    limit_data_60_at_six_floor_all_passengers_data = []
    limit_data_60_at_six_floor_floor_quartile_stats = []
    limit_data_60_at_six_floor_total_quartile_stats = []
    limit_data_60_at_six_floor_arrivals_per_minute = []

    data_120_at_six_floor_summary = []
    data_120_at_six_floor_all_passengers_data = []
    data_120_at_six_floor_floor_quartile_stats = []
    data_120_at_six_floor_total_quartile_stats = []
    data_120_at_six_floor_arrivals_per_minute = []

    limit_data_120_at_six_floor_summary = []
    limit_data_120_at_six_floor_all_passengers_data = []
    limit_data_120_at_six_floor_floor_quartile_stats = []
    limit_data_120_at_six_floor_total_quartile_stats = []
    limit_data_120_at_six_floor_arrivals_per_minute = []

    data_180_at_six_floor_summary = []
    data_180_at_six_floor_all_passengers_data = []
    data_180_at_six_floor_floor_quartile_stats = []
    data_180_at_six_floor_total_quartile_stats = []
    data_180_at_six_floor_arrivals_per_minute = []

    limit_data_180_at_six_floor_summary = []
    limit_data_180_at_six_floor_all_passengers_data = []
    limit_data_180_at_six_floor_floor_quartile_stats = []
    limit_data_180_at_six_floor_total_quartile_stats = []
    limit_data_180_at_six_floor_arrivals_per_minute = []

    no_limit_summary = []
    no_limit_all_passengers_data = []
    no_limit_floor_quartile_stats = []
    no_limit_total_quartile_stats = []
    no_limit_arrivals_per_minute = []

    limit_summary = []
    limit_all_passengers_data = []
    limit_floor_quartile_stats = []
    limit_total_quartile_stats = []
    limit_arrivals_per_minute = []

    # ================================================================================================================
    # 有教職員在的資料

    data_with_faculty_in_summary = []
    data_with_faculty_in_all_passengers_data = []
    data_with_faculty_in_floor_quartile_stats = []
    data_with_faculty_in_total_quartile_stats = []
    data_with_faculty_in_arrivals_per_minute = []

    limit_data_with_faculty_in_summary = []
    limit_data_with_faculty_in_all_passengers_data = []
    limit_data_with_faculty_in_floor_quartile_stats = []
    limit_data_with_faculty_in_total_quartile_stats = []
    limit_data_with_faculty_in_arrivals_per_minute = []

    priviliged_data_with_faculty_in_summary = []
    priviliged_data_with_faculty_in_all_passengers_data = []
    priviliged_data_with_faculty_in_floor_quartile_stats = []
    priviliged_data_with_faculty_in_total_quartile_stats = []
    priviliged_data_with_faculty_in_arrivals_per_minute = []

    # Monte Carlo simulation loop
    for i in range(3):  # 改为3次展示蒙特卡罗效果
        # print(f"\n=== Simulation {i+1}/{3} ===")
        
        # # -----------------------limitations-----------------------
        # summary, all_passengers_data, quartile_stats, arrivals_per_minute = run_elevator_simulation(data, sim_pace=100, manager=Look, limitations=limitations, seed=i+5, prefix=f"limit_{i}")
        # limit_summary.append(summary)
        # limit_all_passengers_data.append(all_passengers_data)
        # limit_floor_quartile_stats.append(quartile_stats['floor'])
        # limit_total_quartile_stats.append(quartile_stats['total'])
        # limit_arrivals_per_minute.append(arrivals_per_minute)
        
        # summary, all_passengers_data, quartile_stats, arrivals_per_minute = run_elevator_simulation(data_60_at_six_floor, sim_pace=None, manager=Look, limitations=limitations, seed=i+5, prefix=f"limit_data_60_at_six_floor_{i}")
        # limit_data_60_at_six_floor_summary.append(summary)
        # limit_data_60_at_six_floor_all_passengers_data.append(all_passengers_data)
        # limit_data_60_at_six_floor_floor_quartile_stats.append(quartile_stats['floor'])
        # limit_data_60_at_six_floor_total_quartile_stats.append(quartile_stats['total'])
        # limit_data_60_at_six_floor_arrivals_per_minute.append(arrivals_per_minute)
        
        # summary, all_passengers_data, quartile_stats, arrivals_per_minute = run_elevator_simulation(data_120_at_six_floor, sim_pace=None, manager=Look, limitations=limitations, seed=i+5, prefix=f"limit_data_120_at_six_floor_{i}")
        # limit_data_120_at_six_floor_summary.append(summary)
        # limit_data_120_at_six_floor_all_passengers_data.append(all_passengers_data)
        # limit_data_120_at_six_floor_floor_quartile_stats.append(quartile_stats['floor'])
        # limit_data_120_at_six_floor_total_quartile_stats.append(quartile_stats['total'])
        # limit_data_120_at_six_floor_arrivals_per_minute.append(arrivals_per_minute)

        # summary, all_passengers_data, quartile_stats, arrivals_per_minute = run_elevator_simulation(data_180_at_six_floor, sim_pace=None, manager=Look, limitations=limitations, seed=i+5, prefix=f"limit_data_180_at_six_floor_{i}")
        # limit_data_180_at_six_floor_summary.append(summary)
        # limit_data_180_at_six_floor_all_passengers_data.append(all_passengers_data)
        # limit_data_180_at_six_floor_floor_quartile_stats.append(quartile_stats['floor'])
        # limit_data_180_at_six_floor_total_quartile_stats.append(quartile_stats['total'])
        # limit_data_180_at_six_floor_arrivals_per_minute.append(arrivals_per_minute)

        # # -----------------------no limitations-----------------------
        # summary, all_passengers_data, quartile_stats, arrivals_per_minute = run_elevator_simulation(new_data, sim_pace=None, manager=Look, limitations=None, seed=i+1, prefix=f"no_limit_{i}")
        # no_limit_summary.append(summary)
        # no_limit_all_passengers_data.append(all_passengers_data)
        # no_limit_floor_quartile_stats.append(quartile_stats['floor'])
        # no_limit_total_quartile_stats.append(quartile_stats['total'])
        # no_limit_arrivals_per_minute.append(arrivals_per_minute)
        
        # summary, all_passengers_data, quartile_stats, arrivals_per_minute = run_elevator_simulation(data_60_at_six_floor, sim_pace=None, manager=Look, limitations=None, seed=i+5, prefix=f"data_60_at_six_floor_{i}")
        # data_60_at_six_floor_summary.append(summary)
        # data_60_at_six_floor_all_passengers_data.append(all_passengers_data)
        # data_60_at_six_floor_floor_quartile_stats.append(quartile_stats['floor'])
        # data_60_at_six_floor_total_quartile_stats.append(quartile_stats['total'])
        # data_60_at_six_floor_arrivals_per_minute.append(arrivals_per_minute)
        
        # summary, all_passengers_data, quartile_stats, arrivals_per_minute = run_elevator_simulation(data_120_at_six_floor, sim_pace=None, manager=Look, limitations=None, seed=i+5, prefix=f"data_120_at_six_floor_{i}")
        # data_120_at_six_floor_summary.append(summary)
        # data_120_at_six_floor_all_passengers_data.append(all_passengers_data)
        # data_120_at_six_floor_floor_quartile_stats.append(quartile_stats['floor'])
        # data_120_at_six_floor_total_quartile_stats.append(quartile_stats['total'])
        # data_120_at_six_floor_arrivals_per_minute.append(arrivals_per_minute)

        # summary, all_passengers_data, quartile_stats, arrivals_per_minute = run_elevator_simulation(data_180_at_six_floor, sim_pace=None, manager=Look, limitations=None, seed=i+5, prefix=f"data_180_at_six_floor_{i}")
        # data_180_at_six_floor_summary.append(summary)
        # data_180_at_six_floor_all_passengers_data.append(all_passengers_data)
        # data_180_at_six_floor_floor_quartile_stats.append(quartile_stats['floor'])
        # data_180_at_six_floor_total_quartile_stats.append(quartile_stats['total'])
        # data_180_at_six_floor_arrivals_per_minute.append(arrivals_per_minute)

        # -----------------------with faculty in-----------------------
        summary, all_passengers_data, quartile_stats, arrivals_per_minute = run_elevator_simulation(data_with_faculty_in, sim_pace=None, manager=Look, limitations=None, seed=i+5, prefix=f"data_with_faculty_in_{i}")
        data_with_faculty_in_summary.append(summary)
        data_with_faculty_in_all_passengers_data.append(all_passengers_data)
        data_with_faculty_in_floor_quartile_stats.append(quartile_stats['floor'])
        data_with_faculty_in_total_quartile_stats.append(quartile_stats['total'])
        data_with_faculty_in_arrivals_per_minute.append(arrivals_per_minute)

        summary, all_passengers_data, quartile_stats, arrivals_per_minute = run_elevator_simulation(data_with_faculty_in, sim_pace=None, manager=Look, limitations=limitations, seed=i+5, prefix=f"limit_data_with_faculty_in_{i}")
        limit_data_with_faculty_in_summary.append(summary)
        limit_data_with_faculty_in_all_passengers_data.append(all_passengers_data)
        limit_data_with_faculty_in_floor_quartile_stats.append(quartile_stats['floor']) 
        limit_data_with_faculty_in_total_quartile_stats.append(quartile_stats['total'])
        limit_data_with_faculty_in_arrivals_per_minute.append(arrivals_per_minute)

        summary, all_passengers_data, quartile_stats, arrivals_per_minute = run_elevator_simulation(data_with_faculty_in, sim_pace=None, manager=Look, limitations=privilege_limitations, seed=i+5, prefix=f"priviliged_data_with_faculty_in_{i}")
        priviliged_data_with_faculty_in_summary.append(summary)
        priviliged_data_with_faculty_in_all_passengers_data.append(all_passengers_data)
        priviliged_data_with_faculty_in_floor_quartile_stats.append(quartile_stats['floor'])
        priviliged_data_with_faculty_in_total_quartile_stats.append(quartile_stats['total'])
        priviliged_data_with_faculty_in_arrivals_per_minute.append(arrivals_per_minute)

    # ================================================================================================================
    # 在所有模拟完成后，计算蒙特卡罗平均值
    # data_60_results = calculate_monte_carlo_averages(data_60_at_six_floor_summary, data_60_at_six_floor_all_passengers_data, data_60_at_six_floor_arrivals_per_minute, "60_at_six_floor")
    # limit_data_60_results = calculate_monte_carlo_averages(limit_data_60_at_six_floor_summary, limit_data_60_at_six_floor_all_passengers_data, limit_data_60_at_six_floor_arrivals_per_minute, "limit_60_at_six_floor")
    # data_120_results = calculate_monte_carlo_averages(data_120_at_six_floor_summary, data_120_at_six_floor_all_passengers_data, data_120_at_six_floor_arrivals_per_minute, "120_at_six_floor") 
    # limit_data_120_results = calculate_monte_carlo_averages(limit_data_120_at_six_floor_summary, limit_data_120_at_six_floor_all_passengers_data, limit_data_120_at_six_floor_arrivals_per_minute, "limit_120_at_six_floor")
    # data_180_results = calculate_monte_carlo_averages(data_180_at_six_floor_summary, data_180_at_six_floor_all_passengers_data, data_180_at_six_floor_arrivals_per_minute, "180_at_six_floor")
    # limit_data_180_results = calculate_monte_carlo_averages(limit_data_180_at_six_floor_summary, limit_data_180_at_six_floor_all_passengers_data, limit_data_180_at_six_floor_arrivals_per_minute, "limit_180_at_six_floor")
    # limit_results = calculate_monte_carlo_averages(limit_summary, limit_all_passengers_data, limit_arrivals_per_minute, "with_limitations")
    # no_limit_results = calculate_monte_carlo_averages(no_limit_summary, no_limit_all_passengers_data, no_limit_arrivals_per_minute, "no_limitations")
    # data_with_faculty_in_results = calculate_monte_carlo_averages(data_with_faculty_in_summary, data_with_faculty_in_all_passengers_data, data_with_faculty_in_arrivals_per_minute, "data_with_faculty_in")
    limit_data_with_faculty_in_results = calculate_monte_carlo_averages(limit_data_with_faculty_in_summary, limit_data_with_faculty_in_all_passengers_data, limit_data_with_faculty_in_arrivals_per_minute, "limit_data_with_faculty_in")
    priviliged_data_with_faculty_in_results = calculate_monte_carlo_averages(priviliged_data_with_faculty_in_summary, priviliged_data_with_faculty_in_all_passengers_data, priviliged_data_with_faculty_in_arrivals_per_minute, "priviliged_data_with_faculty_in")

    # ================================================================================================================
    # 將quartile_stats合併成pandas dataframe並存進csv
    os.makedirs("quartile_stats", exist_ok=True)
    os.makedirs("quartile_stats/floor", exist_ok=True)
    os.makedirs("quartile_stats/total", exist_ok=True)

    # data_60_at_six_floor_floor_quartile_stats_df = pd.DataFrame(data_60_at_six_floor_floor_quartile_stats)
    # data_60_at_six_floor_total_quartile_stats_df = pd.DataFrame(data_60_at_six_floor_total_quartile_stats)
    # data_60_at_six_floor_floor_quartile_stats_df.to_csv("quartile_stats/floor/data_60_at_six_floor_floor_quartile_stats.csv", index=False)
    # data_60_at_six_floor_total_quartile_stats_df.to_csv("quartile_stats/total/data_60_at_six_floor_total_quartile_stats.csv", index=False)

    # limit_data_120_at_six_floor_floor_quartile_stats_df = pd.DataFrame(limit_data_60_at_six_floor_floor_quartile_stats)
    # limit_data_120_at_six_floor_total_quartile_stats_df = pd.DataFrame(limit_data_60_at_six_floor_total_quartile_stats)
    # limit_data_120_at_six_floor_floor_quartile_stats_df.to_csv("quartile_stats/floor/limit_data_120_at_six_floor_floor_quartile_stats.csv", index=False)
    # limit_data_120_at_six_floor_total_quartile_stats_df.to_csv("quartile_stats/total/limit_data_120_at_six_floor_total_quartile_stats.csv", index=False)

    # data_120_at_six_floor_floor_quartile_stats_df = pd.DataFrame(data_120_at_six_floor_floor_quartile_stats)
    # data_120_at_six_floor_total_quartile_stats_df = pd.DataFrame(data_120_at_six_floor_total_quartile_stats)
    # data_120_at_six_floor_floor_quartile_stats_df.to_csv("quartile_stats/floor/data_120_at_six_floor_floor_quartile_stats.csv", index=False)
    # data_120_at_six_floor_total_quartile_stats_df.to_csv("quartile_stats/total/data_120_at_six_floor_total_quartile_stats.csv", index=False)

    # limit_data_120_at_six_floor_floor_quartile_stats_df = pd.DataFrame(limit_data_120_at_six_floor_floor_quartile_stats)
    # limit_data_120_at_six_floor_total_quartile_stats_df = pd.DataFrame(limit_data_120_at_six_floor_total_quartile_stats)
    # limit_data_120_at_six_floor_floor_quartile_stats_df.to_csv("quartile_stats/floor/limit_data_120_at_six_floor_floor_quartile_stats.csv", index=False)
    # limit_data_120_at_six_floor_total_quartile_stats_df.to_csv("quartile_stats/total/limit_data_120_at_six_floor_total_quartile_stats.csv", index=False)

    # data_180_at_six_floor_floor_quartile_stats_df = pd.DataFrame(data_180_at_six_floor_floor_quartile_stats)
    # data_180_at_six_floor_total_quartile_stats_df = pd.DataFrame(data_180_at_six_floor_total_quartile_stats)
    # data_180_at_six_floor_floor_quartile_stats_df.to_csv("quartile_stats/floor/data_180_at_six_floor_floor_quartile_stats.csv", index=False)
    # data_180_at_six_floor_total_quartile_stats_df.to_csv("quartile_stats/total/data_180_at_six_floor_total_quartile_stats.csv", index=False)

    # limit_data_180_at_six_floor_floor_quartile_stats_df = pd.DataFrame(limit_data_180_at_six_floor_floor_quartile_stats)
    # limit_data_180_at_six_floor_total_quartile_stats_df = pd.DataFrame(limit_data_180_at_six_floor_total_quartile_stats)
    # limit_data_180_at_six_floor_floor_quartile_stats_df.to_csv("quartile_stats/floor/limit_data_180_at_six_floor_floor_quartile_stats.csv", index=False)
    # limit_data_180_at_six_floor_total_quartile_stats_df.to_csv("quartile_stats/total/limit_data_180_at_six_floor_total_quartile_stats.csv", index=False)

    # limit_floor_quartile_stats_df = pd.DataFrame(limit_floor_quartile_stats)
    # limit_total_quartile_stats_df = pd.DataFrame(limit_total_quartile_stats)
    # limit_floor_quartile_stats_df.to_csv("quartile_stats/floor/limit_floor_quartile_stats.csv", index=False)
    # limit_total_quartile_stats_df.to_csv("quartile_stats/total/limit_total_quartile_stats.csv", index=False)

    # no_limit_floor_quartile_stats_df = pd.DataFrame(no_limit_floor_quartile_stats)
    # no_limit_total_quartile_stats_df = pd.DataFrame(no_limit_total_quartile_stats)
    # no_limit_floor_quartile_stats_df.to_csv("quartile_stats/floor/no_limit_floor_quartile_stats.csv", index=False)
    # no_limit_total_quartile_stats_df.to_csv("quartile_stats/total/no_limit_total_quartile_stats.csv", index=False)


    # ================================================================================================================
    # 有教職員在的資料

    data_with_faculty_in_floor_quartile_stats_df = pd.DataFrame(data_with_faculty_in_floor_quartile_stats)
    data_with_faculty_in_total_quartile_stats_df = pd.DataFrame(data_with_faculty_in_total_quartile_stats)
    data_with_faculty_in_floor_quartile_stats_df.to_csv("quartile_stats/floor/data_with_faculty_in_floor_quartile_stats.csv", index=False)
    data_with_faculty_in_total_quartile_stats_df.to_csv("quartile_stats/total/data_with_faculty_in_total_quartile_stats.csv", index=False)

    limit_data_with_faculty_in_floor_quartile_stats_df = pd.DataFrame(limit_data_with_faculty_in_floor_quartile_stats)
    limit_data_with_faculty_in_total_quartile_stats_df = pd.DataFrame(limit_data_with_faculty_in_total_quartile_stats)
    limit_data_with_faculty_in_floor_quartile_stats_df.to_csv("quartile_stats/floor/limit_data_with_faculty_in_floor_quartile_stats.csv", index=False)
    limit_data_with_faculty_in_total_quartile_stats_df.to_csv("quartile_stats/total/limit_data_with_faculty_in_total_quartile_stats.csv", index=False)

    priviliged_data_with_faculty_in_floor_quartile_stats_df = pd.DataFrame(priviliged_data_with_faculty_in_floor_quartile_stats)
    priviliged_data_with_faculty_in_total_quartile_stats_df = pd.DataFrame(priviliged_data_with_faculty_in_total_quartile_stats)
    priviliged_data_with_faculty_in_floor_quartile_stats_df.to_csv("quartile_stats/floor/priviliged_data_with_faculty_in_floor_quartile_stats.csv", index=False)
    priviliged_data_with_faculty_in_total_quartile_stats_df.to_csv("quartile_stats/total/priviliged_data_with_faculty_in_total_quartile_stats.csv", index=False)



    # ================================================================================================================
    # 使用AnalysisPlotter生成所有模拟结果的综合分析图表
    print(f"\n=== Generating Comprehensive Analysis Plots for All Simulations ===")
    plotter = AnalysisPlotter(n_floors=13)
    
    # 为每个场景生成分析图表
    scenarios_data = [
        # ("data_60_at_six_floor", data_60_at_six_floor_all_passengers_data),
        # ("limit_data_60_at_six_floor", limit_data_60_at_six_floor_all_passengers_data),
        # ("data_120_at_six_floor", data_120_at_six_floor_all_passengers_data),
        # ("limit_data_120_at_six_floor", limit_data_120_at_six_floor_all_passengers_data),
        # ("data_180_at_six_floor", data_180_at_six_floor_all_passengers_data),
        # ("limit_data_180_at_six_floor", limit_data_180_at_six_floor_all_passengers_data),
        # ("with_limitations", limit_all_passengers_data),
        # ("no_limitations", no_limit_all_passengers_data),
        ("data_with_faculty_in", data_with_faculty_in_all_passengers_data),
        ("limit_data_with_faculty_in", limit_data_with_faculty_in_all_passengers_data),
        ("priviliged_data_with_faculty_in", priviliged_data_with_faculty_in_all_passengers_data)
    ]
    
    for scenario_name, all_passengers_data_list in scenarios_data:
        if not all_passengers_data_list:  # 跳过空的数据集
            continue
            
        print(f"\nGenerating analysis plots for scenario: {scenario_name}")
        
        # 生成该场景的综合分析（合并所有模拟的数据）
        print(f"  Generating comprehensive analysis for {scenario_name}...")
        
        # 合并所有模拟的乘客数据
        combined_data = {
            'completed': [],
            'moving': [],
            'waiting': [],
            'sim_time': all_passengers_data_list[0]['sim_time'] if all_passengers_data_list else 600  # 默认10分钟
        }
        
        for all_passengers_data in all_passengers_data_list:
            combined_data['completed'].extend(all_passengers_data['completed'])
            combined_data['moving'].extend(all_passengers_data['moving'])
            combined_data['waiting'].extend(all_passengers_data['waiting'])
        

        # 生成综合分析图表
        quartile_stats = plotter.generate_simulation_analysis_all_passengers(
            combined_data,
            f"{scenario_name}_combined",  # prefix作为第二个位置参数
            save_plots=True,
            show_plots=False
        )


        print(f"  ✓ Analysis plots generated for {scenario_name}")
        print(f"    - Individual simulations: {len(all_passengers_data_list)} sets")

        # ================================================================================================================
        # 將data combined_data存進csv
        # 1. 將 floor data 正確轉成 DataFrame（每列代表一個樓層）
        combined_floor_data_df = pd.DataFrame.from_dict(quartile_stats['floor'], orient='index')
        combined_floor_data_df.index.name = 'floor'  # 加上 index 名稱

        # 2. 將 total scalar dict 包成一列
        combined_total_data_df = pd.DataFrame([quartile_stats['total']])

        combined_floor_data_df.to_csv(f"quartile_stats/floor/{scenario_name}_combined_floor_data.csv", index=False)
        combined_total_data_df.to_csv(f"quartile_stats/total/{scenario_name}_combined_total_data.csv", index=False)


    # 比较不同场景
    print(f"\n=== Scenario Comparison (Monte Carlo Averages) ===")
    scenarios = [
        # ("60_at_six_floor", data_60_results),
        # ("limit_60_at_six_floor", limit_data_60_results),
        # ("120_at_six_floor", data_120_results), 
        # ("limit_120_at_six_floor", limit_data_120_results),
        # ("180_at_six_floor", data_180_results),
        # ("limit_180_at_six_floor", limit_data_180_results),
        # ("with_limitations", limit_results),
        # ("no_limitations", no_limit_results),
        ("data_with_faculty_in", data_with_faculty_in_all_passengers_data),
        ("limit_data_with_faculty_in", limit_data_with_faculty_in_all_passengers_data),
        ("priviliged_data_with_faculty_in", priviliged_data_with_faculty_in_all_passengers_data)
    ]
    
    print(f"{'Scenario':<20} {'Avg Served':<12} {'Avg Wait Time':<15} {'Avg Inside Time':<16} {'Avg Service Time':<16}")
    print("-" * 85)
    for name, results in scenarios:
        if results:  # 确保结果不为空
            print(f"{name:<20} {results['avg_served']:<12.1f} {results['avg_waiting_time']:<15.1f} "
                  f"{results['avg_inside_time']:<16.1f} {results['avg_service_time']:<16.1f}")
            
