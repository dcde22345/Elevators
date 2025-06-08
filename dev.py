import sys
from MyTools import *
from algorithm.DDSAlgorithm import DDSManager
from algorithm.LookAlgorithm import Look
from ElevatorSimulator import Simulator


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

    # Calculate arrival rates based on row sums of data matrix
    # Each row represents passengers starting from that floor
    # Data is for 10 minutes, so convert to arrivals per second
    floor_arrival_rates = [sum(row)/(10*60) for row in data]  # 10 minutes = 600 seconds

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
        floor_arrival_rates=floor_arrival_rates,  # 使用樓層特定的到達率
        destination_probabilities=destination_probabilities,  # 使用機率矩陣
        manager=DDSManager,
        debug_mode=True
    )

    x.generate_scenario()
    summary = x.run_simulation()
    
    # 輸出統計結果
    served_passengers = summary['passengers']['served']
    print(f"\n=== Elevator system performance statistics ===")
    print(f"Served passengers: {served_passengers}")
    print(f"Total waiting time: {summary['info']['total_waiting_time']:.2f}s")
    print(f"Total service time: {summary['info']['total_inside_time']:.2f}s")
    
    if served_passengers > 0:
        avg_waiting_time = summary['info']['waiting_time'][1]  # 平均等待時間
        avg_inside_time = summary['info']['inside_time'][1]    # 平均乘坐時間
        avg_service_time = summary['goals']['service_time'][1] # 平均服務時間
        
        print(f"Average waiting time: {avg_waiting_time:.2f}s")
        print(f"Average service time: {avg_inside_time:.2f}s")
        print(f"Average time in the system time: {avg_service_time:.2f}s")
        print(f"Verification: {avg_service_time:.2f} ≈ {avg_waiting_time:.2f} + {avg_inside_time:.2f} = {avg_waiting_time + avg_inside_time:.2f}")
    else:
        print("沒有乘客完成服務")