from MyTools import *
import ElevatorManager
from Elevator import Elevator
from Passenger import Arrival, Passenger
from SimulationPlotter import SimPlotter
from algorithm.LookAlgorithm import Look
import matplotlib.pyplot as plt
import numpy as np
import sys, pprint
from warnings import warn
from typing import List
from collections import defaultdict


class Simulator:
    '''
    Simulator
    This class implements a simulation of elevators which serve arriving passengers.

    Involved classes:
    Simulator       = simulation manager
    ElevatorManager = decision maker (very naive algorithm intended to be inherited
                      by smarter managers which would overwrite its methods)
    Elevator        = represent the elevators in the simulation
    Arrival         = encode a single event of arrival of passengers
    Passenger       = encode and track a single passenger
    SimPlot         = visualize the simulation dynamically via matplotlib

    Main flow:
    generate_scenario()
    run_simulation()
        sim_initialize()
            Manager.handle_initialization()
            update_missions()
        loop: handle_next_event()
            update the state to the time of the next event: sim_update()
            handle next event: handle_arrival() / end_mission() / sim_end()
                Manager.handle_arrival() / Manager.handle_no_missions()
                update_missions()

    Parameters:
        manager (ElevatorManager): 電梯管理演算法，預設為NaiveManager
        debug_mode (bool): 是否開啟除錯模式，預設為False
        limitations (list): 電梯樓層限制，預設為None
        verbose (bool): 是否顯示詳細資訊，預設為True
        sim_len (int): 模擬時長(秒)，預設為120秒
        sim_pace (float): 視覺化速度，None為不視覺化，預設為None
        time_resolution (float): 時間解析度(秒)，預設為0.5秒
        logfile (str): 日誌檔案路徑，預設為None
        seed (int): 隨機種子，預設為1
        n_floors (int): 樓層數，預設為3層
        n_elevators (int): 電梯數量，預設為2台
        capacity (int): 電梯容量，預設為15人
        speed (float): 電梯速度(樓層/秒)，預設為1
        open_time (int): 開關門時間(秒)，預設為2秒
        floor_arrival_rates (list): 各樓層乘客到達頻率，預設為None
        destination_probabilities (list): 目的地機率矩陣，預設為None
        size (float): 乘客大小，預設為1.5
        delay (int): 延遲時間(秒)，預設為3秒

        - 在floor_arrival_rates為None時，有arrival_pace時才會啟用
        arrival_pace (float): 乘客到達頻率(人/秒)，預設為0.1
        p_between (float): 樓層間移動機率，預設為0.1
        p_up (float): 向上移動機率，預設為0.45

    TODO:
    1. add acceleration_time in addition to open_time (for stops w/o opening)

    Written by Ido Greenberg, 2018
    '''

    def __init__(self, manager=ElevatorManager.NaiveManager, debug_mode=False, limitations=None, verbose=True,
                 sim_len=120, sim_pace=None, time_resolution=0.5, logfile=None, seed=1,
                 n_floors=3, n_elevators=2, capacity=15, speed=1, open_time=2, 
                 arrival_pace=1/10, floor_arrival_rates=None, destination_probabilities=None, p_between=0.1, p_up=0.45, size=1.5, delay=3):

        # Note: default time unit is seconds.
        self.debug = debug_mode
        self.verbose = verbose

        ## Simulation world
        # conf
        self.sim_len = sim_len
        sim_pace = sim_pace # visualization's fast-forward; None for no visualization
        self.time_resolution = np.inf if sim_pace is None else time_resolution
        self.logfile = logfile
        self.seed = seed
        # init
        self.sim_time = 0
        self.real_time = None
        self.logger = None
        # stats init
        self.useless_opens = 0
        self.blocked_entrances = 0
        self.moves_without_open = 0

        ## Elevators
        # conf
        self.n_floors = n_floors
        self.n_elevators = n_elevators
        el_capacity = capacity
        el_speed = speed # floors per second. Note: acceleration is neglected
        el_open_time = open_time # = time to open = time to close
        # init
        self.el_next_event_time = [np.inf for _ in range(self.n_elevators)]
        
        # if limitations is not None, use the limitations
        if limitations:
            self.el = [
                Elevator(i, self.n_floors, el_capacity, el_speed, el_open_time, limitations[i])
                for i in range(self.n_elevators)
            ]
        else:
            self.el = [Elevator(i, self.n_floors, el_capacity, el_speed, el_open_time)
                       for i in range(self.n_elevators)]
        self.sim_plot = SimPlotter(self.n_floors, self.n_elevators, sim_pace) \
            if sim_pace is not None else None

        ## Passengers
        # conf
        self.arrivals_pace = arrival_pace # arrivals per second (global, for backward compatibility)
        # NEW: floor-specific arrival rates (arrivals per second per floor)
        if floor_arrival_rates is not None:
            if len(floor_arrival_rates) != n_floors:
                raise ValueError(f"floor_arrival_rates must have {n_floors} elements (floors 0 to {n_floors-1})")
            self.floor_arrival_rates = floor_arrival_rates
        else:
            # Use global arrival_pace for all floors if not specified
            self.floor_arrival_rates = None
            
        # Store destination probabilities matrix
        self.destination_probabilities = destination_probabilities
            
        self.p_go_between = p_between
        self.p_go_up = p_up
        self.p_go_down = 1 - (self.p_go_between + self.p_go_up)
        self.arrival_size = size # mean number of passengers per arrival
        self.delay = delay # mean delay on passengers entrance
        # init
        self.scenario = []
        self.future_arrivals = []
        self.waiting_passengers = []
        self.moving_passengers = [[] for _ in range(self.n_elevators)]
        self.completed_passengers = []

        ## Manager
        self.manager_info = manager.version_info()
        self.manager = manager(self.n_floors, self.el,
                                       el_capacity, el_speed, el_open_time,
                                       self.arrivals_pace,
                                       self.p_go_up, self.p_go_down, self.p_go_between,
                                       self.arrival_size, self.delay)

    def _can_serve_passenger(self, xi: int, xf: int) -> bool:
        """檢查是否有任何一台電梯能夠服務這個乘客請求"""
        for el in self.el:
            if el.can_go_to(xi) and el.can_go_to(xf):
                return True
        return False

    def _get_valid_destinations_for_floor(self, floor: int) -> List[int]:
        """獲得指定樓層的所有可達目的地"""
        valid_destinations = []
        for dest_floor in range(self.n_floors + 1):
            if dest_floor != floor and self._can_serve_passenger(floor, dest_floor):
                valid_destinations.append(dest_floor)
        return valid_destinations

    def generate_scenario(self, verbose=None):

        if verbose is None: verbose = self.debug
        if self.seed is not None: np.random.seed(self.seed)

        all_arrivals = []
        
        if self.floor_arrival_rates is not None:
            # Use floor-specific arrival rates
            if self.verbose:
                print("Using floor-specific arrival rates:")
                for floor, rate in enumerate(self.floor_arrival_rates):
                    print(f"  Floor {floor}: {rate:.3f} arrivals/sec")
            
            for floor in range(self.n_floors):
                if self.floor_arrival_rates[floor] > 0:
                    # Generate arrivals for this specific floor
                    n_arrivals_floor = np.random.poisson(self.sim_len * self.floor_arrival_rates[floor])
                    
                    if n_arrivals_floor > 0:
                        a_times = list(np.sort(self.sim_len * np.random.rand(n_arrivals_floor)))
                        a_sizes = list(np.floor(1+np.random.exponential(self.arrival_size-1, n_arrivals_floor)).astype(int))
                        a_delays = list(np.random.gamma(self.delay/8, 8, n_arrivals_floor))
                        
                        # For floor-specific arrivals, set starting floor and determine destinations
                        for i in range(n_arrivals_floor):
                            # 首先檢查這個樓層是否有可達的目的地
                            valid_destinations_for_floor = self._get_valid_destinations_for_floor(floor)
                            if not valid_destinations_for_floor:
                                # 這個樓層沒有任何可達的目的地，跳過這個arrival
                                continue
                                
                            # Use destination probabilities if available
                            if self.destination_probabilities is not None:
                                target_probs = self.destination_probabilities[floor]
                                # Normalize probabilities and handle zero sum case
                                prob_sum = sum(target_probs)
                                if prob_sum > 0:
                                    normalized_probs = [p/prob_sum for p in target_probs]
                                    # Remove self-destination and unreachable destinations
                                    valid_destinations = []
                                    valid_probs = []
                                    for dest_floor, prob in enumerate(normalized_probs):
                                        if dest_floor != floor and prob > 0 and dest_floor in valid_destinations_for_floor:
                                            valid_destinations.append(dest_floor)
                                            valid_probs.append(prob)
                                    
                                    if valid_destinations:
                                        # Renormalize after removing invalid destinations
                                        prob_sum = sum(valid_probs)
                                        if prob_sum > 0:
                                            valid_probs = [p/prob_sum for p in valid_probs]
                                            a_to = np.random.choice(valid_destinations, p=valid_probs)
                                        else:
                                            # Fallback: choose randomly from valid destinations
                                            a_to = np.random.choice(valid_destinations_for_floor)
                                    else:
                                        # Fallback: choose randomly from valid destinations
                                        a_to = np.random.choice(valid_destinations_for_floor)
                                else:
                                    # Fallback: choose randomly from valid destinations
                                    a_to = np.random.choice(valid_destinations_for_floor)
                            else:
                                # No probability matrix available, choose randomly from valid destinations
                                a_to = np.random.choice(valid_destinations_for_floor)
                            
                            arrival = Arrival(a_times[i], a_sizes[i], a_delays[i], floor, a_to)
                            all_arrivals.append(arrival)
            
            
            
            # Sort all arrivals by time
            all_arrivals.sort(key=lambda x: x.t)
            self.scenario = tuple(all_arrivals)
            
        else:
            # Use original global arrival_pace method for backward compatibility
            n_arrivals = np.random.poisson(self.sim_len * self.arrivals_pace)

            a_times = list(np.sort(self.sim_len * np.random.rand(n_arrivals)))
            a_sizes = list(np.floor(1+np.random.exponential(self.arrival_size-1,n_arrivals)).astype(int))
            a_delays = list(np.random.gamma(self.delay/8,8,n_arrivals))
            #list(np.random.lognormal(0,np.sqrt(self.delay),n_arrivals))
            a_types = list(np.random.choice(['up', 'down', 'between'], n_arrivals, True,
                                                   (self.p_go_up, self.p_go_down, self.p_go_between)))
            a_from = [(0 if tp=='up' else int(1+self.n_floors*np.random.rand()))
                      for tp in a_types]
            
            # 修改目的地生成邏輯，考慮電梯限制
            a_to = []
            valid_indices = []  # 記錄有效的索引
            for i, tp in enumerate(a_types):
                from_floor = a_from[i]
                valid_destinations = self._get_valid_destinations_for_floor(from_floor)
                
                if not valid_destinations:
                    # 如果這個起始樓層沒有可達目的地，跳過這個arrival
                    continue
                    
                valid_indices.append(i)  # 記錄這個索引是有效的
                
                if tp == 'down':
                    # 只選擇向下的目的地
                    down_destinations = [dest for dest in valid_destinations if dest < from_floor]
                    if down_destinations:
                        a_to.append(np.random.choice(down_destinations))
                    else:
                        # 沒有向下的可達目的地，隨機選擇一個可達目的地
                        a_to.append(np.random.choice(valid_destinations))
                elif tp == 'up':
                    # 只選擇向上的目的地
                    up_destinations = [dest for dest in valid_destinations if dest > from_floor]
                    if up_destinations:
                        a_to.append(np.random.choice(up_destinations))
                    else:
                        # 沒有向上的可達目的地，隨機選擇一個可達目的地
                        a_to.append(np.random.choice(valid_destinations))
                else:  # 'between'
                    # 隨機選擇任何可達目的地
                    a_to.append(np.random.choice(valid_destinations))

            # 根據有效索引重新組織所有陣列
            if valid_indices:
                a_from = [a_from[i] for i in valid_indices]
                a_times = [a_times[i] for i in valid_indices]
                a_sizes = [a_sizes[i] for i in valid_indices]
                a_delays = [a_delays[i] for i in valid_indices]
            else:
                # 沒有有效的arrival，清空所有陣列
                a_from = []
                a_times = []
                a_sizes = []
                a_delays = []
                a_to = []

            self.scenario = tuple([Arrival(t,n,d,xi,xf)
                             for (t,n,d,xi,xf) in zip(a_times,a_sizes,a_delays,a_from,a_to)])

        if verbose:
            print(f"\nGenerated {len(self.scenario)} arrival events:")
            for i,arr in enumerate(self.scenario):
                arr.print(i)
        
        # 使用 AnalysisPlotter 生成乘客流量分析圖表
        if len(self.scenario) > 0:
            try:
                from AnalysisPlotter import AnalysisPlotter
                
                # 創建分析繪圖器
                plotter = AnalysisPlotter(n_floors=self.n_floors)
                
                # 分析數據並生成圖表
                flow_matrix, arrivals_per_minute = plotter.analyze_passenger_flow(self.scenario)
                
                print(f"\n=== 乘客流量分析 ===")
                print(f"場景總乘客數: {int(flow_matrix.sum())}")
                print(f"模擬時長: {max(arrivals_per_minute.keys()) + 1} 分鐘" if arrivals_per_minute else "模擬時長: 0 分鐘")
                
                # 生成並保存圖表
                print("正在生成乘客流量熱力圖...")
                plotter.plot_passenger_flow_heatmap(
                    flow_matrix, 
                    save_path="images/passenger_flow_heatmap.png",
                    show_plot=False
                )
                
                print("正在生成每分鐘抵達人數直方圖...")
                plotter.plot_arrivals_per_minute_histogram(
                    arrivals_per_minute, 
                    save_path="arrivals_per_minute_histogram.png",
                    show_plot=False
                )
                
                # 打印詳細統計信息
                plotter._print_detailed_statistics(flow_matrix, arrivals_per_minute)
                
                print("\n圖表已保存:")
                print("- passenger_flow_heatmap.png (乘客流量熱力圖)")
                print("- arrivals_per_minute_histogram.png (每分鐘抵達人數直方圖)")
                
            except ImportError:
                print("\n警告: 無法導入 AnalysisPlotter，跳過圖表生成")
            except Exception as e:
                print(f"\n警告: 圖表生成失敗: {str(e)}")
        else:
            print("\n場景中沒有乘客抵達事件，跳過圖表生成")

    def _fallback_destination(self, floor):
        """Fallback logic for destination selection when probability matrix is not available or invalid"""
        if floor == 0:
            # Ground floor - only go up
            return int(1 + self.n_floors * np.random.rand())
        elif floor == self.n_floors:
            # Top floor - only go down
            return int(self.n_floors * np.random.rand())
        else:
            # Middle floors - can go up or down
            if np.random.rand() < 0.5:
                # Go up
                return int(floor + 1 + (self.n_floors - floor) * np.random.rand())
            else:
                # Go down  
                return int(floor * np.random.rand())

    def run_simulation(self, prefix):
        self.sim_initialize()
        end_sim = False
        while not end_sim:
            end_sim = self.handle_next_event()
        
        # 在模擬完成後生成額外的分析圖表
        try:
            from AnalysisPlotter import AnalysisPlotter
            
            # 創建分析繪圖器
            plotter = AnalysisPlotter(n_floors=self.n_floors)
            
            # 準備所有乘客的數據
            all_passengers_data = {
                'completed': self.completed_passengers,
                'moving': [ps for ps_list in self.moving_passengers for ps in ps_list],
                'waiting': self.waiting_passengers,
                'sim_time': self.sim_time
            }
            
            # 生成模擬結果分析（包含所有乘客）
            quartile_stats=plotter.generate_simulation_analysis_all_passengers(
                all_passengers_data,
                prefix=prefix,
                save_plots=True,
                show_plots=False
            )
            
        except ImportError:
            print("\n警告: 無法導入 AnalysisPlotter，跳過模擬結果分析")
        except Exception as e:
            print(f"\n警告: 模擬結果分析失敗: {str(e)}")
        
        return end_sim, all_passengers_data, quartile_stats

    def sim_initialize(self):
        if self.verbose: print("\n\nSIMULATION BEGAN\n")
        self.logger = open(self.logfile, 'w') if self.logfile else None

        # world
        self.sim_time = 0
        
        # arrivals
        self.future_arrivals = list(self.scenario)
        self.waiting_passengers = []
        self.moving_passengers = [[] for _ in range(self.n_elevators)]
        self.completed_passengers = []
        
        # elevators
        for el in self.el:
            el.initialize()
        
        # manager
        self.manager.initialize()
        missions = self.manager.handle_initialization()
        self.update_missions(missions)
        
        # stats
        self.useless_opens = 0
        self.blocked_entrances = 0
        self.moves_without_open = 0
        
        # visualization
        if self.sim_plot is not None:
            self.sim_plot.initialize()
        self.real_time = PrintTime()

    def handle_next_event(self):
        # Find next event's time
        t_arrival = self.future_arrivals[0].t if self.future_arrivals else np.inf
        t_finish_mission = min(self.el_next_event_time)
        t_forced_update = self.sim_time + self.time_resolution
        t = min(t_forced_update, t_arrival, t_finish_mission, self.sim_len)

        # update simulation to next event's time
        dt = self.sim_update(t)

        # handle next event
        if t_forced_update < min(self.sim_len, t_arrival, t_finish_mission):
            pass
        elif self.sim_len < min(t_arrival, t_finish_mission):
            self.update_plot(dt)
            summary = self.sim_end()
            return summary
        elif t_arrival < t_finish_mission:
            self.handle_arrival()
        else:
            self.end_mission()

        self.update_plot(dt)

        return 0

    def sim_update(self, t):
        dt = t - self.sim_time
        assert(dt>=0), dt
        if dt==0:
            return dt
        # update elevators location
        for el in self.el:
            if el.missions and el.missions[0] is not None:
                el.x += np.sign(el.missions[0]-el.x) * el.speed * dt
                el.total_distance += el.speed * dt
        self.sim_time = t
        return dt

    def handle_arrival(self):
        a = self.future_arrivals[0]
        if self.debug:
            self.log("Arrive", f"n={a.n:d}\t({a.d:.1f})\t{a.xi:d} -> {a.xf:d}")
        del(self.future_arrivals[0])
        new_passengers = [Passenger(a) for _ in range(a.n)]
        missions = self.manager.handle_arrival(self.sim_time, a.xi, a.xf)
        if self.debug:
            print(missions)
        if -1 in missions:
            for ps in new_passengers:
                ps.assigned_el = missions[-1]
        self.waiting_passengers.extend(new_passengers)
        self.update_missions(missions)

    def end_mission(self):
        i_el = int(np.argmin(self.el_next_event_time))
        el = self.el[i_el]
        
        # Check if there are any missions to process
        if not el.missions:
            # No missions left, handle this case
            missions = self.manager.handle_no_missions(self.sim_time, i_el)
            if self.debug:
                print(f"No missions for elevator {i_el}, getting new missions:", missions)
            if not i_el in missions or not missions[i_el]:
                self.el_next_event_time[i_el] = np.inf
                el.sleep()
            else:
                self.update_missions(missions)
            return
        
        m = el.missions[0]
        del(el.missions[0])

        # end previous mission
        if m is not None:
            assert_zero(m-el.x, eps=1e-10)
            el.x = m
            if self.debug:
                self.log("Moved", f"#{i_el:02d}\t-> {el.x:d}")
            # detect move without open
            if el.missions and el.missions[0] is not None:
                self.moves_without_open += 1

        # begin new mission
        if not el.missions:
            missions = self.manager.handle_no_missions(self.sim_time, i_el)
            if self.debug:
                print(missions)
            if not i_el in missions or not missions[i_el]:
                self.el_next_event_time[i_el] = np.inf
                el.sleep()
            self.update_missions(missions)
        elif el.missions[0] is None:
            # 檢查是否需要在這個樓層開門
            if self._check_need_to_open(i_el):
                delay = self.open_el(i_el)
                self.el_next_event_time[i_el] = self.sim_time + 2*el.open_time + delay
                el.open()
            else:
                # 不需要開門，跳過這個開門任務
                del el.missions[0]
                if self.debug: 
                    self.log("Skip Open", f"#{i_el:02d} at floor {el.x} - no passengers to serve")
                
                # 繼續處理下一個任務（避免遞歸調用）
                while el.missions and el.missions[0] is None:
                    # 如果還有連續的開門任務也跳過
                    del el.missions[0]
                    if self.debug:
                        self.log("Skip Open", f"#{i_el:02d} at floor {el.x} - consecutive open task skipped")
                
                if el.missions:
                    # 還有任務，立即執行下一個
                    if el.missions[0] is not None:
                        # 下一個是移動任務
                        self.el_next_event_time[i_el] = self.sim_time + abs(el.missions[0]-el.x)/el.speed
                        el.move()
                        if self.debug:
                            self.log("Moving", f"#{i_el:02d}\t-> {el.missions[0]:d}")
                    else:
                        # 下一個又是開門任務，再次檢查
                        if self._check_need_to_open(i_el):
                            delay = self.open_el(i_el)
                            self.el_next_event_time[i_el] = self.sim_time + 2*el.open_time + delay
                            el.open()
                        else:
                            # 設置為立即重新檢查
                            self.el_next_event_time[i_el] = self.sim_time
                else:
                    # 沒有更多任務了，進入空閒處理
                    missions = self.manager.handle_no_missions(self.sim_time, i_el)
                    if self.debug:
                        print(missions)
                    if not i_el in missions or not missions[i_el]:
                        self.el_next_event_time[i_el] = np.inf
                        el.sleep()
                    self.update_missions(missions)

        else:
            self.el_next_event_time[i_el] = self.sim_time+abs(el.missions[0]-el.x)/el.speed
            el.move()
            for ps in self.moving_passengers[i_el]:
                if el.motion != np.sign(ps.xf-el.x):
                    ps.indirect_motion += 1
                    if self.debug:
                        self.log("INDIRECT", f"{el.motion:d} != {el.x:d}->{ps.xf:d}")
            if self.debug:
                self.log("Moving", f"#{i_el:02d}\t-> {el.missions[0]:d}")

    def _check_need_to_open(self, i_el):
        """
        檢查電梯是否需要在當前樓層開門
        條件：
        1. 有乘客要在這層下車，或
        2. 有等待的乘客要上車且符合條件：
           - 在同一樓層
           - 方向匹配或電梯靜止
           - 分配給這台電梯或方向匹配的可搭乘
        3. 有乘客無法到達目的樓層（因為電梯limitations），需要下車轉乘
        """
        el = self.el[i_el]
        current_floor = int(el.x)  # 確保是整數樓層
        
        # 檢查1：有乘客要下車？
        passengers_exiting = [ps for ps in self.moving_passengers[i_el] if ps.xf == current_floor]
        has_exit = len(passengers_exiting) > 0
        
        # 檢查3：有乘客因為limitations無法到達目的樓層？
        stranded_passengers = [ps for ps in self.moving_passengers[i_el] 
                              if not el.can_go_to(ps.xf)]
        has_stranded = len(stranded_passengers) > 0
        
        # 檢查2：有乘客要上車？
        passengers_waiting_here = [ps for ps in self.waiting_passengers if ps.xi == current_floor]
        has_entry = False
        
        for ps in passengers_waiting_here:
            passenger_direction = 1 if ps.xf > ps.xi else -1  # 1=up, -1=down
            
            # 檢查方向是否匹配
            direction_compatible = (el.motion == 0 or el.motion == passenger_direction)
            
            # 檢查是否分配給這台電梯，或者方向匹配可以搭乘
            assigned_to_this = (ps.assigned_el == i_el)
            can_board = direction_compatible and (assigned_to_this or ps.assigned_el == -1)
            
            # 檢查電梯是否有空間
            has_capacity = len(self.moving_passengers[i_el]) < el.capacity
            
            if can_board and has_capacity:
                has_entry = True
                break
        
        # Debug信息
        if self.debug and (has_exit or has_entry or has_stranded):
            exit_count = len(passengers_exiting) if has_exit else 0
            stranded_count = len(stranded_passengers) if has_stranded else 0
            entry_count = len([ps for ps in passengers_waiting_here 
                              if ps.assigned_el == i_el or ps.assigned_el == -1]) if has_entry else 0
            self.log("Open Check", f"#{i_el:02d} floor {current_floor}: exit={exit_count}, stranded={stranded_count}, entry={entry_count}")
        elif self.debug:
            self.log("Open Check", f"#{i_el:02d} floor {current_floor}: no passengers to serve")
        
        return has_exit or has_entry or has_stranded

    def update_missions(self, missions):
        '''
        Get manager missions per elevator in format:
          (destination, whether_to_open, mission_to_split/remove)
        convert to elevator missions in format:
          n for moving to floor n, None for opening and loading.
        and add to elevators missions lists.
        '''
        for i_el in missions:
            if i_el == -1: continue # elevator assignment rather than mission
            el = self.el[i_el]
            immediate_mission = missions[i_el] and not el.missions

            # Separate removal missions and process them in reverse order to avoid index shifting
            removal_missions = []
            other_missions = []
            
            for m in missions[i_el]:
                if m[0] is None:
                    # remove mission m: (None, *, m)
                    removal_missions.append(m)
                else:
                    other_missions.append(m)
            
            # Sort removal missions by index in descending order to avoid index shifting issues
            removal_missions.sort(key=lambda x: x[2], reverse=True)
            
            # Process removal missions first
            for m in removal_missions:
                if m[2] < 0 or m[2] >= len(el.missions):
                    warn(f"Skip removing mission: index {m[2]} out of range for elevator {i_el} (missions={el.missions})")
                    continue
                del(el.missions[m[2]])
                if m[2]==0: immediate_mission = True
            
            # Process other missions
            for m in other_missions:
                if m[2]==-1:
                    # new mission: (destination floor, open/not, -1)
                    el.missions.append(m[0])
                    if m[1]: el.missions.append(None)
                else:
                    # split existing mission m: (destination floor, open/not, m)
                    if m[2] < 0 or m[2] > len(el.missions):
                        warn(f"Skip splitting mission: index {m[2]} out of range for elevator {i_el} (missions={el.missions})")
                        continue
                    if m[2] < len(el.missions) and el.missions[m[2]] is None:
                        warn("Trying to split an OPEN mission.")
                        continue
                    el.missions.insert(m[2], m[0])
                    if m[1]: el.missions.insert(m[2]+1, None)
                    if m[2]==0: immediate_mission = True

            if immediate_mission:
                if el.missions and el.missions[0] is None:
                    warn("Unexpected mission assignment: open which does not follow motion.")
                    delay = self.open_el(i_el)
                    self.el_next_event_time[i_el] = self.sim_time + 2*el.open_time + delay
                    el.open()
                elif el.missions:
                    self.el_next_event_time[i_el] = self.sim_time+abs(el.missions[0]-el.x)/el.speed
                    el.move()

        if self.debug:
            print([el.missions for el in self.el])

    def open_el(self, i):
        el = self.el[i]
        any_activity = False
        blocked_entrance = False

        # exiting passengers (normal destination reached)
        picked_up = []
        for j,ps in enumerate(self.moving_passengers[i]):
            if ps.xf == el.x:
                any_activity = True
                picked_up.append(j)
                self.completed_passengers.append(ps)
                ps.t2 = self.sim_time
                if self.debug:
                    self.log("Exit", f"#{i:02d}\tt={ps.t2-ps.t0:.0f}s")
        for j in sorted(picked_up, reverse=True):
            del self.moving_passengers[i][j]

        # handle stranded passengers (cannot reach destination due to limitations)
        stranded_picked_up = []
        for j,ps in enumerate(self.moving_passengers[i]):
            if not el.can_go_to(ps.xf):
                any_activity = True
                stranded_picked_up.append(j)
                # Re-add passenger to waiting list with updated starting floor
                ps.xi = int(el.x)  # Update starting floor to current elevator position
                ps.assigned_el = -1  # Reset assignment
                self.waiting_passengers.append(ps)
                # Re-request elevator service for this passenger
                missions = self.manager.handle_arrival(self.sim_time, ps.xi, ps.xf)
                if -1 in missions: ps.assigned_el = missions[-1]
                self.update_missions(missions)
                if self.debug:
                    self.log("Stranded", f"#{i:02d} passenger {ps.xi}->{ps.xf} offloaded - elevator cannot reach destination")
        for j in sorted(stranded_picked_up, reverse=True):
            del self.moving_passengers[i][j]

        # entering passengers
        delay = 0
        picked_up = []
        for j,ps in enumerate(self.waiting_passengers):
            # Check if passenger is on the same floor as the elevator
            if ps.xi == el.x:
                # Check if elevator direction matches passenger's desired direction
                passenger_direction = 1 if ps.xf > ps.xi else -1  # 1 for up, -1 for down
                
                # Only allow passengers to board if elevator direction matches their need
                # or if elevator is stationary (motion == 0)
                if el.motion != 0 and el.motion != passenger_direction:
                    # Passenger wants different direction, skip this elevator
                    continue
                
                # If passenger was assigned to a different elevator, update assignment
                if ps.assigned_el != i:
                    ps.assigned_el = i
                    if self.debug: self.log("Reassigned", f"Passenger {ps.xi}->{ps.xf} switched to available elevator #{i}")
                
                if el.capacity <= len(self.moving_passengers[i]):
                    # Elevator is full - count block and re-push the button
                    blocked_entrance = True
                    missions = self.manager.handle_arrival(self.sim_time, ps.xi, ps.xf)
                    if -1 in missions: ps.assigned_el = missions[-1]
                    else: ps.assigned_el = -1
                    self.update_missions(missions)
                    if self.debug: self.log("Blocked", f"#{i:02d}")
                    continue
                any_activity = True
                picked_up.append(j)
                self.moving_passengers[i].append(ps)
                ps.t1 = self.sim_time
                delay = max(delay, ps.d)
                if self.debug: self.log("Enter", f"#{i:02d}\tt={ps.t1-ps.t0:.0f}s direction={'up' if passenger_direction == 1 else 'down'}")
        
        # remove passengers from waiting list
        for j in sorted(picked_up, reverse=True):
            del self.waiting_passengers[j]

        if blocked_entrance: self.blocked_entrances += 1

        # count useless opens
        if not any_activity:
            self.useless_opens += 1
            if self.debug: self.log("USELESS", f"#{i:02d}")

        return delay

    def sim_end(self, verbose=None):
        if verbose is None: verbose = self.verbose
        if self.verbose: print("\nSIMULATION FINISHED\n\n")
        # classify passengers
        n_ps_scenario = sum([a.n for a in self.scenario])
        n_ps_completed = len(self.completed_passengers)
        n_ps_moving = sum([len(ps) for ps in self.moving_passengers])
        n_ps_waiting = len(self.waiting_passengers)
        n_ps_future = len(self.future_arrivals)
        assert(n_ps_future == 0), n_ps_future
        assert(n_ps_scenario == n_ps_completed + n_ps_moving + n_ps_waiting)
        moving_max_time = max([self.sim_time-ps.t1 for ps_list in self.moving_passengers for ps in ps_list]) \
            if n_ps_moving else 0
        waiting_max_time = max([self.sim_time-ps.t0 for ps in self.waiting_passengers]) \
            if n_ps_waiting else 0
        
        # Print passenger status
        if verbose:
            print(f"Passenger status at simulation end:")
            print(f"  Waiting: {n_ps_waiting} passengers")
            print(f"  Inside elevators: {n_ps_moving} passengers") 
            print(f"  Served: {n_ps_completed} passengers")
            print(f"  Total arrivals: {n_ps_scenario} passengers")
        # Calculate waiting_time for ALL passengers
        all_waiting_times = []
        # 1. Completed passengers: t1 - t0
        all_waiting_times.extend([ps.t1 - ps.t0 for ps in self.completed_passengers])
        # 2. Moving passengers: t1 - t0
        for ps_list in self.moving_passengers:
            all_waiting_times.extend([ps.t1 - ps.t0 for ps in ps_list])
        # 3. Waiting passengers: sim_time - t0 (假設現在進入電梯)
        all_waiting_times.extend([self.sim_time - ps.t0 for ps in self.waiting_passengers])
        
        # Calculate inside_time for relevant passengers
        all_inside_times = []
        # 1. Completed passengers: t2 - t1
        all_inside_times.extend([ps.t2 - ps.t1 for ps in self.completed_passengers])
        # 2. Moving passengers: sim_time - t1 (假設現在離開電梯)
        for ps_list in self.moving_passengers:
            all_inside_times.extend([self.sim_time - ps.t1 for ps in ps_list])
        
        # Calculate service_time for ALL passengers
        all_service_times = []
        # 1. Completed passengers: t2 - t0
        all_service_times.extend([ps.t2 - ps.t0 for ps in self.completed_passengers])
        # 2. Moving passengers: sim_time - t0
        for ps_list in self.moving_passengers:
            all_service_times.extend([self.sim_time - ps.t0 for ps in ps_list])
        # 3. Waiting passengers: sim_time - t0
        all_service_times.extend([self.sim_time - ps.t0 for ps in self.waiting_passengers])
        
        # 計算按樓層分組的等待時間統計
        floor_waiting_stats = {}
        floor_waiting_times_dict = defaultdict(list)
        
        # 1. Completed passengers: t1 - t0
        for ps in self.completed_passengers:
            waiting_time = ps.t1 - ps.t0
            floor_waiting_times_dict[ps.xi].append(waiting_time)
        
        # 2. Moving passengers: t1 - t0  
        for ps_list in self.moving_passengers:
            for ps in ps_list:
                waiting_time = ps.t1 - ps.t0
                floor_waiting_times_dict[ps.xi].append(waiting_time)
        
        # 3. Waiting passengers: sim_time - t0 (假設現在進入電梯)
        for ps in self.waiting_passengers:
            waiting_time = self.sim_time - ps.t0
            floor_waiting_times_dict[ps.xi].append(waiting_time)
        
        # 計算每個樓層的等待時間統計
        for floor, times in floor_waiting_times_dict.items():
            if times:
                floor_waiting_stats[floor] = {
                    'count': len(times),
                    'mean': np.mean(times),
                    'std': np.std(times),
                    'min': np.min(times),
                    'max': np.max(times),
                    'median': np.median(times)
                }
        
        # 打印樓層等待時間變異性分析
        if verbose:
            print(f"\n=== Floor-by-Floor Waiting Time Variability Analysis ===")
            print(f"{'Floor':<6} {'Count':<6} {'Mean':<8} {'Std':<8} {'CV':<10} {'Min':<8} {'Max':<8} {'Median':<8}")
            print("-" * 70)
            
            for floor in sorted(floor_waiting_stats.keys()):
                stats = floor_waiting_stats[floor]
                floor_name = f"{floor}F" if floor > 0 else "B1"
                cv = stats['std'] / stats['mean'] if stats['mean'] > 0 else 0  # Coefficient of Variation
                
                print(f"{floor_name:<6} {stats['count']:<6} {stats['mean']:<8.1f} {stats['std']:<8.1f} "
                      f"{cv:<10.3f} {stats['min']:<8.1f} {stats['max']:<8.1f} {stats['median']:<8.1f}")
            
            # Analyze floors with highest variability
            if floor_waiting_stats:
                max_std_floor = max(floor_waiting_stats.items(), key=lambda x: x[1]['std'])
                min_std_floor = min(floor_waiting_stats.items(), key=lambda x: x[1]['std'])
                
                max_cv_floor = max(floor_waiting_stats.items(), 
                                 key=lambda x: x[1]['std']/x[1]['mean'] if x[1]['mean'] > 0 else 0)
                
                print(f"\nVariability Analysis:")
                print(f"Floor with highest std dev: {max_std_floor[0]}F (std dev: {max_std_floor[1]['std']:.1f}s)")
                print(f"Floor with lowest std dev: {min_std_floor[0]}F (std dev: {min_std_floor[1]['std']:.1f}s)")
                print(f"Floor with highest CV: {max_cv_floor[0]}F (CV: {max_cv_floor[1]['std']/max_cv_floor[1]['mean']:.3f})")

        # summarize
        summary = {
            "general": {
                "time": self.sim_time,
                "runtime": int(time.time()-self.real_time.ti+0.5)
            },
            "goals": {
                "service_time": dist(all_service_times, do_round=0),
                "total_distance": [el.total_distance for el in self.el],
            },
            "passengers": {
                "arrived": n_ps_scenario,
                "served": n_ps_completed,
                "on_board": [n_ps_moving, moving_max_time],
                "waiting": [n_ps_waiting, waiting_max_time]
            },
            "sanity": {
                "unassigned_passengers": sum([ps.assigned_el==-1 for ps in self.waiting_passengers]),
                "unnecessary_opens": self.useless_opens,
                "blocked_entrances": self.blocked_entrances,
                "indirect_motions": [
                    sum([ps.indirect_motion>0 for ps in self.completed_passengers]),
                    sum([ps.indirect_motion   for ps in self.completed_passengers])
                ]
            },
            "info": {
                # 所有乘客等待時間的統計分佈 (包括waiting, moving, completed)
                "waiting_time": dist(all_waiting_times, do_round=0),
                # 所有乘客等待時間的總和
                "total_waiting_time": sum(all_waiting_times),
                # 所有相關乘客在電梯內時間的總和 (moving + completed)
                "total_inside_time": sum(all_inside_times),
                # 所有相關乘客在電梯內時間的統計分佈
                "inside_time": dist(all_inside_times, do_round=0),
                # 電梯開門的次數
                "total_opens": [el.total_opens for el in self.el],
                # 電梯沒有開門的次數
                "moves_without_open": self.moves_without_open,
                # 電梯的剩餘任務數
                "remaining_missions": [len(el.missions) for el in self.el],
                # 按樓層分組的等待時間統計 (新增)
                "floor_waiting_stats": floor_waiting_stats
            }
        }

        if verbose:
            if self.debug:
                pprint.pprint(summary)
            self.plot_results(summary)

        if self.logger: self.logger.close()

        return summary

    def plot_results(self, S):
        f, axs = plt.subplots(2, 2)
        # service time
        ax = axs[0,0]

        # return [len(x), np.mean(x), np.percentile(x, 0~100)]
        quants = tuple(range(0,101))
        # 使用新的全包含計算方式，但需要重新計算以獲得百分位數據
        # 計算所有乘客的等待時間和在電梯內時間
        all_waiting_times = []
        # 1. Completed passengers: t1 - t0
        all_waiting_times.extend([ps.t1 - ps.t0 for ps in self.completed_passengers])
        # 2. Moving passengers: t1 - t0
        for ps_list in self.moving_passengers:
            all_waiting_times.extend([ps.t1 - ps.t0 for ps in ps_list])
        # 3. Waiting passengers: sim_time - t0 (假設現在進入電梯)
        all_waiting_times.extend([self.sim_time - ps.t0 for ps in self.waiting_passengers])
        
        all_inside_times = []
        # 1. Completed passengers: t2 - t1
        all_inside_times.extend([ps.t2 - ps.t1 for ps in self.completed_passengers])
        # 2. Moving passengers: sim_time - t1 (假設現在離開電梯)
        for ps_list in self.moving_passengers:
            all_inside_times.extend([self.sim_time - ps.t1 for ps in ps_list])
            
        all_service_times = []
        # 1. Completed passengers: t2 - t0
        all_service_times.extend([ps.t2 - ps.t0 for ps in self.completed_passengers])
        # 2. Moving passengers: sim_time - t0
        for ps_list in self.moving_passengers:
            all_service_times.extend([self.sim_time - ps.t0 for ps in ps_list])
        # 3. Waiting passengers: sim_time - t0
        all_service_times.extend([self.sim_time - ps.t0 for ps in self.waiting_passengers])
        
        # 計算分布數據用於繪圖
        t_tot = dist(all_service_times, quants) if all_service_times else [0, 0] + [0]*len(quants)
        t_wait = dist(all_waiting_times, quants) if all_waiting_times else [0, 0] + [0]*len(quants)
        t_inside = dist(all_inside_times, quants) if all_inside_times else [0, 0] + [0]*len(quants)

        # 繪製總等待時間、外部等待時間和內部等待時間的分布曲線
        ax.plot(quants, t_tot[2:], 'k-')
        ax.plot(quants, t_wait[2:], 'm-')
        ax.plot(quants, t_inside[2:], 'y-')
        ax.hlines(y=t_tot[1],    xmin=0, xmax=100, linestyles='dashed', color='k')
        ax.hlines(y=t_wait[1],   xmin=0, xmax=100, linestyles='dashed', color='m')
        ax.hlines(y=t_inside[1], xmin=0, xmax=100, linestyles='dashed', color='y')
        ax.set_xlim((0,100))
        ax.set_ylim((0,None))
        ax.legend(("Total", "Outside", "Inside"))
        ax.set_xlabel('Quantile [%]')
        ax.set_ylabel('Time [s]')
        ax.set_title('Service Time Distribution (for ALL passengers)')
        # passengers
        ax = axs[0,1]
        s = S['passengers']
        ax.bar(list(range(3)), [s['waiting'][0],s['on_board'][0],s['served']], color='k')
        ax.set_ylabel('Passengers')
        ax.set_title('Eventual Passengers Status')
        ax.set_xticks(list(range(3)))
        ax.set_xticklabels((f"Waiting\n(longest={s['waiting'][1]:.0f}[s])",
                            f"On-board\n(longest={s['on_board'][1]:.0f}[s])",
                            "Served"))
        # bad incidences
        ax = axs[1,0]
        s = S['sanity']
        ax.bar(list(range(4)),
                    [s['unassigned_passengers'],s['indirect_motions'][0],
                     s['unnecessary_opens'],s['blocked_entrances']],
                    color='r'
                    )
        ax.set_ylabel('Occurences')
        ax.set_title('Bad Behavior')
        ax.set_xticks(list(range(4)))
        ax.set_xticklabels(('Unassigned\npassengers', 'Indirect\ntravels',
                            'Unnecessary\nopens', 'Blocked\nentrances'))
        # text info
        ax = axs[1,1]
        text1 = '\n'.join((
            f"        ADDITIONAL INFO",
            f"Simulation:",
            f"    Time:            {S['general']['time']:.0f}",
            f"    Runtime:         {S['general']['runtime']:.0f}",
            f"Passengers:",
            f"    Total waiting:   {S['info']['total_waiting_time']:.1f}s",
            f"    Total inside:    {S['info']['total_inside_time']:.1f}s",
            f"Elevators:",
            f"    Total distance:  " + ", ".join([f"{x:.0f}" for x in S['goals']['total_distance']]),
            f"    Total opens:     " + ", ".join([f"{x:.0f}" for x in S['info']['total_opens']]),
            f"    Non-open moves:  {S['info']['moves_without_open']:.0f}",
            f"    Remaining tasks: " + ", ".join([f"{x:.0f}" for x in S['info']['remaining_missions']])
        ))
        text2 = '\n'.join((
            f"        CONFIGURATION",
            f"Manager:         {self.manager_info[0]:s}",
            f"World:",
            f"    Time length: {self.sim_len:.0f}",
            f"    Floors:      {self.n_floors:.0f}",
            f"    Elevators:   {self.n_elevators:.0f}",
            f"    Seed:        {self.seed:f}",
            f"Elevators:",
            f"    Capacity:    {self.el[0].capacity:.0f}",
            f"    Speed:       {self.el[0].speed:.0f}",
            f"    Open time:   {self.el[0].open_time:.0f}",
            f"Passengers:",
            f"    Average time between: {self.arrivals_pace:.0f}",
            f"    Average number:       {self.arrival_size:.0f}",
            f"    Typical delay:        {self.delay:.0f}",
            f"    Going up:             {100*self.p_go_up:.0f}%",
            f"    Going down:           {100*self.p_go_down:.0f}%",
            f"    Going between:        {100*self.p_go_between:.0f}%"
        ))
        ax.text(0.05, 0.95, text1, transform=ax.transAxes, family='monospace',
                fontsize=10, verticalalignment='top')
        ax.text(0.55, 0.95, text2, transform=ax.transAxes, family='monospace',
                fontsize=10, verticalalignment='top')
        ax.set_xticks(())
        ax.set_yticks(())
        # draw
        try:
            # Try to maximize window if supported by the backend
            fig_manager = plt.get_current_fig_manager()
            if hasattr(fig_manager, 'window'):
                if hasattr(fig_manager.window, 'showMaximized'):
                    fig_manager.window.showMaximized()
                elif hasattr(fig_manager.window, 'wm_state'):
                    fig_manager.window.wm_state('zoomed')  # Windows
                elif hasattr(fig_manager, 'window') and hasattr(fig_manager.window, 'state'):
                    fig_manager.window.state('zoomed')  # Alternative for some backends
        except Exception:
            # If maximizing fails, just continue without maximizing
            pass
        plt.draw()
        plt.pause(1e-17)

    def log(self, event, data, t=None):
        if t is None: t = self.sim_time
        if self.logger is None:
            print('\t'.join([f"[{t:.1f}]", f"{event:s}:", data]))
        else:
            self.logger.write('\t'.join([f"[{t:.1f}]", f"{event:s}:", data])+'\n')

    def update_plot(self, dt):
        if self.sim_plot is not None:
            self.sim_plot.update_plot(dt, self.el,
                                      [wp.xi for wp in self.waiting_passengers],
                                      [len(mp) for mp in self.moving_passengers])
            