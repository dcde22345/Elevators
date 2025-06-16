from .MyTools import *
from .ElevatorManager import NaiveManager
import numpy as np
from warnings import warn

class DDSManager(NaiveManager):
    """
    Destination Dispatch System (DDS) 電梯調度演算法
    - 乘客在大廳輸入目的地，系統集中分派
    - 同目的地或相近樓層的乘客盡量分配到同一台電梯
    - 分派後，該電梯的行程不再臨時加人，直到完成再重新分派
    """

    # 電梯狀態
    UP = 1
    DOWN = -1
    INIT = 3
    REST = 0

    def __init__(self, n_floors, n_elevators,
                 capacity, speed, open_time,
                 arrivals_pace=None, p_up=0.5, p_down=0.5, p_between=0., size=1., delay=3.):
        super().__init__(n_floors, n_elevators, capacity, speed, open_time,
                         arrivals_pace, p_up, p_down, p_between, size, delay)
        # 等待池：[(t, xi, xf)]
        self.waiting_pool = []
        # 每台電梯的目的地清單
        self.destinations = [[] for _ in range(self.N)]
        self.state = [DDSManager.REST for _ in range(self.N)]

    @staticmethod
    def version_info():
        return ("DDSManager", "Destination Dispatch System (DDS) elevator scheduling algorithm.")

    def handle_arrival(self, t, xi, xf):
        """
        新乘客到達時，加入等待池並觸發分派

        parameters:
            t: 時間
            xi: 起始樓層
            xf: 目的地樓層

        return:
            tasks: 所有電梯的任務
        """
        self.waiting_pool.append((t, xi, xf))
        self.dispatch_requests()

        # 回傳所有電梯的任務
        tasks = {}
        for l in range(self.N):
            tasks[l] = self.generate_tasks(l)
        return tasks

    def dispatch_requests(self):
        """
        根據等待池中的乘客請求，將相同目的地或相近樓層的請求
        分派給目前處於 REST 狀態的電梯。封閉式策略，不會動到執行中電梯。
        """
        # 僅允許空閒電梯參與分派
        available_elevators = [l for l in range(self.N) if self.state[l] == DDSManager.REST]
        if not available_elevators or not self.waiting_pool:
            return  # 沒有空閒電梯或沒有乘客等待

        # 分群：依目的地樓層為 key（你也可以改成 cluster，例如 12–15 為一群）
        groups = {}
        for req in self.waiting_pool:
            _, xi, xf = req
            if xf not in groups:
                groups[xf] = []
            groups[xf].append(req)

        used = set()

        # 分派給最佳空閒電梯
        for xf, reqs in groups.items():
            if not available_elevators:
                break  # 空電梯都被分配完了

            # 找出負載最小且距離最近的電梯（在 available_elevators 範圍內）
            best_l = min(available_elevators, key=lambda l: (
                len(self.destinations[l]),  # 優先分給目前任務少的
                abs(self.el[l].x - reqs[0][1])  # 距離目前樓層最近的
            ))

            for req in reqs:
                used.add(req)
                self.destinations[best_l].append(req)

            # 一旦這台電梯被派任務，就設定為執行中
            self.state[best_l] = DDSManager.UP if xf > self.el[best_l].x else DDSManager.DOWN
            available_elevators.remove(best_l)

        # 移除已經分派的請求
        self.waiting_pool = [req for req in self.waiting_pool if req not in used]

    def generate_tasks(self, l):
        """
        根據分派結果產生該電梯的行程
        """
        if not self.destinations[l]:
            # 沒有分派的目的地時，不回傳任何任務
            return []
        # 依目的地排序（可優化為最短路徑）
        stops = sorted(set([xf for (_, _, xf) in self.destinations[l]]))
        # 產生任務格式 (floor, open, -1)
        return [(x, True, -1) for x in stops]

    def handle_no_missions(self, t, idx):
        """
        當電梯無任務時，觸發重新分派或讓電梯休息
        """
        self.state[idx] = DDSManager.REST
        # 重新檢查是否有未分派的請求
        if self.waiting_pool:
            self.dispatch_requests()
            # 如果有新分派的任務，回傳該電梯的任務
            tasks = self.generate_tasks(idx)
            if tasks:
                return {idx: tasks}
        # 沒有任務時，回傳空字典，讓電梯進入休息狀態
        return {}
