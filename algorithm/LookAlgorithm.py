
from MyTools import *
from ElevatorManager import NaiveManager
import numpy as np
from warnings import warn

class DirectManager(NaiveManager):

    # Possible elevator states
    UP = 1 # in the middle of a journey upward
    DOWN = -1 # in the middle of a journey downward
    UP_INIT = 2 # moving toward initial floor for upward journey
    DOWN_INIT = -2 # moving toward initial floor for downward journey
    INIT = 3 # move toward resting point
    REST = 0 # not moving

    def __init__(self, n_floors, n_elevators,
                 capacity, speed, open_time,
                 arrivals_pace=None, p_up=0.5, p_down=0.5, p_between=0., size=1., delay=3.):
        NaiveManager.__init__(self, n_floors, n_elevators, capacity, speed, open_time,
                     arrivals_pace, p_up, p_down, p_between, size, delay)
        # waiting dispersion state
        self.n0 = int(np.ceil(self.N/3))
        self.nn = self.N - self.n0
        delta = self.H / (2 * self.nn + 1)
        self.bases = [int(b) for b in np.round(2 * delta * (1 + np.arange(self.nn)))]
        # elevators status
        self.state = [DirectManager.REST for _ in range(self.N)]
        self.onway = [[] for _ in range(self.N)]
        self.onway_load = [[] for _ in range(self.N)]
        self.ascending = [[] for _ in range(self.N)]
        self.asc_load = [[] for _ in range(self.N)]
        self.descending = [[] for _ in range(self.N)]
        self.dec_load = [[] for _ in range(self.N)]
        self.force_open = [[False for _ in range(self.N)] for _ in range(2)]

    @staticmethod
    def version_info():
        return ("DirectManager",
                '''Serve passengers under the constraint of direct journey only.''')

    def handle_no_missions(self, t, idx):
        self.onway[idx] = []
        self.onway_load[idx] = []
        opened = False

        if self.state[idx] == DirectManager.UP_INIT:
            self.begin_ascend(idx, self.el[idx].x)

        elif self.state[idx] == DirectManager.DOWN_INIT:
            self.begin_descend(idx, self.el[idx].x)

        elif self.state[idx] == DirectManager.INIT:
            return self.goto_rest(idx)

        elif self.state[idx] in (DirectManager.UP, DirectManager.DOWN):
            opened = True
            d = self.get_next_direction(idx)
            if self.force_open[d>0][idx]: opened = False
            self.force_open[0][idx] = False
            self.force_open[1][idx] = False

            if   d > 0:
                self.begin_ascend(idx, self.el[idx].x)
            elif d < 0:
                self.begin_descend(idx, self.el[idx].x)
            else:
                return self.goto_rest(idx)

        else:
            raise ValueError("Tasks are not supposed to be finished while resting.",
                             self.state[idx])

        return {idx: self.cancel_tasks(self.el[idx]) + self.generate_tasks(idx,opened)}

    def handle_arrival(self, t, xi, xf):
        self.update_onways()
        l = self.choose_elevator(t, xi, xf)
        return self.get_tasks(l, xi, xf)

    def get_tasks(self, l, xi, xf):
        if self.state[l] in (DirectManager.REST,DirectManager.INIT):
            self.push_task(l, xi, xf)
            if xi < xf:
                self.begin_ascend(l, self.el[l].x)
            else:
                self.begin_descend(l, self.el[l].x)
            return {-1:l, l: self.cancel_tasks(self.el[l])+self.generate_tasks(l)}
        elif self.is_onway(l, self.el[l].x, xi, xf):
            self.add_onway(l,xi,xf)
            return {-1:l, l: self.cancel_tasks(self.el[l])+self.generate_tasks(l)}
        else:
            if len(self.onway[l])==1 and self.onway[l][0]==self.el[l].x and \
               self.onway[l][0]==xi and self.el[l].is_open:
                self.force_open[xf-xi>0][l] = True
            self.push_task(l, xi, xf)
            return {-1:l}

    def push_task(self, l, xi, xf):
        if   xi < xf:
            self.add_ascend(l,xi,xf)
        elif xi > xf:
            self.add_descend(l,xi,xf)
        else:
            raise ValueError("Source & destination floors must be different.", xi, xf)

    def is_onway(self, l, x, xi, xf):
        return self.state[l]==(DirectManager.UP if xi<xf else DirectManager.DOWN) \
               and (xf-xi)*(xi-x)>=0 # not sure if > or >=

    def add_onway(self, l, xi, xf):
        s = self.arrivals_info['size']
        self.add_stop(self.onway[l], self.onway_load[l], xi,  s, self.state[l]<0)
        self.add_stop(self.onway[l], self.onway_load[l], xf, -s, self.state[l]<0)

    def add_ascend(self,l,xi,xf):
        self.add_stop(self.ascending[l], self.asc_load[l], xi,  1, False)
        self.add_stop(self.ascending[l], self.asc_load[l], xf, -1, False)

    def add_descend(self,l,xi,xf):
        self.add_stop(self.descending[l], self.dec_load[l], xi,  1, True)
        self.add_stop(self.descending[l], self.dec_load[l], xf, -1, True)

    def add_stop(self, A, loads, x, count, dec):
        i = DirectManager.insort(A, x, dec)
        if i < 0:
            loads[-i-1] += count
        else:
            loads.insert(i,count)

    def cancel_tasks(self, el):
        return [(None,False,0) for _ in range(len(el.missions))]

    def generate_tasks(self, l, opened=False):
        if self.state[l] == DirectManager.INIT:
            warn("generate_tasks() was called at state INIT rather than goto_rest().")
            return []
        if self.state[l] == DirectManager.UP_INIT:
            return [(self.ascending[l][0],False,-1)]
        if self.state[l] == DirectManager.DOWN_INIT:
            return [(self.descending[l][0],False,-1)]
        return [((x,False,-1) if (i==0 and opened and self.el[l].x==x) else (x,True,-1))
                for i,x in enumerate(self.onway[l])]

    @staticmethod
    def insort(A, x, dec=False):
        if not A:
            A.append(x)
            return 0
        l = 0
        r = len(A)
        while l<r:
            m = (l+r)//2
            if (A[m]-x)*(-1)**dec>=0: r=m
            else: l=m+1
        if l != len(A) and x == A[l]:
            return -1-l
        else:
            A.insert(l,x)
            return l

    def update_onways(self):
        for l in range(self.N):
            self.update_onway(l, self.el[l].x)

    def update_onway(self, l, x):
        n = 0
        if self.state[l] == DirectManager.UP:
            n = sum(stop<x for stop in self.onway[l]) # inefficient - doesn't exploit the sorted list
        elif self.state[l] == DirectManager.DOWN:
            n = sum(stop>x for stop in self.onway[l]) # inefficient - doesn't exploit the sorted list
        del(self.onway[l][:n])
        del(self.onway_load[l][:n])

    def begin_ascend(self, l, x):
        xi = self.ascending[l][0]
        if xi < x:
            self.state[l] = DirectManager.UP_INIT
            self.onway[l] = []
            self.onway_load[l] = []
        else:
            self.state[l] = DirectManager.UP
            self.onway[l] = self.ascending[l]
            self.onway_load[l] = self.asc_load[l]
            self.ascending[l] = []
            self.asc_load[l] = []

    def begin_descend(self, l, x):
        xi = self.descending[l][0]
        if xi > x:
            self.state[l] = DirectManager.DOWN_INIT
            self.onway[l] = []
            self.onway_load[l] = []
        else:
            self.state[l] = DirectManager.DOWN
            self.onway[l] = self.descending[l]
            self.onway_load[l] = self.dec_load[l]
            self.descending[l] = []
            self.dec_load[l] = []

    def goto_rest(self, l):
        x = self.choose_rest_floor(l)
        if x == self.el[l].x:
            self.state[l] = DirectManager.REST
            return {l: self.cancel_tasks(self.el[l])}
        else:
            self.state[l] = DirectManager.INIT
            return {l: self.cancel_tasks(self.el[l])+[(x,False,-1)]}

    def get_next_direction(self, l):
        if self.ascending[l] and self.descending[l]:
            return self.choose_next_direction(l)
        elif self.ascending[l]:
            return 1
        elif self.descending[l]:
            return -1
        else:
            return 0

    def choose_rest_floor(self, l):
        """
        選擇電梯休息樓層
        讓所有電梯在沒有任務時回到一樓（樓層0）等待
        """
        return 0  # 回到一樓

    def choose_elevator(self, t, xi, xf):
        """
        選擇電梯的策略：
        1. 計算乘客想要的方向
        2. 優先選擇方向匹配的電梯（包括靜止的電梯）
        3. 在匹配的電梯中選擇負載最小的
        4. 如果沒有匹配的電梯，選擇總負載最小的
        """
        # 計算乘客想要的方向
        passenger_direction = 1 if xf > xi else -1  # 1 for up, -1 for down
        
        # 找出方向匹配的電梯
        compatible_elevators = []
        for l in range(self.N):
            elevator_motion = self.el[l].motion
            
            # 電梯方向匹配條件：
            # 1. 電梯靜止 (motion == 0) - 可以接任何方向
            # 2. 電梯方向與乘客方向相同
            if elevator_motion == 0 or elevator_motion == passenger_direction:
                compatible_elevators.append(l)
        
        if compatible_elevators:
            # 在方向匹配的電梯中選擇負載最小的
            def get_total_load(l):
                return len(self.onway[l]) + len(self.ascending[l]) + len(self.descending[l])
            
            best_elevator = min(compatible_elevators, key=get_total_load)
            
            # Debug 信息（可選）
            if hasattr(self, 'debug') and self.debug:
                direction_str = "up" if passenger_direction == 1 else "down"
                print(f"乘客 {xi}->{xf} (方向:{direction_str}) 分配給方向匹配的電梯 #{best_elevator}")
            
            return best_elevator
        else:
            # 如果沒有方向匹配的電梯，回到原來的邏輯（選擇負載最小的）
            best_elevator = np.argmin([len(self.onway[l])+len(self.ascending[l])+len(self.descending[l])
                                      for l in range(self.N)])
            
            if hasattr(self, 'debug') and self.debug:
                direction_str = "up" if passenger_direction == 1 else "down"
                print(f"乘客 {xi}->{xf} (方向:{direction_str}) 無匹配方向電梯，分配給負載最小的電梯 #{best_elevator}")
            
            return best_elevator

    def choose_next_direction(self, l):
        # TODO
        return -1 if self.state[l]==DirectManager.UP else 1


if __name__ == "__main__":
    import ElevatorTester, matplotlib.pyplot as plt
    c = ElevatorTester.ELEVATOR_TESTS_CONFS[0]
    c['sim_len'] = 120
    x = ElevatorTester.ManagerTester(DirectManager, c, -1)
    x.single_test(c)
    plt.show()
