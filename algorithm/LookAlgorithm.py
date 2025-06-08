from MyTools import *
from ElevatorManager import NaiveManager
import numpy as np
from warnings import warn

class Look(NaiveManager):

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
        self.state = [Look.REST for _ in range(self.N)]
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

        if self.state[idx] == Look.UP_INIT:
            self.begin_ascend(idx, self.el[idx].x)

        elif self.state[idx] == Look.DOWN_INIT:
            self.begin_descend(idx, self.el[idx].x)

        elif self.state[idx] == Look.INIT:
            return self.goto_rest(idx)

        elif self.state[idx] in (Look.UP, Look.DOWN):
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
        if self.state[l] in (Look.REST,Look.INIT):
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
        # 驗證電梯能夠到達起始樓層和目的樓層
        elevator = self.el[l]
        if not elevator.can_go_to(xi):
            raise ValueError(f"Elevator {l} cannot reach source floor {xi}")
        if not elevator.can_go_to(xf):
            raise ValueError(f"Elevator {l} cannot reach destination floor {xf}")
            
        if   xi < xf:
            self.add_ascend(l,xi,xf)
        elif xi > xf:
            self.add_descend(l,xi,xf)
        else:
            raise ValueError("Source & destination floors must be different.", xi, xf)

    def is_onway(self, l, x, xi, xf):
        return self.state[l]==(Look.UP if xi<xf else Look.DOWN) \
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
        i = Look.insort(A, x, dec)
        if i < 0:
            loads[-i-1] += count
        else:
            loads.insert(i,count)

    def cancel_tasks(self, el):
        return [(None,False,0) for _ in range(len(el.missions))]

    def generate_tasks(self, l, opened=False):
        if self.state[l] == Look.INIT:
            warn("generate_tasks() was called at state INIT rather than goto_rest().")
            return []
        if self.state[l] == Look.UP_INIT:
            return [(self.ascending[l][0],False,-1)]
        if self.state[l] == Look.DOWN_INIT:
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
        if self.state[l] == Look.UP:
            n = sum(stop<x for stop in self.onway[l]) # inefficient - doesn't exploit the sorted list
        elif self.state[l] == Look.DOWN:
            n = sum(stop>x for stop in self.onway[l]) # inefficient - doesn't exploit the sorted list
        del(self.onway[l][:n])
        del(self.onway_load[l][:n])

    def begin_ascend(self, l, x):
        xi = self.ascending[l][0]
        if xi < x:
            self.state[l] = Look.UP_INIT
            self.onway[l] = []
            self.onway_load[l] = []
        else:
            self.state[l] = Look.UP
            self.onway[l] = self.ascending[l]
            self.onway_load[l] = self.asc_load[l]
            self.ascending[l] = []
            self.asc_load[l] = []

    def begin_descend(self, l, x):
        xi = self.descending[l][0]
        if xi > x:
            self.state[l] = Look.DOWN_INIT
            self.onway[l] = []
            self.onway_load[l] = []
        else:
            self.state[l] = Look.DOWN
            self.onway[l] = self.descending[l]
            self.onway_load[l] = self.dec_load[l]
            self.descending[l] = []
            self.dec_load[l] = []

    def goto_rest(self, l):
        x = self.choose_rest_floor(l)
        if x == self.el[l].x:
            self.state[l] = Look.REST
            return {l: self.cancel_tasks(self.el[l])}
        else:
            self.state[l] = Look.INIT
            return {l: self.cancel_tasks(self.el[l])+[(x,False,-1)]}

    def get_next_direction(self, l):
        """
        LOOK 演算法方向決策：
        - 若當前方向還有請求，繼續原方向。
        - 若無，反向（若有反方向請求）。
        - 若都無，則停止。
        """
        if self.state[l] == Look.UP and (self.ascending[l] or self.onway[l]):
            return 1
        if self.state[l] == Look.DOWN and (self.descending[l] or self.onway[l]):
            return -1
        if self.ascending[l]:
            return 1
        if self.descending[l]:
            return -1
        return 0

    def choose_rest_floor(self, l):
        """
        選擇電梯休息樓層
        優先選擇一樓（樓層1），如果不可達則選擇電梯能到達的最低樓層
        """
        elevator = self.el[l]
        
        # 優先選擇一樓
        if elevator.can_go_to(1):
            return 1
        
        # 如果一樓不可達，選擇能到達的最低樓層
        for floor in range(self.H + 1):
            if elevator.can_go_to(floor):
                return floor
        
        # 如果沒有任何樓層可達（不應該發生），返回當前樓層
        return int(elevator.x)

    def choose_elevator(self, t, xi, xf):
        """
        LOOK 演算法分配邏輯：
        1. 優先分配給正在朝請求方向運行且會經過該樓層的電梯（順路同方向）。
        2. 若無順路電梯，分配給最近的靜止電梯。
        3. 若都沒有，分配給最近的電梯。
        
        新增：考慮電梯limitations，只分配給能到達起始樓層和目的樓層的電梯
        """
        passenger_direction = 1 if xf > xi else -1

        # 過濾出能夠服務這個請求的電梯（能到達起始樓層和目的樓層）
        capable_elevators = []
        for l in range(self.N):
            if self.el[l].can_go_to(xi) and self.el[l].can_go_to(xf):
                capable_elevators.append(l)
        
        # 如果沒有電梯能服務這個請求，拋出異常
        if not capable_elevators:
            raise ValueError(f"No elevator can serve passenger from floor {xi} to floor {xf}")

        # 1. 順路同方向的電梯（從capable_elevators中選擇）
        candidates = []
        for l in capable_elevators:
            elevator = self.el[l]
            if elevator.motion == passenger_direction:
                if (passenger_direction == 1 and elevator.x <= xi) or \
                   (passenger_direction == -1 and elevator.x >= xi):
                    candidates.append(l)
        if candidates:
            return min(candidates, key=lambda l: abs(self.el[l].x - xi))

        # 2. 最近的靜止電梯（從capable_elevators中選擇）
        idle = [l for l in capable_elevators if self.el[l].motion == 0]
        if idle:
            return min(idle, key=lambda l: abs(self.el[l].x - xi))

        # 3. 都沒有，分配給最近的可用電梯
        return min(capable_elevators, key=lambda l: abs(self.el[l].x - xi))


if __name__ == "__main__":
    import ElevatorTester, matplotlib.pyplot as plt
    c = ElevatorTester.ELEVATOR_TESTS_CONFS[0]
    c['sim_len'] = 120
    x = ElevatorTester.ManagerTester(Look, c, -1)
    x.single_test(c)
    plt.show()
