# 引入必要的套件
import sys
import time
import numpy as np

class PrintTime():
    """用於計算和顯示程式執行時間的類別"""
    def __init__(self):
        # 初始化參考時間點
        self.ti = time.time()
        
    def update_reference(self):
        """更新參考時間點"""
        self.ti = time.time()
        
    def print_time(self):
        """計算並印出從參考時間點到現在經過的時間"""
        tf = time.time()
        dt = tf - self.ti
        mins = int(dt//60)  # 計算分鐘數
        secs = int(dt%60)   # 計算秒數
        print('Elapsed runtime:\t{:d}:{:02d} minutes'.format(mins,secs))

# 生成指定長度的None列表
nones = lambda n: [None for _ in range(n)]

def assert_integers(*args):
    """檢查所有輸入參數是否為整數"""
    for x in args:
        assert (x==int(x)), x

def assert_zero(*args, eps=sys.float_info.epsilon):
    """檢查所有輸入參數是否接近於零(誤差範圍內)"""
    for x in args:
        assert (abs(x)<eps), x

def dist(x, quantiles=(0,10,50,90,100), do_round=False):
    """
    計算數據的分布統計
    
    參數:
        x: 輸入數據列表
        quantiles: 要計算的百分位數
        do_round: 是否對結果進行四捨五入
        
    返回:
        [
            總數,               # len(x)
            平均值,             # np.mean(x)
            第 0 百分位數,       # min
            第 10 百分位數,
            第 50 百分位數,     # 中位數
            第 90 百分位數,
            第 100 百分位數     # max
        ]
    """
    if not len(x):
        return nones(2+len(quantiles))
    s = [len(x), np.mean(x)] + list(np.percentile(x,quantiles))
    return [int(z+np.sign(z)*0.5) for z in s] if do_round else s
