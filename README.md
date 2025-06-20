# Elevators Management: Visual Simulator and Optimization Algorithms

This repo implements a visual simulator of elevators system, along with several simple optimization algorithms and analysis of their results.

Some of the results are demonstrated [here](https://github.com/ido90/Elevators/tree/master/Demonstrations), and some of them can be reproduced using [these](https://github.com/ido90/Elevators/tree/master/Main) scripts.

A [Reinforcement-Learning manager](#module-reinforcementelevator) is intended to be implemented and tested vs. the classic [DirectManager](#implemented-managers).

## Implemented managers

Note: all the currently-implemented managers are either naive or incomplete, and were mainly used for testing and demonstration of the simulative infrastructure, as well as future reference for more advanced algorithms.

- **NaiveManager**: Use the first elevator to handle passengers arrivals sequentially.
- **NaiveRoundRobin**: Use the elevators in turns to handle passengers arrivals.
- **GreedyManager**: Try to disperse waiting elevators, and assign elevators to passengers greedily.
- **DirectManager**: Go on while there're more passengers in the current motion direction, then turn around (variant of the classic elevator algorithm).

| ![](https://github.com/ido90/Elevators/blob/master/Demonstrations/tests%20summary.png) |
| :--: |
| Summary of the results of the various managers |

## Class: ElevatorSimulator.Simulator

This class implements a simulation of elevators which serve arriving passengers.

Involved classes:
- Simulator       = simulation manager
- ElevatorManager = decision makers
- Elevator        = represent the elevators in the simulation
- Arrival         = encode a single event of arrival of passengers
- Passenger       = encode and track a single passenger
- SimPlot         = visualize the simulation dynamically via matplotlib

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

| ![](https://idogreenberg.neocities.org/linked_images/elevators.JPG) |
| :--: |
| A screenshot from the visual simulation |

## Module: ElevatorTester

This module defines various scenarios, tests the managers of ElevatorManager using ElevatorSimulator, and summarizes the results.

## Module: ElevatorManager

This module contains the elevator-managers (one class per manager).

A manager handles 3 kinds of events:
1. Initialization of elevators locations.
2. Arrival of passengers.
3. End of tasks of a certain elevator.

A manager can assign tasks in the format {elevator_index : list_of_missions}
where a single task is encoded as a 3D-tuple:
- (n,True,-1)      = go to floor n and open.
- (n,False,-1)     = go to floor n (without opening).
- (n,True/False,k) = get in the middle of another mission - go to n and push it as the k'th task of the elevator.
- (None,False,k)   = remove the current k'th mission of the elevator.

In cases of new arrival, the output dict must also include: {-1 : elevator_assigned_to_arrival}.

## Module: ReinforcementElevator

This module is **NOT IMPLEMENTED**, up to definition of states and a simple count of them (or at least a lower-bound of the count), demonstrating that direct search in state-space (e.g. Value Iteration) is impractical for any interesting configuration.
Instead, some encoding of the states should be used (e.g. like [here](https://papers.nips.cc/paper/1073-improving-elevator-performance-using-reinforcement-learning.pdf)).

Implementation of this module should take care of the following issues:
1. **Sampling resolution**: high-resolution (e.g. sample every time the elevators move one floor) permits simple state-space, but low-resolution (e.g. sample when an elevator reaches its destination, etc.) is better synced with the simulator interface.
2. **State encoding**: compact encoding of the states so that a learner can use an encoded state to make a decision.
3. **Train & test process**: train the decision-maker in various scenarios and test it.


## To be continue
1. **Elevator Chooser**: Need to fix chooser's dispatcher method
2. **Presentation**:
    - poisson distribution generation with timestamp
    - model with only one elevator
        - demo
    - model with 4 elevator -> Look 演算法怪怪的，他不會正確pick up
        - waiting time boxing plot
        - waiting time accross time
        - service time box plot
        - service time accress time
    - model with 4 elevator with limitation vs without limitation
        - waiting time boxing plot
        - waiting time accross time
        - service time box plot
        - service time accress time

    - model with DDS algorithm
        - waiting time boxing plot
        - waiting time accross time
        - service time box plot
        - service time accress time
        - 各樓層waiting time

    - service level
        - 體驗

    - 動機
    - 資料分析 -> 上課人數
    - 資料搜集 -> 人流分析
    - 假設
    - Simulation
    - 建議
    - Exploration