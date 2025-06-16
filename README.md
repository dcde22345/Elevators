# Elevators Management: Visual Simulator and Optimization Algorithms

This project implements a visual simulator for an elevator system, including several optimization algorithms and an analysis of their results.

## Project Structure

To make the project easier to maintain and expand, we have organized the directory structure as follows:

```
.
├── Raw_and_Cleaned_Data/  # Raw and cleaned data
├── images/                  # Contains all analysis charts
│   ├── monte_carlo/
│   ├── no_limitations/
│   ├── privileged_faculty/
│   └── with_limitations/
├── results/                 # Contains simulation result data (e.g., .csv)
│   └── quartile_stats/
├── scripts/                 # Scripts for running simulations and tests
│   ├── dev.py               # Main development and simulation script
│   └── test_dev.py          # Script for testing
├── src/                     # All core source code
│   ├── AnalysisPlotter.py
│   ├── DDSAlgorithm.py
│   ├── Elevator.py
│   ├── ElevatorManager.py
│   ├── ElevatorSimulator.py
│   ├── ElevatorTester.py
│   ├── LookAlgorithm.py
│   ├── MyTools.py
│   ├── Passenger.py
│   ├── ReinforcementElevator.py
│   └── SimulationPlotter.py
└── README.md                # Project documentation
```

## Installation and Setup

A Python 3.x environment is recommended. You will need to install some dependencies. It is recommended to create a `requirements.txt` file with the following content:

```
numpy
matplotlib
pandas
prettytable
seaborn
```

Then, install them using the following command:
```bash
pip install -r requirements.txt
```

## How to Run

You can run the simulation by executing the scripts in the `scripts` folder. The main script is `dev.py`.

Run from the project root directory:
```bash
python scripts/dev.py
```
After execution, the latest simulation result charts will be saved in the corresponding subdirectories of `images/`, and the data will be stored in the `results/` folder.

## Implemented Elevator Algorithms

This project implements several elevator dispatch algorithms as a basis for testing and comparison.

*   **NaiveManager**: A simple baseline algorithm that uses the first elevator to handle all passenger requests sequentially.
*   **NaiveRoundRobin**: Assigns elevators in turn to handle passenger requests.
*   **GreedyManager**: Attempts to disperse waiting elevators and assigns elevators to passengers greedily.
*   **Look (DirectManager)**: A variant of the classic elevator algorithm where the elevator continues to move in its current direction as long as there are requests in that direction, before changing direction.
*   **DDSManager**: Destination Dispatch System, where passengers input their destination when calling the elevator, allowing a central system to dispatch cars more efficiently.

## Core Module Descriptions

*   `ElevatorSimulator`: The core of the simulator, responsible for managing the simulation flow, event handling, and state updates.
*   `ElevatorManager`: The base class and implementation for various elevator management algorithms.
*   `Elevator`: Represents a single elevator object, including its state and properties.
*   `Passenger`: Represents a passenger object.
*   `AnalysisPlotter`: Used for analyzing simulation data and generating various charts (e.g., passenger flow heatmaps, waiting time distribution plots).
*   `SimulationPlotter`: Provides a real-time visualization interface using `matplotlib` to dynamically observe elevator operations.

## Future Work and Outlook

*   **Reinforcement Learning Algorithm**: The `ReinforcementElevator.py` module is currently a preliminary concept and has not been implemented. The future direction is to implement a reinforcement learning elevator controller and compare its performance with traditional algorithms. This will require addressing issues such as state space definition, sampling frequency, and model training.
*   **User Interface**: Develop a more user-friendly graphical user interface (GUI) to allow users to more easily adjust parameters, select algorithms, and observe simulation results.
*   **More Algorithms**: Implement more advanced elevator dispatch algorithms, such as OTIS's FPA (Fuzzy-logic-based Peak-shaving Algorithm).
*   **Parameter Optimization**: Optimize the parameters of the algorithms for different traffic patterns (e.g., morning peak, evening peak, off-peak).
*   **Detailed Documentation**: Write more complete documentation (Docstrings) for each module and function.
