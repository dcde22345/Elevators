import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import matplotlib.font_manager as fm
from Passenger import Arrival
import matplotlib
from contextlib import contextmanager

@contextmanager
def temp_matplotlib_backend(backend='Agg'):
    """
    Context manager for temporarily setting matplotlib backend
    """
    original_backend = matplotlib.get_backend()
    try:
        if original_backend != backend:
            matplotlib.use(backend)
        yield
    finally:
        if original_backend != backend:
            matplotlib.use(original_backend)

class AnalysisPlotter:
    """
    Analysis plotting class for visualizing passenger flow data in elevator systems
    """
    
    def __init__(self, n_floors=13, use_gui=False):
        """
        Initialize analysis plotter
        
        Args:
            n_floors (int): Number of floors, default 13 (B1-12F)
            use_gui (bool): Whether to use GUI mode, default False (non-interactive)
        """
        self.n_floors = n_floors
        self.use_gui = use_gui
        self.floor_names = self._generate_floor_names()
        
        # Set up font support
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    
    def _generate_floor_names(self):
        """Generate floor name labels"""
        floor_names = []
        for i in range(self.n_floors):
            if i == 0:
                floor_names.append('B1')
            else:
                floor_names.append(f'{i}F')
        return floor_names
    
    def analyze_passenger_flow(self, scenario):
        """
        Analyze passenger flow data
        
        Args:
            scenario: Sequence containing Arrival objects
            
        Returns:
            tuple: (flow_matrix, arrivals_per_minute)
        """
        # Initialize 13x13 flow matrix
        flow_matrix = np.zeros((self.n_floors, self.n_floors))
        
        # Track arrivals per minute
        arrivals_per_minute = defaultdict(int)
        
        for arrival in scenario:
            # Count passengers going from floor xi to xf
            flow_matrix[arrival.xi][arrival.xf] += arrival.n
            
            # Count arrivals per minute
            minute = int(arrival.t // 60)  # Convert seconds to minutes
            arrivals_per_minute[minute] += arrival.n
        
        return flow_matrix, arrivals_per_minute
    
    def plot_passenger_flow_heatmap(self, flow_matrix, save_path=None, show_plot=None):
        """
        Plot passenger flow heatmap
        
        Args:
            flow_matrix (np.array): 13x13 flow matrix
            save_path (str, optional): Path to save the plot
            show_plot (bool, optional): Whether to display the plot, defaults to use_gui value
        """
        if show_plot is None:
            show_plot = self.use_gui
        
        # Use non-interactive backend if not showing plot
        backend = 'Agg' if not show_plot else matplotlib.get_backend()
        
        with temp_matplotlib_backend(backend):
            plt.figure(figsize=(12, 10))
            
            # Create heatmap
            sns.heatmap(
                flow_matrix,
                annot=True,
                fmt='.0f',
                cmap='YlOrRd',
                xticklabels=self.floor_names,
                yticklabels=self.floor_names,
                cbar_kws={'label': 'Number of Passengers'},
                square=True
            )
            
            plt.title('Passenger Flow Heatmap (Start Floor → Destination Floor)', fontsize=16, fontweight='bold')
            plt.xlabel('Destination Floor', fontsize=14)
            plt.ylabel('Start Floor', fontsize=14)
            
            # Rotate labels for better readability
            plt.xticks(rotation=45)
            plt.yticks(rotation=0)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Heatmap saved to: {save_path}")
            
            if show_plot:
                plt.show()
            else:
                plt.close()  # Close plot to save memory
                
    def plot_arrivals_per_minute_histogram(self, arrivals_per_minute, save_path=None, show_plot=None):
        """
        Plot histogram of passenger arrivals per minute
        
        Args:
            arrivals_per_minute (dict): Dictionary of passenger arrivals per minute
            save_path (str, optional): Path to save the plot
            show_plot (bool, optional): Whether to display the plot, defaults to use_gui value
        """
        if show_plot is None:
            show_plot = self.use_gui
        
        # Use non-interactive backend if not showing plot
        backend = 'Agg' if not show_plot else matplotlib.get_backend()
        
        with temp_matplotlib_backend(backend):
            # Prepare data
            minutes = list(range(max(arrivals_per_minute.keys()) + 1))
            passenger_counts = [arrivals_per_minute.get(minute, 0) for minute in minutes]
            
            plt.figure(figsize=(14, 8))
            
            # Create histogram
            bars = plt.bar(minutes, passenger_counts, color='skyblue', edgecolor='navy', alpha=0.7)
            
            # Display values on each bar
            for bar, count in zip(bars, passenger_counts):
                if count > 0:
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                            f'{count}', ha='center', va='bottom', fontsize=10)
            
            plt.title('Distribution of Passenger Arrivals per Minute', fontsize=16, fontweight='bold')
            plt.xlabel('Time (minutes)', fontsize=14)
            plt.ylabel('Number of Arriving Passengers', fontsize=14)
            
            # Set X-axis ticks
            plt.xticks(range(0, len(minutes), max(1, len(minutes)//20)))
            
            # Add grid lines
            plt.grid(True, alpha=0.3)
            
            # Add statistics
            total_passengers = sum(passenger_counts)
            avg_passengers_per_minute = total_passengers / len(minutes) if minutes else 0
            max_passengers = max(passenger_counts) if passenger_counts else 0
            
            stats_text = f'Total Passengers: {total_passengers}\nAverage per Minute: {avg_passengers_per_minute:.1f}\nPeak Value: {max_passengers}'
            plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Histogram saved to: {save_path}")
            
            if show_plot:
                plt.show()
            else:
                plt.close()  # Close plot to save memory
    
    def plot_waiting_inside_time_boxplot(self, completed_passengers, save_path=None, show_plot=None):
        """
        Plot boxplots of waiting time and elevator inside time
        
        Args:
            completed_passengers: List of passengers who completed service
            save_path (str, optional): Path to save the plot
            show_plot (bool, optional): Whether to display the plot, defaults to use_gui value
        """
        if show_plot is None:
            show_plot = self.use_gui
        
        # Use non-interactive backend if not showing plot
        backend = 'Agg' if not show_plot else matplotlib.get_backend()
        
        with temp_matplotlib_backend(backend):
            # Prepare data
            waiting_times = [ps.t1 - ps.t0 for ps in completed_passengers]
            inside_times = [ps.t2 - ps.t1 for ps in completed_passengers]
            service_times = [ps.t2 - ps.t0 for ps in completed_passengers]
            
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # Waiting time boxplot
            box1 = axes[0].boxplot(waiting_times, patch_artist=True)
            box1['boxes'][0].set_facecolor('lightblue')
            axes[0].set_title('Waiting Time Distribution', fontsize=14, fontweight='bold')
            axes[0].set_ylabel('Time (seconds)', fontsize=12)
            axes[0].grid(True, alpha=0.3)
            
            # Add statistics
            wait_stats = f'Mean: {np.mean(waiting_times):.1f}s\nMedian: {np.median(waiting_times):.1f}s\nStd Dev: {np.std(waiting_times):.1f}s'
            axes[0].text(0.02, 0.98, wait_stats, transform=axes[0].transAxes, 
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Inside time boxplot
            box2 = axes[1].boxplot(inside_times, patch_artist=True)
            box2['boxes'][0].set_facecolor('lightgreen')
            axes[1].set_title('Inside Time Distribution', fontsize=14, fontweight='bold')
            axes[1].set_ylabel('Time (seconds)', fontsize=12)
            axes[1].grid(True, alpha=0.3)
            
            # Add statistics
            inside_stats = f'Mean: {np.mean(inside_times):.1f}s\nMedian: {np.median(inside_times):.1f}s\nStd Dev: {np.std(inside_times):.1f}s'
            axes[1].text(0.02, 0.98, inside_stats, transform=axes[1].transAxes,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Total service time boxplot
            box3 = axes[2].boxplot(service_times, patch_artist=True)
            box3['boxes'][0].set_facecolor('lightyellow')
            axes[2].set_title('Total Service Time Distribution', fontsize=14, fontweight='bold')
            axes[2].set_ylabel('Time (seconds)', fontsize=12)
            axes[2].grid(True, alpha=0.3)
            
            # Add statistics
            service_stats = f'Mean: {np.mean(service_times):.1f}s\nMedian: {np.median(service_times):.1f}s\nStd Dev: {np.std(service_times):.1f}s'
            axes[2].text(0.02, 0.98, service_stats, transform=axes[2].transAxes,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.suptitle('Passenger Service Time Analysis', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Boxplot saved to: {save_path}")
            
            if show_plot:
                plt.show()
            else:
                plt.close()
    
    def plot_floor_waiting_time_comparison(self, completed_passengers, save_path=None, show_plot=None):
        """
        Plot waiting time comparison across floors
        
        Args:
            completed_passengers: List of passengers who completed service
            save_path (str, optional): Path to save the plot
            show_plot (bool, optional): Whether to display plot, defaults to use_gui value
        """
        if show_plot is None:
            show_plot = self.use_gui
        
        # Use non-interactive backend if not showing plot
        backend = 'Agg' if not show_plot else matplotlib.get_backend()
        
        with temp_matplotlib_backend(backend):
            # Group waiting time data by floor
            floor_waiting_times = defaultdict(list)
            for ps in completed_passengers:
                waiting_time = ps.t1 - ps.t0
                floor_waiting_times[ps.xi].append(waiting_time)
            
            # Prepare data for boxplot
            floors = sorted(floor_waiting_times.keys())
            waiting_data = [floor_waiting_times[floor] for floor in floors]
            floor_labels = [self.floor_names[floor] for floor in floors]
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
            
            # Boxplot
            if waiting_data:
                boxes = ax1.boxplot(waiting_data, labels=floor_labels, patch_artist=True)
                
                # Set different colors for each box
                colors = plt.cm.Set3(np.linspace(0, 1, len(boxes['boxes'])))
                for box, color in zip(boxes['boxes'], colors):
                    box.set_facecolor(color)
                
                ax1.set_title('Waiting Time Distribution by Floor', fontsize=14, fontweight='bold')
                ax1.set_xlabel('Floor', fontsize=12)
                ax1.set_ylabel('Waiting Time (seconds)', fontsize=12)
                ax1.grid(True, alpha=0.3)
                ax1.tick_params(axis='x', rotation=45)
            
            # Average waiting time bar chart
            avg_waiting_times = []
            passenger_counts = []
            
            for floor in floors:
                times = floor_waiting_times[floor]
                if times:
                    avg_waiting_times.append(np.mean(times))
                    passenger_counts.append(len(times))
                else:
                    avg_waiting_times.append(0)
                    passenger_counts.append(0)
            
            bars = ax2.bar(floor_labels, avg_waiting_times, alpha=0.7, color='skyblue', edgecolor='navy')
            
            # Display passenger count above each bar
            for bar, count in zip(bars, passenger_counts):
                if count > 0:
                    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                            f'{count} pax', ha='center', va='bottom', fontsize=10)
            
            ax2.set_title('Average Waiting Time by Floor', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Floor', fontsize=12)
            ax2.set_ylabel('Average Waiting Time (seconds)', fontsize=12)
            ax2.grid(True, alpha=0.3)
            ax2.tick_params(axis='x', rotation=45)
            
            # Add statistics table
            stats_text = "Floor Statistics:\n"
            for floor, label in zip(floors, floor_labels):
                times = floor_waiting_times[floor]
                if times:
                    avg_time = np.mean(times)
                    std_time = np.std(times)
                    stats_text += f"{label}: {len(times)} pax, avg {avg_time:.1f}s, std {std_time:.1f}s\n"
                else:
                    stats_text += f"{label}: 0 pax\n"
            
            ax2.text(1.02, 0.98, stats_text, transform=ax2.transAxes,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    fontsize=9, fontfamily='sans-serif')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Floor waiting time comparison plot saved to: {save_path}")
            
            if show_plot:
                plt.show()
            else:
                plt.close()
    
    def generate_simulation_analysis(self, completed_passengers, save_plots=False, show_plots=None):
        """
        Generate complete analysis report for simulation results
        
        Args:
            completed_passengers: List of passengers who completed service
            save_plots (bool): Whether to save plots to files
            show_plots (bool, optional): Whether to display plots, defaults to use_gui value
        """
        if show_plots is None:
            show_plots = self.use_gui
        
        if not completed_passengers:
            print("No completed passenger data available, skipping simulation analysis")
            return
        
        print(f"\n=== Simulation Results Analysis ===")
        print(f"Number of completed passengers: {len(completed_passengers)}")
        
        # Analyze floor distribution of completed passengers
        completed_floor_counts = {}
        for ps in completed_passengers:
            floor = ps.xi
            if floor not in completed_floor_counts:
                completed_floor_counts[floor] = 0
            completed_floor_counts[floor] += 1
        
        print(f"Completed passengers by floor:")
        for floor in sorted(completed_floor_counts.keys()):
            floor_name = self.floor_names[floor]
            count = completed_floor_counts[floor]
            print(f"  {floor_name}: {count} passengers")
        
        # Generate box plot
        print("Generating waiting time and in-elevator time box plot...")
        boxplot_path = "images/waiting_inside_time_boxplot.png" if save_plots else None
        self.plot_waiting_inside_time_boxplot(completed_passengers, boxplot_path, show_plots)
        
        # Generate floor waiting time comparison plot
        print("Generating floor waiting time comparison plot...")
        floor_comparison_path = "images/floor_waiting_time_comparison.png" if save_plots else None
        self.plot_floor_waiting_time_comparison(completed_passengers, floor_comparison_path, show_plots)
        
        if save_plots:
            print("\nAdditional analysis plots saved to:")
            print("- images/waiting_inside_time_boxplot.png (Waiting and in-elevator time box plot)")
            print("- images/floor_waiting_time_comparison.png (Floor waiting time comparison plot)")
        
        print(f"\nNote: Waiting time analysis only shows completed passengers.")
        print(f"If a floor is not shown in the plots, it means passengers from that floor had not completed service when simulation ended.")

    def generate_analysis_report(self, scenario, save_plots=False, show_plots=None):
        """
        Generate complete analysis report including two plots
        
        Args:
            scenario: Sequence of Arrival objects
            save_plots (bool): Whether to save plots to files
            show_plots (bool, optional): Whether to display plots, defaults to use_gui value
        """
        if show_plots is None:
            show_plots = self.use_gui
            
        print("Starting passenger flow data analysis...")
        
        # Analyze data
        flow_matrix, arrivals_per_minute = self.analyze_passenger_flow(scenario)
        
        print(f"Analysis complete! Processed {len(scenario)} arrival events")
        print(f"Total passengers: {int(np.sum(flow_matrix))}")
        print(f"Simulation duration: {max(arrivals_per_minute.keys()) + 1} minutes")
        
        # Generate heatmap
        heatmap_path = "passenger_flow_heatmap.png" if save_plots else None
        self.plot_passenger_flow_heatmap(flow_matrix, heatmap_path, show_plots)
        
        # Generate histogram
        histogram_path = "arrivals_per_minute_histogram.png" if save_plots else None
        self.plot_arrivals_per_minute_histogram(arrivals_per_minute, histogram_path, show_plots)
        
        # Print detailed statistics
        self._print_detailed_statistics(flow_matrix, arrivals_per_minute)
    
    def _print_detailed_statistics(self, flow_matrix, arrivals_per_minute):
        """Print detailed statistics"""
        print("\n=== Detailed Statistics ===")
        
        # Floor statistics
        print("\nFloor traffic statistics:")
        for i, floor_name in enumerate(self.floor_names):
            outgoing = np.sum(flow_matrix[i, :])  # Total passengers departing from this floor
            incoming = np.sum(flow_matrix[:, i])  # Total passengers arriving at this floor
            print(f"{floor_name:>3}: Departing {outgoing:>3.0f}, Arriving {incoming:>3.0f}, Net flow {incoming-outgoing:>+4.0f}")
        
        # Time statistics
        print(f"\nTime distribution statistics:")
        passenger_counts = list(arrivals_per_minute.values())
        if passenger_counts:
            print(f"Average arrivals per minute: {np.mean(passenger_counts):.1f}")
            print(f"Busiest minute: {max(passenger_counts):.0f}")
            print(f"Quietest minute: {min(passenger_counts):.0f}")
            print(f"Standard deviation: {np.std(passenger_counts):.1f}")

    def plot_floor_waiting_time_comparison_all_passengers(self, all_passengers_data, save_path=None, show_plot=None):
        """
        Plot floor waiting time comparison (including all passengers)
        
        Args:
            all_passengers_data: Dictionary containing all passenger data
                - completed: List of passengers who completed service
                - moving: List of passengers in elevators
                - waiting: List of waiting passengers
                - sim_time: Simulation end time
            save_path (str, optional): Path to save the plot
            show_plot (bool, optional): Whether to display plot, defaults to use_gui value
        """
        if show_plot is None:
            show_plot = self.use_gui
        
        # Use non-interactive backend if not showing plot
        backend = 'Agg' if not show_plot else matplotlib.get_backend()
        
        with temp_matplotlib_backend(backend):
            # Group waiting time data by floor (including all passengers)
            floor_waiting_times = defaultdict(list)
            floor_passenger_counts = defaultdict(lambda: {'completed': 0, 'moving': 0, 'waiting': 0})
            
            # 1. Completed passengers: t1 - t0
            for ps in all_passengers_data['completed']:
                waiting_time = ps.t1 - ps.t0
                floor_waiting_times[ps.xi].append(waiting_time)
                floor_passenger_counts[ps.xi]['completed'] += 1
            
            # 2. Passengers in elevators: t1 - t0
            for ps in all_passengers_data['moving']:
                waiting_time = ps.t1 - ps.t0
                floor_waiting_times[ps.xi].append(waiting_time)
                floor_passenger_counts[ps.xi]['moving'] += 1
            
            # 3. Waiting passengers: sim_time - t0 (assuming entering elevator now)
            sim_time = all_passengers_data['sim_time']
            for ps in all_passengers_data['waiting']:
                waiting_time = sim_time - ps.t0
                floor_waiting_times[ps.xi].append(waiting_time)
                floor_passenger_counts[ps.xi]['waiting'] += 1
            
            # Prepare data for box plot
            floors = sorted(floor_waiting_times.keys())
            waiting_data = [floor_waiting_times[floor] for floor in floors]
            floor_labels = [self.floor_names[floor] for floor in floors]
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 14))
            
            # Box plot
            if waiting_data:
                boxes = ax1.boxplot(waiting_data, labels=floor_labels, patch_artist=True)
                
                # Set different colors for each box
                colors = plt.cm.Set3(np.linspace(0, 1, len(boxes['boxes'])))
                for box, color in zip(boxes['boxes'], colors):
                    box.set_facecolor(color)
                
                ax1.set_title('Floor Waiting Time Comparison (All Passengers)', fontsize=14, fontweight='bold')
                ax1.set_xlabel('Floor', fontsize=12)
                ax1.set_ylabel('Waiting Time (seconds)', fontsize=12)
                ax1.grid(True, alpha=0.3)
                ax1.tick_params(axis='x', rotation=45)
            
            # Average waiting time stacked bar chart
            avg_waiting_times = []
            total_passenger_counts = []
            completed_counts = []
            moving_counts = []
            waiting_counts = []
            
            for floor in floors:
                times = floor_waiting_times[floor]
                counts = floor_passenger_counts[floor]
                
                if times:
                    avg_waiting_times.append(np.mean(times))
                    total_passenger_counts.append(len(times))
                    completed_counts.append(counts['completed'])
                    moving_counts.append(counts['moving'])
                    waiting_counts.append(counts['waiting'])
                else:
                    avg_waiting_times.append(0)
                    total_passenger_counts.append(0)
                    completed_counts.append(0)
                    moving_counts.append(0)
                    waiting_counts.append(0)
            
            # Create stacked bar chart showing passenger counts by status
            width = 0.6
            x_pos = np.arange(len(floor_labels))
            
            # Main average waiting time bars
            bars = ax2.bar(x_pos, avg_waiting_times, width, alpha=0.7, color='skyblue', edgecolor='navy')
            
            ax2.set_title('Average Waiting Time by Floor (All Passengers)', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Floor', fontsize=12)
            ax2.set_ylabel('Average Waiting Time (seconds)', fontsize=12)
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(floor_labels)
            ax2.grid(True, alpha=0.3)
            ax2.tick_params(axis='x', rotation=45)
            
            # Add detailed statistics table
            stats_text = "Floor Statistics (All Passengers):\n"
            total_all = 0
            total_completed = 0
            total_moving = 0
            total_waiting = 0
            
            stats_text += f"\nTotal: {total_all} pax\n"
            stats_text += f"Completed: {total_completed} pax\n"
            stats_text += f"In Elevator: {total_moving} pax\n"
            stats_text += f"Waiting: {total_waiting} pax"
            
            ax2.text(1.02, 0.98, stats_text, transform=ax2.transAxes,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    fontsize=9, fontfamily='sans-serif')

            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Floor waiting time comparison plot (all passengers) saved to: {save_path}")
            
            if show_plot:
                plt.show()
            else:
                plt.close()

    def plot_waiting_inside_time_boxplot_all_passengers(self, all_passengers_data, save_path=None, show_plot=None):
        """
        Plot waiting time and in-elevator time box plots (including all passengers)
        
        Args:
            all_passengers_data: Dictionary containing all passenger data
                - completed: List of passengers who completed service
                - moving: List of passengers in elevators
                - waiting: List of waiting passengers
                - sim_time: Simulation end time
            save_path (str, optional): Path to save the plot
            show_plot (bool, optional): Whether to display plot, defaults to use_gui value
        """
        if show_plot is None:
            show_plot = self.use_gui
        
        # Use non-interactive backend if not showing plot
        backend = 'Agg' if not show_plot else matplotlib.get_backend()
        
        with temp_matplotlib_backend(backend):
            # Prepare waiting time data for all passengers
            all_waiting_times = []
            sim_time = all_passengers_data['sim_time']
            
            # 1. Completed passengers: t1 - t0
            for ps in all_passengers_data['completed']:
                all_waiting_times.append(ps.t1 - ps.t0)
            
            # 2. Passengers in elevators: t1 - t0
            for ps in all_passengers_data['moving']:
                all_waiting_times.append(ps.t1 - ps.t0)
            
            # 3. Waiting passengers: sim_time - t0 (assuming entering elevator now)
            for ps in all_passengers_data['waiting']:
                all_waiting_times.append(sim_time - ps.t0)
            
            # Prepare in-elevator time data for relevant passengers
            all_inside_times = []
            
            # 1. Completed passengers: t2 - t1
            for ps in all_passengers_data['completed']:
                all_inside_times.append(ps.t2 - ps.t1)
            
            # 2. Passengers in elevators: sim_time - t1 (assuming exiting elevator now)
            for ps in all_passengers_data['moving']:
                all_inside_times.append(sim_time - ps.t1)
            
            # Prepare total service time data for all passengers
            all_service_times = []
            
            # 1. Completed passengers: t2 - t0
            for ps in all_passengers_data['completed']:
                all_service_times.append(ps.t2 - ps.t0)
            
            # 2. Passengers in elevators: sim_time - t0
            for ps in all_passengers_data['moving']:
                all_service_times.append(sim_time - ps.t0)
            
            # 3. Waiting passengers: sim_time - t0
            for ps in all_passengers_data['waiting']:
                all_service_times.append(sim_time - ps.t0)
            
            # Check if data exists
            if not all_waiting_times:
                print("No passenger data available for analysis")
                return
            
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # Waiting time box plot
            box1 = axes[0].boxplot(all_waiting_times, patch_artist=True)
            box1['boxes'][0].set_facecolor('lightblue')
            axes[0].set_title('Waiting Time Distribution (All Passengers)', fontsize=14, fontweight='bold')
            axes[0].set_ylabel('Time (seconds)', fontsize=12)
            axes[0].grid(True, alpha=0.3)
            
            # Add statistics
            wait_stats = f'n: {len(all_waiting_times)}\nMean: {np.mean(all_waiting_times):.1f}s\nMedian: {np.median(all_waiting_times):.1f}s\nStd: {np.std(all_waiting_times):.1f}s'
            axes[0].text(0.02, 0.98, wait_stats, transform=axes[0].transAxes, 
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # In-elevator time box plot
            if all_inside_times:
                box2 = axes[1].boxplot(all_inside_times, patch_artist=True)
                box2['boxes'][0].set_facecolor('lightgreen')
                axes[1].set_title('In-Elevator Time Distribution (Relevant Passengers)', fontsize=14, fontweight='bold')
                axes[1].set_ylabel('Time (seconds)', fontsize=12)
                axes[1].grid(True, alpha=0.3)
                
                # Add statistics
                inside_stats = f'n: {len(all_inside_times)}\nMean: {np.mean(all_inside_times):.1f}s\nMedian: {np.median(all_inside_times):.1f}s\nStd: {np.std(all_inside_times):.1f}s'
                axes[1].text(0.02, 0.98, inside_stats, transform=axes[1].transAxes,
                            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            else:
                axes[1].text(0.5, 0.5, 'No in-elevator time data available', transform=axes[1].transAxes,
                           ha='center', va='center', fontsize=12)
                axes[1].set_title('In-Elevator Time Distribution (Relevant Passengers)', fontsize=14, fontweight='bold')
            
            # Total service time box plot
            box3 = axes[2].boxplot(all_service_times, patch_artist=True)
            box3['boxes'][0].set_facecolor('lightyellow')
            axes[2].set_title('Total Service Time Distribution (All Passengers)', fontsize=14, fontweight='bold')
            axes[2].set_ylabel('Time (seconds)', fontsize=12)
            axes[2].grid(True, alpha=0.3)
            
            # Add statistics
            service_stats = f'n: {len(all_service_times)}\nMean: {np.mean(all_service_times):.1f}s\nMedian: {np.median(all_service_times):.1f}s\nStd: {np.std(all_service_times):.1f}s'
            axes[2].text(0.02, 0.98, service_stats, transform=axes[2].transAxes,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.suptitle('Passenger Service Time Analysis (All Passengers)', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Box plots (all passengers) saved to: {save_path}")
            
            if show_plot:
                plt.show()
            else:
                plt.close()

    def generate_simulation_analysis_all_passengers(self, all_passengers_data, save_plots=False, show_plots=None):
        """
        Generate complete simulation analysis report (including all passengers)
        
        Args:
            all_passengers_data: Dictionary containing all passenger data
                - completed: List of passengers who completed service
                - moving: List of passengers in elevators
                - waiting: List of waiting passengers
                - sim_time: Simulation end time
            save_plots (bool): Whether to save plots as files
            show_plots (bool, optional): Whether to display plots, defaults based on use_gui
        """
        if show_plots is None:
            show_plots = self.use_gui
        
        completed_passengers = all_passengers_data['completed']
        moving_passengers = all_passengers_data['moving']
        waiting_passengers = all_passengers_data['waiting']
        
        total_passengers = len(completed_passengers) + len(moving_passengers) + len(waiting_passengers)
        
        print(f"\n=== Simulation Results Analysis (All Passengers) ===")
        print(f"Total passengers: {total_passengers}")
        print(f"Service completed: {len(completed_passengers)}")
        print(f"In elevator: {len(moving_passengers)}")
        print(f"Waiting: {len(waiting_passengers)}")
        
        # Analyze floor distribution for all passengers
        all_floor_counts = defaultdict(lambda: {'completed': 0, 'moving': 0, 'waiting': 0, 'total': 0})
        
        for ps in completed_passengers:
            all_floor_counts[ps.xi]['completed'] += 1
            all_floor_counts[ps.xi]['total'] += 1
        
        for ps in moving_passengers:
            all_floor_counts[ps.xi]['moving'] += 1
            all_floor_counts[ps.xi]['total'] += 1
        
        for ps in waiting_passengers:
            all_floor_counts[ps.xi]['waiting'] += 1
            all_floor_counts[ps.xi]['total'] += 1
        
        print(f"\nPassenger distribution by floor:")
        for floor in sorted(all_floor_counts.keys()):
            floor_name = self.floor_names[floor]
            counts = all_floor_counts[floor]
            print(f"  {floor_name}: {counts['total']} passengers (Completed:{counts['completed']}, In elevator:{counts['moving']}, Waiting:{counts['waiting']})")
        
        # Generate box plots for all passengers
        print("Generating waiting time and in-elevator time box plots (All passengers)...")
        boxplot_path = "images/waiting_inside_time_boxplot_all_passengers.png" if save_plots else None
        self.plot_waiting_inside_time_boxplot_all_passengers(all_passengers_data, boxplot_path, show_plots)
        
        # Generate floor waiting time comparison plot for all passengers
        print("Generating floor waiting time comparison plot (All passengers)...")
        floor_comparison_path = "images/floor_waiting_time_comparison_all_passengers.png" if save_plots else None
        self.plot_floor_waiting_time_comparison_all_passengers(all_passengers_data, floor_comparison_path, show_plots)
        
        if save_plots:
            print("\nAnalysis plots saved:")
            print("- images/waiting_inside_time_boxplot_all_passengers.png (Waiting time and in-elevator time box plots - All passengers)")
            print("- images/floor_waiting_time_comparison_all_passengers.png (Floor waiting time comparison plot - All passengers)")
        
        print(f"\nNote: This analysis includes all arriving passengers, including those still waiting or using elevators at the end of simulation.")