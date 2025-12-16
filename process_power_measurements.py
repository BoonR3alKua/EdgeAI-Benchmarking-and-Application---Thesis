#!/usr/bin/env python3
"""
Energy Measurement Processing for MAX78000FTHR
Processes oscilloscope/power analyzer data to calculate energy consumption
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple
import csv

@dataclass
class PowerMeasurement:
    """Container for power measurement data"""
    time: np.ndarray      # Time in seconds
    voltage: np.ndarray   # Voltage across shunt (V)
    supply_voltage: float = 3.3  # Supply voltage (V)
    shunt_resistance: float = 0.1  # Shunt resistance (Ω)
    
    @property
    def current(self) -> np.ndarray:
        """Calculate current from voltage drop"""
        return self.voltage / self.shunt_resistance
    
    @property
    def power(self) -> np.ndarray:
        """Calculate power consumption"""
        return self.current * self.supply_voltage
    
    @property
    def energy_joules(self) -> float:
        """Calculate total energy in Joules"""
        return np.trapz(self.power, self.time)
    
    @property
    def energy_mj(self) -> float:
        """Calculate total energy in millijoules"""
        return self.energy_joules * 1000
    
    @property
    def energy_uj(self) -> float:
        """Calculate total energy in microjoules"""
        return self.energy_joules * 1_000_000
    
    @property
    def avg_power_mw(self) -> float:
        """Calculate average power in milliwatts"""
        return np.mean(self.power) * 1000
    
    @property
    def peak_power_mw(self) -> float:
        """Calculate peak power in milliwatts"""
        return np.max(self.power) * 1000
    
    @property
    def duration_ms(self) -> float:
        """Measurement duration in milliseconds"""
        return (self.time[-1] - self.time[0]) * 1000


class EnergyAnalyzer:
    """Analyze energy consumption for MAX78000FTHR"""
    
    def __init__(self, supply_voltage: float = 3.3, shunt_resistance: float = 0.1):
        self.supply_voltage = supply_voltage
        self.shunt_resistance = shunt_resistance
        self.measurements = {}
    
    def load_csv(self, filename: str, phase_name: str) -> PowerMeasurement:
        """
        Load measurement data from CSV file
        Expected format: time(s), voltage(V)
        """
        data = np.loadtxt(filename, delimiter=',', skiprows=1)
        time = data[:, 0]
        voltage = data[:, 1]
        
        measurement = PowerMeasurement(
            time=time,
            voltage=voltage,
            supply_voltage=self.supply_voltage,
            shunt_resistance=self.shunt_resistance
        )
        
        self.measurements[phase_name] = measurement
        return measurement
    
    def load_oscilloscope_data(self, filename: str, phase_name: str) -> PowerMeasurement:
        """
        Load data from oscilloscope (various formats supported)
        Modify this based on your specific oscilloscope output format
        """
        # Example for common oscilloscope CSV format
        try:
            with open(filename, 'r') as f:
                # Skip header lines (adjust based on your scope)
                lines = f.readlines()
                data_start = 0
                for i, line in enumerate(lines):
                    if 'Time' in line or 'X' in line:
                        data_start = i + 1
                        break
            
            # Load numerical data
            data = np.loadtxt(filename, delimiter=',', skiprows=data_start)
            time = data[:, 0]
            voltage = data[:, 1]
            
            measurement = PowerMeasurement(
                time=time,
                voltage=voltage,
                supply_voltage=self.supply_voltage,
                shunt_resistance=self.shunt_resistance
            )
            
            self.measurements[phase_name] = measurement
            return measurement
        except Exception as e:
            print(f"Error loading oscilloscope data: {e}")
            return None
    
    def manual_entry(self, phase_name: str, avg_current_ma: float, duration_ms: float) -> dict:
        """
        Calculate energy from manually measured average current and duration
        Useful when using just a multimeter
        """
        avg_current_a = avg_current_ma / 1000
        duration_s = duration_ms / 1000
        
        power_w = avg_current_a * self.supply_voltage
        energy_j = power_w * duration_s
        
        result = {
            'phase': phase_name,
            'avg_current_ma': avg_current_ma,
            'avg_power_mw': power_w * 1000,
            'duration_ms': duration_ms,
            'energy_mj': energy_j * 1000,
            'energy_uj': energy_j * 1_000_000
        }
        
        return result
    
    def analyze_phase(self, phase_name: str) -> dict:
        """Analyze a specific measurement phase"""
        if phase_name not in self.measurements:
            print(f"Phase '{phase_name}' not found in measurements")
            return None
        
        m = self.measurements[phase_name]
        
        results = {
            'phase': phase_name,
            'duration_ms': m.duration_ms,
            'energy_mj': m.energy_mj,
            'energy_uj': m.energy_uj,
            'avg_power_mw': m.avg_power_mw,
            'peak_power_mw': m.peak_power_mw,
            'avg_current_ma': np.mean(m.current) * 1000,
            'peak_current_ma': np.max(m.current) * 1000
        }
        
        return results
    
    def compare_cores(self, arm_phase: str, riscv_phase: str, cnn_phase: str) -> dict:
        """Compare energy consumption between cores"""
        arm_results = self.analyze_phase(arm_phase)
        riscv_results = self.analyze_phase(riscv_phase)
        cnn_results = self.analyze_phase(cnn_phase)
        
        comparison = {
            'ARM': arm_results,
            'RISC-V': riscv_results,
            'CNN': cnn_results,
            'total_energy_mj': (
                arm_results['energy_mj'] + 
                riscv_results['energy_mj'] + 
                cnn_results['energy_mj']
            )
        }
        
        return comparison
    
    def plot_power_profile(self, phase_name: str, save_path: str = None):
        """Plot power consumption over time"""
        if phase_name not in self.measurements:
            print(f"Phase '{phase_name}' not found")
            return
        
        m = self.measurements[phase_name]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Power vs Time
        ax1.plot(m.time * 1000, m.power * 1000, 'b-', linewidth=1.5)
        ax1.axhline(y=m.avg_power_mw, color='r', linestyle='--', 
                   label=f'Avg: {m.avg_power_mw:.2f} mW')
        ax1.set_xlabel('Time (ms)')
        ax1.set_ylabel('Power (mW)')
        ax1.set_title(f'Power Consumption - {phase_name}')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Current vs Time
        ax2.plot(m.time * 1000, m.current * 1000, 'g-', linewidth=1.5)
        ax2.axhline(y=np.mean(m.current) * 1000, color='r', linestyle='--',
                   label=f'Avg: {np.mean(m.current) * 1000:.2f} mA')
        ax2.set_xlabel('Time (ms)')
        ax2.set_ylabel('Current (mA)')
        ax2.set_title(f'Current Consumption - {phase_name}')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_report(self, output_file: str = 'energy_report.txt'):
        """Generate comprehensive energy report"""
        with open(output_file, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("MAX78000FTHR ENERGY MEASUREMENT REPORT\n")
            f.write("=" * 70 + "\n\n")
            
            for phase_name, measurement in self.measurements.items():
                results = self.analyze_phase(phase_name)
                
                f.write(f"\n{'=' * 70}\n")
                f.write(f"PHASE: {phase_name}\n")
                f.write(f"{'=' * 70}\n")
                f.write(f"Duration:        {results['duration_ms']:.3f} ms\n")
                f.write(f"Energy:          {results['energy_mj']:.3f} mJ ({results['energy_uj']:.1f} µJ)\n")
                f.write(f"Avg Power:       {results['avg_power_mw']:.3f} mW\n")
                f.write(f"Peak Power:      {results['peak_power_mw']:.3f} mW\n")
                f.write(f"Avg Current:     {results['avg_current_ma']:.3f} mA\n")
                f.write(f"Peak Current:    {results['peak_current_ma']:.3f} mA\n")
            
            f.write("\n" + "=" * 70 + "\n")
        
        print(f"Report saved to: {output_file}")


def example_usage():
    """Example usage of the energy analyzer"""
    
    # Initialize analyzer
    analyzer = EnergyAnalyzer(supply_voltage=3.3, shunt_resistance=0.1)
    
    # Example 1: Manual entry (using multimeter measurements)
    print("\n=== Manual Entry Example ===")
    cnn_result = analyzer.manual_entry(
        phase_name='CNN_Inference',
        avg_current_ma=10.5,  # Measured with multimeter
        duration_ms=2.5       # From GPIO timing
    )
    print(f"CNN Inference: {cnn_result['energy_mj']:.3f} mJ")
    
    # Example 2: Load from CSV file (if you have oscilloscope data)
    # measurement = analyzer.load_csv('cnn_inference.csv', 'CNN_Inference')
    # results = analyzer.analyze_phase('CNN_Inference')
    # print(f"Energy: {results['energy_mj']:.3f} mJ")
    
    # Example 3: Expected values for MAX78000
    print("\n=== Expected Values (from Analog Devices App Note) ===")
    expected_phases = [
        ('Weight_Load', 19.8, 11.6),   # Current (mA), Duration (ms)
        ('Input_Load', 19.3, 1.1),
        ('CNN_Inference', 37.5, 2.0),
        ('Output_Unload', 10.7, 0.15),
        ('ARM_Softmax', 9.8, 0.5)
    ]
    
    total_energy = 0
    for phase, current_ma, duration_ms in expected_phases:
        result = analyzer.manual_entry(phase, current_ma, duration_ms)
        print(f"{phase:20s}: {result['energy_mj']:.3f} mJ ({result['avg_power_mw']:.2f} mW)")
        total_energy += result['energy_mj']
    
    print(f"\nTotal Energy per Inference: {total_energy:.3f} mJ")
    print(f"Energy per 1000 Inferences: {total_energy:.3f} J")
    
    # Battery life estimation
    print("\n=== Battery Life Estimation ===")
    battery_capacity_mah = 300  # Example: 300mAh battery
    avg_current_ma = total_energy / (sum([d for _, _, d in expected_phases]) / 1000) / 3.3
    inferences_per_second = 1
    
    avg_current_total = avg_current_ma * inferences_per_second + 0.5  # Add idle current
    battery_life_hours = battery_capacity_mah / avg_current_total
    
    print(f"Battery Capacity: {battery_capacity_mah} mAh")
    print(f"Avg Current (1 inf/s): {avg_current_total:.2f} mA")
    print(f"Estimated Battery Life: {battery_life_hours:.1f} hours")
    print(f"Total Inferences: {int(battery_life_hours * 3600 * inferences_per_second):,}")


if __name__ == '__main__':
    example_usage()
