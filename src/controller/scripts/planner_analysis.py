#!/usr/bin/python3
"""
Planner Comparison Analysis Script
Analyzes and visualizes comparison results from multiple planners
"""

import os
import csv
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import argparse


class PlannerAnalyzer:
    """Analyzes planner comparison data"""

    def __init__(self, data_dir: str = './planner_comparison_data'):
        self.data_dir = data_dir
        self.results = {}
        self.load_data()

    def load_data(self):
        """Load all CSV files from data directory"""
        if not os.path.exists(self.data_dir):
            print(f"Data directory {self.data_dir} not found")
            return

        for filename in os.listdir(self.data_dir):
            if filename.endswith('.csv'):
                filepath = os.path.join(self.data_dir, filename)
                planner_name = filename.split('_')[0]

                if planner_name not in self.results:
                    self.results[planner_name] = []

                try:
                    with open(filepath, 'r') as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            # Convert numeric values
                            for key in row:
                                try:
                                    row[key] = float(row[key])
                                except ValueError:
                                    if row[key].lower() in ['true', 'false']:
                                        row[key] = row[key].lower() == 'true'

                            self.results[planner_name].append(row)
                except Exception as e:
                    print(f"Error reading {filepath}: {e}")

    def get_summary_stats(self) -> dict:
        """Calculate summary statistics for all planners"""
        summary = {}

        for planner_name, data_list in self.results.items():
            if not data_list:
                continue

            df = pd.DataFrame(data_list)

            summary[planner_name] = {
                'num_tests': len(data_list),
                'success_rate': df['success'].sum() / len(df),
                'avg_time_to_goal': df['time_to_goal'].mean(),
                'avg_total_distance': df['total_distance'].mean(),
                'avg_efficiency': df['efficiency'].mean(),
                'avg_collisions': df['collision_count'].mean(),
                'avg_oscillations': df['num_oscillations'].mean(),
                'avg_computation_time': df['avg_computation_time'].mean(),
            }

        return summary

    def print_summary(self):
        """Print summary statistics"""
        summary = self.get_summary_stats()

        print("\n" + "="*80)
        print("PLANNER COMPARISON SUMMARY")
        print("="*80 + "\n")

        for planner_name, stats in summary.items():
            print(f"\n{'='*40}")
            print(f"  {planner_name.upper()} Planner")
            print(f"{'='*40}")
            print(f"  Number of Tests: {stats['num_tests']}")
            print(f"  Success Rate: {stats['success_rate']*100:.1f}%")
            print(f"  Avg Time to Goal: {stats['avg_time_to_goal']:.2f} s")
            print(f"  Avg Total Distance: {stats['avg_total_distance']:.2f} m")
            print(f"  Avg Efficiency: {stats['avg_efficiency']:.3f}")
            print(f"  Avg Near Collisions: {stats['avg_collisions']:.1f}")
            print(f"  Avg Oscillations: {stats['avg_oscillations']:.1f}")
            print(f"  Avg Computation Time: {stats['avg_computation_time']*1000:.2f} ms")
            print()

    def plot_comparison(self, output_dir: str = './planner_comparison_plots'):
        """Create comparison plots"""
        os.makedirs(output_dir, exist_ok=True)

        summary = self.get_summary_stats()

        if not summary:
            print("No data to plot")
            return

        planners = list(summary.keys())
        metrics = [
            ('success_rate', 'Success Rate (%)', lambda x: x * 100),
            ('avg_time_to_goal', 'Time to Goal (s)', lambda x: x),
            ('avg_total_distance', 'Total Distance (m)', lambda x: x),
            ('avg_efficiency', 'Path Efficiency', lambda x: x),
            ('avg_collisions', 'Near Collisions', lambda x: x),
            ('avg_oscillations', 'Oscillations', lambda x: x),
            ('avg_computation_time', 'Computation Time (ms)', lambda x: x * 1000),
        ]

        # Create subplots
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle('Planner Comparison Results', fontsize=16, fontweight='bold')

        axes = axes.flatten()

        for idx, (metric_key, metric_label, transform) in enumerate(metrics):
            ax = axes[idx]

            values = [transform(summary[p].get(metric_key, 0)) for p in planners]
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

            bars = ax.bar(planners, values, color=colors[:len(planners)], alpha=0.7, edgecolor='black')

            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}',
                        ha='center', va='bottom', fontweight='bold')

            ax.set_ylabel(metric_label)
            ax.set_title(metric_label)
            ax.grid(axis='y', alpha=0.3)

        # Remove extra subplots
        for idx in range(len(metrics), len(axes)):
            fig.delaxes(axes[idx])

        plt.tight_layout()
        output_path = os.path.join(output_dir, f'comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to {output_path}")
        plt.show()

    def plot_time_series(self, planner_name: str, output_dir: str = './planner_comparison_plots'):
        """Plot time series of metrics for a specific planner"""
        if planner_name not in self.results or not self.results[planner_name]:
            print(f"No data for planner {planner_name}")
            return

        os.makedirs(output_dir, exist_ok=True)
        data_list = self.results[planner_name]
        df = pd.DataFrame(data_list)

        # Sort by timestamp
        df = df.sort_values('timestamp')

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f'{planner_name.upper()} Planner Time Series', fontsize=14, fontweight='bold')

        # Time to goal
        ax = axes[0, 0]
        ax.plot(df['time_to_goal'], marker='o', linewidth=2, markersize=6, color='#FF6B6B')
        ax.set_ylabel('Time to Goal (s)')
        ax.set_title('Time to Goal Over Tests')
        ax.grid(True, alpha=0.3)

        # Total distance
        ax = axes[0, 1]
        ax.plot(df['total_distance'], marker='o', linewidth=2, markersize=6, color='#4ECDC4')
        ax.set_ylabel('Total Distance (m)')
        ax.set_title('Total Distance Over Tests')
        ax.grid(True, alpha=0.3)

        # Efficiency
        ax = axes[1, 0]
        ax.plot(df['efficiency'], marker='o', linewidth=2, markersize=6, color='#45B7D1')
        ax.set_ylabel('Efficiency')
        ax.set_title('Path Efficiency Over Tests')
        ax.grid(True, alpha=0.3)

        # Collisions
        ax = axes[1, 1]
        ax.bar(range(len(df)), df['collision_count'], color='#F7B731', alpha=0.7, edgecolor='black')
        ax.set_ylabel('Near Collisions')
        ax.set_title('Near Collisions Over Tests')
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        output_path = os.path.join(output_dir, f'timeseries_{planner_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Time series plot saved to {output_path}")
        plt.show()

    def export_json_report(self, output_dir: str = './planner_comparison_plots'):
        """Export summary as JSON report"""
        os.makedirs(output_dir, exist_ok=True)

        summary = self.get_summary_stats()
        report = {
            'timestamp': datetime.now().isoformat(),
            'data_directory': self.data_dir,
            'summary': {},
        }

        for planner_name, stats in summary.items():
            report['summary'][planner_name] = {k: float(v) for k, v in stats.items()}

        output_path = os.path.join(output_dir, f'comparison_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"JSON report saved to {output_path}")
        return report


def main():
    parser = argparse.ArgumentParser(description='Analyze planner comparison results')
    parser.add_argument('--data-dir', type=str, default='./planner_comparison_data',
                        help='Directory containing CSV data files')
    parser.add_argument('--plot-dir', type=str, default='./planner_comparison_plots',
                        help='Output directory for plots')
    parser.add_argument('--timeseries-planner', type=str,
                        help='Generate time series plots for specific planner (dwa, teb, ampcc)')
    parser.add_argument('--json-report', action='store_true',
                        help='Export summary as JSON report')

    args = parser.parse_args()

    # Create analyzer
    analyzer = PlannerAnalyzer(args.data_dir)

    # Print summary
    analyzer.print_summary()

    # Generate comparison plot
    analyzer.plot_comparison(args.plot_dir)

    # Generate time series if requested
    if args.timeseries_planner:
        analyzer.plot_time_series(args.timeseries_planner, args.plot_dir)

    # Export JSON report if requested
    if args.json_report:
        analyzer.export_json_report(args.plot_dir)


if __name__ == "__main__":
    main()
