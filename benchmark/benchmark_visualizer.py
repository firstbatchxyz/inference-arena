import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from pathlib import Path


class BenchmarkVisualizer:
    """
    A focused visualizer for benchmark results that displays key metrics:
    - Avg. TTFT (Time To First Token)
    - Input Token/Request  
    - Avg. Throughput
    - Total Requests
    - Throughput performance chart
    - Latency vs Load analysis
    """
    
    def __init__(self, results_file: str):
        """
        Initialize the visualizer with benchmark results.
        
        Args:
            results_file: Path to the benchmark results JSON file
        """
        self.results_file = results_file
        self.data = self._load_data()
        self.benchmark = self.data['benchmarks'][0] if self.data['benchmarks'] else None
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def _load_data(self) -> Dict[str, Any]:
        """Load benchmark data from JSON file."""
        with open(self.results_file, 'r') as f:
            return json.load(f)
    
    def _extract_metrics(self) -> Dict[str, Any]:
        """Extract key metrics from benchmark data."""
        if not self.benchmark:
            return {}
            
        metrics = self.benchmark.get('metrics', {})
        return {
            'requests_per_second': metrics.get('requests_per_second', {}),
            'request_latency': metrics.get('request_latency', {}),
            'prompt_token_count': metrics.get('prompt_token_count', {}),
            'output_token_count': metrics.get('output_token_count', {}),
            'time_to_first_token_ms': metrics.get('time_to_first_token_ms', {}),
            'inter_token_latency_ms': metrics.get('inter_token_latency_ms', {}),
            'time_per_output_token_ms': metrics.get('time_per_output_token_ms', {})
        }
    
    def _extract_request_data(self) -> pd.DataFrame:
        """Extract individual request data as a DataFrame."""
        if not self.benchmark or not self.benchmark.get('requests', {}).get('successful'):
            return pd.DataFrame()
        
        requests = self.benchmark['requests']['successful']
        
        # Convert to DataFrame
        df_data = []
        for req in requests:
            df_data.append({
                'request_id': req.get('request_id', ''),
                'start_time': req.get('start_time', 0),
                'end_time': req.get('end_time', 0),
                'request_latency': req.get('request_latency', 0),
                'prompt_tokens': req.get('prompt_tokens', 0),
                'output_tokens': req.get('output_tokens', 0),
                'tokens_per_second': req.get('tokens_per_second', 0),
                'output_tokens_per_second': req.get('output_tokens_per_second', 0),
                'time_to_first_token_ms': req.get('time_to_first_token_ms', 0),
                'time_per_output_token_ms': req.get('time_per_output_token_ms', 0),
                'inter_token_latency_ms': req.get('inter_token_latency_ms', 0),
            })
        
        df = pd.DataFrame(df_data)
        
        # Convert timestamps to relative time from start
        if not df.empty:
            start_time = df['start_time'].min()
            df['relative_start_time'] = df['start_time'] - start_time
            df['relative_end_time'] = df['end_time'] - start_time
        
        return df
    
    def create_metrics_dashboard(self, save_path: Optional[str] = None) -> None:
        """
        Create a focused single-page dashboard with key metrics and performance graphs.
        
        Displays:
        - Avg. TTFT (Time To First Token)
        - Input Token/Request  
        - Avg. Throughput
        - Total Requests
        - Throughput comparison chart
        - Latency vs Load scatter plot
        
        Args:
            save_path: Optional path to save the dashboard
        """
        metrics = self._extract_metrics()
        df = self._extract_request_data()
        
        # Set up the figure with custom layout
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 4, height_ratios=[1, 2, 2], width_ratios=[1, 1, 1, 1])
        
        # Color scheme similar to the examples
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        # === TOP ROW: KEY METRICS ===
        
        # Metric 1: Avg. TTFT
        ax1 = fig.add_subplot(gs[0, 0])
        ttft_mean_ms = metrics.get('time_to_first_token_ms', {}).get('successful', {}).get('mean', 0)
        ttft_mean_s = ttft_mean_ms / 1000.0  # Convert from milliseconds to seconds
        ax1.text(0.5, 0.7, f'{ttft_mean_s:.3f}s', ha='center', va='center', 
                fontsize=24, fontweight='bold', color=colors[0])
        ax1.text(0.5, 0.3, 'Avg. TTFT', ha='center', va='center', 
                fontsize=14, color='gray')
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.axis('off')
        ax1.patch.set_facecolor('#f8f9fa')
        
        # Metric 2: Input Tokens/Request
        ax2 = fig.add_subplot(gs[0, 1])
        avg_input_tokens = df['prompt_tokens'].mean() if not df.empty else 0
        ax2.text(0.5, 0.7, f'{avg_input_tokens:.0f}', ha='center', va='center', 
                fontsize=24, fontweight='bold', color=colors[1])
        ax2.text(0.5, 0.3, 'Input Token/Request', ha='center', va='center', 
                fontsize=14, color='gray')
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.axis('off')
        ax2.patch.set_facecolor('#f8f9fa')
        
        # Metric 3: Avg. Throughput
        ax3 = fig.add_subplot(gs[0, 2])
        avg_throughput = metrics.get('requests_per_second', {}).get('successful', {}).get('mean', 0)
        ax3.text(0.5, 0.7, f'{avg_throughput:.2f}', ha='center', va='center', 
                fontsize=24, fontweight='bold', color=colors[2])
        ax3.text(0.5, 0.3, 'Avg. Throughput\n(req/s)', ha='center', va='center', 
                fontsize=14, color='gray')
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)
        ax3.axis('off')
        ax3.patch.set_facecolor('#f8f9fa')
        
        # Metric 4: Total Requests
        ax4 = fig.add_subplot(gs[0, 3])
        total_requests = len(df) if not df.empty else 0
        ax4.text(0.5, 0.7, f'{total_requests}', ha='center', va='center', 
                fontsize=24, fontweight='bold', color=colors[3])
        ax4.text(0.5, 0.3, 'Total Requests', ha='center', va='center', 
                fontsize=14, color='gray')
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        ax4.patch.set_facecolor('#f8f9fa')
        
        # === MIDDLE ROW: THROUGHPUT COMPARISON ===
        ax5 = fig.add_subplot(gs[1, :])
        
        if not df.empty:
            # Create throughput comparison similar to example
            # Group requests by time windows to show throughput variations
            df_sorted = df.sort_values('start_time')
            
            # Calculate rolling throughput over time windows
            window_size = max(1, len(df) // 20)  # 20 data points
            throughput_data = []
            time_windows = []
            
            for i in range(0, len(df_sorted), window_size):
                window_df = df_sorted.iloc[i:i+window_size]
                if len(window_df) > 0:
                    time_span = window_df['end_time'].max() - window_df['start_time'].min()
                    if time_span > 0:
                        throughput = len(window_df) / time_span
                        throughput_data.append(throughput)
                        time_windows.append(f'Window {len(time_windows)+1}')
            
            if throughput_data:
                bars = ax5.bar(range(len(throughput_data)), throughput_data, 
                              color=[colors[i % len(colors)] for i in range(len(throughput_data))],
                              alpha=0.8, edgecolor='white', linewidth=0.5)
                
                ax5.set_xlabel('Time Windows', fontsize=12, fontweight='bold')
                ax5.set_ylabel('Throughput (requests/s)', fontsize=12, fontweight='bold')
                ax5.set_title('How do I benchmark the minimum latency of an LLM engine?', fontsize=14, fontweight='bold', pad=20)
                ax5.grid(True, alpha=0.3, axis='y')
                ax5.set_facecolor('#fafafa')
                
                # Add value labels on bars
                for bar, value in zip(bars, throughput_data):
                    height = bar.get_height()
                    ax5.text(bar.get_x() + bar.get_width()/2., height + max(throughput_data)*0.01,
                            f'{value:.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        else:
            ax5.text(0.5, 0.5, 'No data available for throughput analysis', 
                    ha='center', va='center', fontsize=14, transform=ax5.transAxes)
            ax5.set_title('How do I benchmark the minimum latency of an LLM engine?', fontsize=14, fontweight='bold')
        
        # === BOTTOM ROW: LATENCY vs LOAD ANALYSIS ===
        ax6 = fig.add_subplot(gs[2, :])
        
        if not df.empty and len(df) > 1:
            # Create scatter plot similar to the latency vs load example
            df_sorted = df.sort_values('start_time').reset_index(drop=True)
            
            # Calculate concurrent requests at each point in time
            concurrent_load = []
            latencies = []
            
            for idx, row in df_sorted.iterrows():
                # Count how many requests were running at the start of this request
                concurrent = 0
                current_start = row['start_time']
                current_end = row['end_time']
                
                for _, other_row in df_sorted.iterrows():
                    if (other_row['start_time'] <= current_start <= other_row['end_time'] or
                        other_row['start_time'] <= current_end <= other_row['end_time'] or
                        current_start <= other_row['start_time'] <= current_end):
                        concurrent += 1
                
                concurrent_load.append(concurrent)
                latencies.append(row['request_latency'])
            
            # Create the scatter plot with connected lines (similar to example)
            scatter = ax6.scatter(concurrent_load, latencies, 
                                 c=range(len(latencies)), cmap='viridis', 
                                 alpha=0.6, s=30, edgecolors='white', linewidth=0.5)
            
            # Add trend line
            if len(concurrent_load) > 1:
                z = np.polyfit(concurrent_load, latencies, 1)
                p = np.poly1d(z)
                ax6.plot(concurrent_load, p(concurrent_load), "r--", alpha=0.8, linewidth=2)
            
            ax6.set_xlabel('Concurrent Request Load', fontsize=12, fontweight='bold')
            ax6.set_ylabel('Latency (s)', fontsize=12, fontweight='bold')
            ax6.set_title('How do latency and throughput for LLM engines depend on request load?', 
                         fontsize=14, fontweight='bold', pad=20)
            ax6.grid(True, alpha=0.3)
            ax6.set_facecolor('#fafafa')
            
            # Set log scale for y-axis if there's significant variation
            if max(latencies) / min(latencies) > 10:
                ax6.set_yscale('log')
                ax6.set_ylabel('Latency (s) - Log Scale', fontsize=12, fontweight='bold')
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax6, pad=0.02)
            cbar.set_label('Request Order', fontsize=10)
            
        else:
            ax6.text(0.5, 0.5, 'Insufficient data for latency vs load analysis', 
                    ha='center', va='center', fontsize=14, transform=ax6.transAxes)
            ax6.set_title('How do latency and throughput for LLM engines depend on request load?', 
                         fontsize=14, fontweight='bold')
        
        # Final layout adjustments
        plt.tight_layout(pad=3.0)
        
        # Add overall title
        fig.suptitle('LLM Performance Dashboard', fontsize=18, fontweight='bold', y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Metrics dashboard saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()


# Example usage
if __name__ == "__main__":
    # Create visualizer instance
    visualizer = BenchmarkVisualizer("benchmark_results.json")
    
    # Create the focused metrics dashboard
    print("Creating metrics dashboard...")
    visualizer.create_metrics_dashboard()
    
    # Save to file
    print("Saving metrics dashboard...")
    visualizer.create_metrics_dashboard("metrics_dashboard.png")
