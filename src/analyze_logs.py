import re
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import os

def analyze_pumb_logs(log_file):
    """Comprehensive analysis of PUMB diagnostic logs"""
    
    if not os.path.exists(log_file):
        print(f"❌ Log file not found: {log_file}")
        return
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    # Initialize metrics
    loss_variances = []
    loss_trends = []
    accuracy_trends = []
    client_selections = []
    memory_bank_sizes = []
    round_numbers = []
    aggregation_weights = defaultdict(list)
    loss_improvements = []
    
    for line in lines:
        # Extract round numbers
        round_match = re.search(r'ROUND (\d+)', line)
        if round_match:
            current_round = int(round_match.group(1))
        
        # Loss variance analysis
        if "Loss variance (last 5):" in line:
            variance = float(re.search(r'Loss variance.*?(\d+\.\d+)', line).group(1))
            loss_variances.append(variance)
            round_numbers.append(current_round)
        
        # Loss and accuracy trends
        if "Loss trend:" in line:
            trend = float(re.search(r'Loss trend: ([-+]?\d*\.\d+)', line).group(1))
            loss_trends.append(trend)
        
        if "Accuracy trend:" in line:
            trend = float(re.search(r'Accuracy trend: ([-+]?\d*\.\d+)', line).group(1))
            accuracy_trends.append(trend)
        
        # Client selections
        if "Selected clients:" in line:
            clients_str = line.split("Selected clients:")[1].strip()
            # Extract client IDs from various formats like [1, 2, 3] or (1, 2, 3)
            clients = re.findall(r'\d+', clients_str)
            client_selections.extend([int(c) for c in clients])
        
        # Memory bank size
        if "Memory bank size:" in line:
            size = int(re.search(r'Memory bank size: (\d+)', line).group(1))
            memory_bank_sizes.append(size)
        
        # Aggregation weights (if logged)
        weight_match = re.search(r'Client (\d+).*weight.*?(\d+\.\d+)', line)
        if weight_match:
            client_id = int(weight_match.group(1))
            weight = float(weight_match.group(2))
            aggregation_weights[client_id].append(weight)
        
        # Loss improvements
        if "improvement:" in line:
            improvement = float(re.search(r'improvement: ([-+]?\d*\.\d+)', line).group(1))
            loss_improvements.append(improvement)
    
    # Analysis and reporting
    print(f"=== PUMB LOG ANALYSIS: {os.path.basename(log_file)} ===")
    print(f"Total rounds analyzed: {len(loss_variances)}")
    
    if loss_variances:
        print(f"\n📊 STABILITY METRICS:")
        print(f"  Average loss variance: {np.mean(loss_variances):.4f}")
        print(f"  Max loss variance: {np.max(loss_variances):.4f}")
        print(f"  Variance trend: {np.polyfit(range(len(loss_variances)), loss_variances, 1)[0]:.6f}")
    
    if loss_trends:
        print(f"\n📈 CONVERGENCE ANALYSIS:")
        improving_rounds = sum(1 for trend in loss_trends if trend < 0)
        print(f"  Rounds with improving loss: {improving_rounds}/{len(loss_trends)} ({100*improving_rounds/len(loss_trends):.1f}%)")
        print(f"  Average loss trend: {np.mean(loss_trends):.6f}")
    
    if accuracy_trends:
        improving_acc = sum(1 for trend in accuracy_trends if trend > 0)
        print(f"  Rounds with improving accuracy: {improving_acc}/{len(accuracy_trends)} ({100*improving_acc/len(accuracy_trends):.1f}%)")
    
    if client_selections:
        print(f"\n👥 CLIENT SELECTION ANALYSIS:")
        selection_counts = Counter(client_selections)
        total_selections = len(client_selections)
        most_selected = selection_counts.most_common(5)
        print(f"  Total client selections: {total_selections}")
        print(f"  Unique clients selected: {len(selection_counts)}")
        print(f"  Most selected clients: {most_selected}")
        
        # Selection fairness
        if len(selection_counts) > 1:
            selection_std = np.std(list(selection_counts.values()))
            print(f"  Selection fairness (lower=more fair): {selection_std:.2f}")
    
    if memory_bank_sizes:
        print(f"\n🧠 MEMORY BANK EVOLUTION:")
        print(f"  Final size: {memory_bank_sizes[-1]}")
        print(f"  Growth rate: {(memory_bank_sizes[-1] - memory_bank_sizes[0])/len(memory_bank_sizes):.2f} per round")
    
    if loss_improvements:
        print(f"\n📉 CLIENT IMPROVEMENT ANALYSIS:")
        positive_improvements = [imp for imp in loss_improvements if imp > 0]
        print(f"  Average loss improvement: {np.mean(loss_improvements):.4f}")
        print(f"  Clients with positive improvement: {len(positive_improvements)}/{len(loss_improvements)}")
    
    # Stability assessment
    print(f"\n🎯 OVERALL ASSESSMENT:")
    if loss_variances:
        avg_variance = np.mean(loss_variances)
        if avg_variance > 0.5:
            print("  ⚠️  HIGH INSTABILITY detected - Consider reducing learning rate")
        elif avg_variance > 0.2:
            print("  ⚠️  MODERATE INSTABILITY detected - Monitor closely")
        else:
            print("  ✅ STABLE training detected")
    
    # Detect problematic patterns
    warnings = []
    if loss_trends and np.mean(loss_trends) > 0.01:
        warnings.append("Loss trending upward (possible divergence)")
    if loss_variances and np.max(loss_variances) > 1.0:
        warnings.append("Very high loss variance detected")
    if client_selections and len(set(client_selections)) < len(client_selections) * 0.3:
        warnings.append("Low client diversity in selection")
    
    if warnings:
        print(f"\n⚠️  WARNINGS:")
        for warning in warnings:
            print(f"  • {warning}")
    
    return {
        'loss_variances': loss_variances,
        'loss_trends': loss_trends,
        'accuracy_trends': accuracy_trends,
        'client_selections': client_selections,
        'memory_bank_sizes': memory_bank_sizes,
        'round_numbers': round_numbers
    }

def plot_pumb_analysis(log_file, save_path=None):
    """Create visualization plots from PUMB log analysis"""
    
    metrics = analyze_pumb_logs(log_file)
    
    if not any(metrics.values()):
        print("No data to plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'PUMB Analysis: {os.path.basename(log_file)}', fontsize=16)
    
    # Loss variance over time
    if metrics['loss_variances'] and metrics['round_numbers']:
        axes[0,0].plot(metrics['round_numbers'], metrics['loss_variances'], 'r-', linewidth=2)
        axes[0,0].set_title('Loss Variance Over Time')
        axes[0,0].set_xlabel('Round')
        axes[0,0].set_ylabel('Loss Variance')
        axes[0,0].grid(True, alpha=0.3)
    
    # Client selection distribution
    if metrics['client_selections']:
        selection_counts = Counter(metrics['client_selections'])
        clients = list(selection_counts.keys())
        counts = list(selection_counts.values())
        axes[0,1].bar(clients, counts)
        axes[0,1].set_title('Client Selection Frequency')
        axes[0,1].set_xlabel('Client ID')
        axes[0,1].set_ylabel('Times Selected')
    
    # Memory bank growth
    if metrics['memory_bank_sizes']:
        axes[1,0].plot(range(len(metrics['memory_bank_sizes'])), metrics['memory_bank_sizes'], 'g-', linewidth=2)
        axes[1,0].set_title('Memory Bank Size Growth')
        axes[1,0].set_xlabel('Measurement')
        axes[1,0].set_ylabel('Size')
        axes[1,0].grid(True, alpha=0.3)
    
    # Trends analysis
    if metrics['loss_trends'] and metrics['accuracy_trends']:
        axes[1,1].scatter(metrics['loss_trends'], metrics['accuracy_trends'], alpha=0.6)
        axes[1,1].set_title('Loss vs Accuracy Trends')
        axes[1,1].set_xlabel('Loss Trend')
        axes[1,1].set_ylabel('Accuracy Trend')
        axes[1,1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        axes[1,1].axvline(x=0, color='k', linestyle='--', alpha=0.5)
        axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()

# Usage examples
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        log_file = sys.argv[1]
    else:
        log_file = "../save/logs/pumb_diagnostics_mnist_cnn_iid[True]_alpha[NA].log"

    if os.path.exists(log_file):
        analyze_pumb_logs(log_file)
        # If you have a plotting function, call it here as well
        # plot_pumb_analysis(log_file)
    else:
        print(f"Log file not found: {log_file}")
        print("Available log files:")
        log_dir = os.path.dirname(log_file) or "."
        for file in os.listdir(log_dir):
            if file.endswith('.log'):
                print(f"  {file}")
