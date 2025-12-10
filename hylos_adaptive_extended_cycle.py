import math
import time
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 1. THE ADAPTIVE KERNEL
# ==========================================
def new_calculus_adaptive_cycle(f, a, b, tol=1e-6, alpha=8.0, init_n=100):
    """
    Core adaptive algorithm returning total integral and profiling dataframe.
    """
    h = (b - a) / init_n
    h_min = 1e-8 * (b - a) # Slightly finer limit for cusps
    h_max = (b - a) / 5.0
    x, total = a, 0.0
    prev_slope = None
    
    # Analysis Data
    profile_data = {'x': [], 'h': [], 'slope': []}
    
    while x < b:
        # Cap step to not exceed b
        if x + h > b:
            h = b - x

        x_next = x + h
        
        # Emergency break for precision limits
        if x_next <= x:
            break

        y0, y1 = f(x), f(x_next)
        
        # Avoid division by zero
        denom = (x_next - x)
        if denom == 0: break
            
        slope = (y1 - y0) / denom
        arc = denom * math.sqrt(1 + slope * slope)
        total += arc
        
        # Store State
        profile_data['x'].append(x)
        profile_data['h'].append(h)
        profile_data['slope'].append(slope)

        # Hylomorphic Feedback Loop
        if prev_slope is not None:
            curvature = abs(slope - prev_slope)
            if curvature > tol:
                h /= (1 + alpha * curvature)
            else:
                h *= (1 + 0.3 * alpha * (tol - curvature))

        # Clamp Step Size
        h = min(max(h, h_min), h_max)
        
        prev_slope = slope
        x = x_next
        
    return total, pd.DataFrame(profile_data)

# ==========================================
# 2. EXTENDED PRESET CYCLE
# ==========================================

# A. Singularity (Circle)
def func_singularity(x):
    x = min(max(x, 0.0), 1.0)
    return math.sqrt(1 - x**2)

# B. Oscillator (Chirp)
def func_oscillator(x):
    return 0.5 * math.sin(20 * x * x)

# C. Localized Pulse
def func_pulse(x):
    return math.exp(-100 * (x - 0.5)**2)

# D. Mid-Domain Cusp (New)
# Feature: Vertical tangent at x=0.5
def func_cusp(x):
    return math.sqrt(abs(x - 0.5))

# E. Piecewise Corner (New)
# Feature: Discontinuous derivative at x=0.5
def func_piecewise_corner(x):
    if x < 0.5:
        return 2 * x
    else:
        return 2 * (1 - x)

cycle_presets = {
    "Singularity (Circle)": func_singularity,
    "Oscillator (Chirp)": func_oscillator,
    "Localized Pulse": func_pulse,
    "Mid-Domain Cusp": func_cusp,
    "Piecewise Corner": func_piecewise_corner
}

# ==========================================
# 3. EXECUTION AND VISUALIZATION
# ==========================================

def run_extended_analysis():
    print("Hylos Systems — Extended Adaptive Cycle Analysis")
    print("================================================")
    
    results = []
    
    # Create a grid of subplots (5 rows now)
    fig, axes = plt.subplots(5, 1, figsize=(10, 18), sharex=True)
    plt.subplots_adjust(hspace=0.4)
    
    for i, (name, func) in enumerate(cycle_presets.items()):
        print(f"Running Cycle: {name}...")
        
        start_t = time.perf_counter()
        val, df = new_calculus_adaptive_cycle(func, 0.0, 1.0, tol=1e-5)
        end_t = time.perf_counter()
        
        # --- Feature Extraction ---
        min_h = df['h'].min()
        max_h = df['h'].max()
        steps = len(df)
        dyn_range = max_h / min_h if min_h > 0 else 0
        naive_steps = 1.0 / min_h
        compression = naive_steps / steps
        
        results.append({
            "Preset": name,
            "Steps": steps,
            "Min h": f"{min_h:.2e}",
            "Dyn Range": f"{dyn_range:.1f}x",
            "Compression": f"{compression:.1f}x",
            "Runtime": f"{(end_t-start_t)*1000:.2f} ms"
        })
        
        # --- Visualization ---
        ax = axes[i]
        
        # Plot Step Size (Left Axis) - The "Heartbeat" of the algorithm
        ln1 = ax.plot(df['x'], df['h'], color='#007ACC', label='Step Size (h)', linewidth=1.5)
        ax.set_ylabel("Step Size (h)", color='#007ACC', fontweight='bold')
        ax.tick_params(axis='y', labelcolor='#007ACC')
        ax.set_yscale('log') # Log scale reveals the magnitude of adaptation
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
        
        # Plot Function Shape (Right Axis) - Context
        ax2 = ax.twinx()
        y_vals = [func(x) for x in df['x']]
        ln2 = ax2.plot(df['x'], y_vals, color='black', alpha=0.3, linestyle='--', label='f(x)')
        ax2.set_ylabel("f(x)", color='black')
        
        # Combined Legend
        lines = ln1 + ln2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper right', fontsize='small')
        
        ax.set_title(f"Cycle {i+1}: {name} | Compression: {compression:.1f}x", fontsize=11, fontweight='bold')

        if i == 4:
            ax.set_xlabel("Integration Domain (x)", fontsize=12)

    # Output Report
    print("\n--- EXTENDED ANALYSIS REPORT ---")
    res_df = pd.DataFrame(results)
    print(res_df.to_string(index=False))
    
    # Save visualization
    plt.savefig("hylos_extended_cycle_analysis.png")
    print("\nVisualization saved to 'hylos_extended_cycle_analysis.png'")

if __name__ == "__main__":
    run_extended_analysis()
