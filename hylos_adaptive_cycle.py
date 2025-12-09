import math
import time
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 1. THE ADAPTIVE KERNEL (Identical to Dashboard)
# ==========================================
def new_calculus_adaptive_cycle(f, a, b, tol=1e-6, alpha=8.0, init_n=100):
    h = (b - a) / init_n
    h_min = 1e-7 * (b - a)
    h_max = (b - a) / 5.0  # Allow larger steps for "easy" parts
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
            # If curvature is high, shrink step. If low, grow step.
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
# 2. THE PRESET CYCLE: DEFINING EXPRESSIONS
# ==========================================

# Case A: The Singularity (Original Circle)
# Feature: Infinite curvature at x=1
def func_singularity(x):
    x = min(max(x, 0.0), 1.0)
    return math.sqrt(1 - x**2)

# Case B: The Oscillator (Chirp)
# Feature: Frequency increases with x. Tests rapid response.
def func_oscillator(x):
    return 0.5 * math.sin(20 * x * x)

# Case C: The Pulse (Gaussian Spike)
# Feature: Flat regions (easy) with one sharp event in the middle.
def func_pulse(x):
    return math.exp(-100 * (x - 0.5)**2)

cycle_presets = {
    "Singularity (Circle)": func_singularity,
    "Oscillator (Chirp)": func_oscillator,
    "Localized Pulse": func_pulse
}

# ==========================================
# 3. EXECUTION AND FEATURE EXTRACTION
# ==========================================

def run_cycle_analysis():
    print("Hylos Systems — Adaptive Cycle Analysis")
    print("=======================================")
    
    results = []
    
    # Setup plot grid
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    plt.subplots_adjust(hspace=0.3)
    
    for i, (name, func) in enumerate(cycle_presets.items()):
        print(f"Running Cycle: {name}...")
        
        start_t = time.perf_counter()
        val, df = new_calculus_adaptive_cycle(func, 0.0, 1.0, tol=1e-5)
        end_t = time.perf_counter()
        
        # --- Feature Extraction ---
        min_h = df['h'].min()
        max_h = df['h'].max()
        avg_h = df['h'].mean()
        steps = len(df)
        
        # Dynamic Range: How much does the step size "breathe"?
        dyn_range = max_h / min_h if min_h > 0 else 0
        
        # Compression Factor: How many steps vs a fixed grid of min_h?
        # If we had to use min_h everywhere, we'd need (1.0/min_h) steps.
        # Compression = (Naive Steps) / (Actual Steps)
        naive_steps = 1.0 / min_h
        compression = naive_steps / steps
        
        results.append({
            "Preset": name,
            "Steps": steps,
            "Min h": f"{min_h:.2e}",
            "Max h": f"{max_h:.2e}",
            "Dyn Range": f"{dyn_range:.1f}x",
            "Compression": f"{compression:.1f}x",
            "Runtime": f"{(end_t-start_t)*1000:.2f} ms"
        })
        
        # --- Visualization ---
        ax = axes[i]
        # Plot Step Size (Left Axis)
        ax.plot(df['x'], df['h'], color='tab:blue', label='Step Size (h)', linewidth=1.5)
        ax.set_ylabel("Step Size (h)", color='tab:blue')
        ax.tick_params(axis='y', labelcolor='tab:blue')
        ax.set_yscale('log') # Log scale is crucial for adaptive analysis
        
        # Plot Function Shape (Right Axis) - Reference
        ax2 = ax.twinx()
        y_vals = [func(x) for x in df['x']]
        ax2.plot(df['x'], y_vals, color='tab:gray', alpha=0.3, linestyle='--', label='f(x)')
        ax2.set_ylabel("f(x)", color='tab:gray')
        
        ax.set_title(f"Cycle: {name} | Dynamic Range: {dyn_range:.0f}x")
        if i == 2:
            ax.set_xlabel("Integration Domain (x)")

    # Output Report
    print("\n--- ANALYSIS REPORT ---")
    res_df = pd.DataFrame(results)
    print(res_df.to_string(index=False))
    
    # Save visualization
    plt.savefig("hylos_cycle_analysis.png")
    print("\nVisualization saved to 'hylos_cycle_analysis.png'")

if __name__ == "__main__":
    run_cycle_analysis()
