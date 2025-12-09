import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import math
import time
from typing import Callable, Dict, List, Tuple

# ==========================================
# 1. CORE ALGORITHMS
# ==========================================

def f(x: float) -> float:
    if x < 0.0: x = 0.0
    elif x > 1.0: x = 1.0
    return math.sqrt(max(0.0, 1.0 - x * x))

def fprime(x: float) -> float:
    if abs(x) >= 1.0:
        x = math.copysign(0.999999, x)
    return -x / math.sqrt(1 - x**2)

# --- Standard Methods ---
def standard_trapezoid(fprime, a, b, n) -> float:
    dx = (b - a) / n
    total = 0.5 * (math.sqrt(1 + fprime(a)**2) + math.sqrt(1 + fprime(b - 1e-12)**2))
    for i in range(1, n):
        x = a + i * dx
        total += math.sqrt(1 + fprime(x)**2)
    return total * dx

def standard_simpson(fprime, a, b, n) -> float:
    if n % 2 == 1: n += 1
    dx = (b - a) / n
    total = math.sqrt(1 + fprime(a)**2) + math.sqrt(1 + fprime(b - 1e-12)**2)
    for i in range(1, n):
        x = a + i * dx
        weight = 4 if i % 2 == 1 else 2
        total += weight * math.sqrt(1 + fprime(x)**2)
    return total * dx / 3.0

# --- Hylos New Calculus Methods ---
def new_calculus_base(f, a, b, n) -> float:
    dx = (b - a) / n
    total = 0.0
    for i in range(n):
        x0, x1 = a + i * dx, a + (i + 1) * dx
        slope = (f(x1) - f(x0)) / dx
        total += math.sqrt(1 + slope * slope)
    return total * dx

def new_calculus_richardson(f, a, b, n) -> float:
    A = new_calculus_base(f, a, b, n)
    A2 = new_calculus_base(f, a, b, 2 * n)
    return (4.0 * A2 - A) / 3.0

def new_calculus_quadratic(f, a, b, n) -> float:
    if n % 2 == 1: n += 1
    dx = (b - a) / n
    total = 0.0
    for j in range(0, n, 2):
        x0 = a + j * dx
        x1 = x0 + dx
        x2 = x0 + 2 * dx
        if x2 > b: break
        f0, f1, f2 = f(x0), f(x1), f(x2)
        # Coefficients
        a_coef = (f2 - 2*f1 + f0) / dx**2
        b_coef = (f2 - f0) / (2*dx) - a_coef * x1
        # Integral approx
        g0 = math.sqrt(1 + (a_coef*x0 + b_coef)**2)
        g1 = math.sqrt(1 + (a_coef*x1 + b_coef)**2)
        g2 = math.sqrt(1 + (a_coef*x2 + b_coef)**2)
        total += (dx / 3.0) * (g0 + 4*g1 + g2)
    return total

# --- Modified for Profiling: Returns (result, profiling_data) ---
def new_calculus_adaptive_fast_profiled(f, a, b, tol=1e-6, alpha=8.0, init_n=100):
    h = (b - a) / init_n
    h_min = 1e-6 * (b - a)
    h_max = (b - a) / 10.0
    x, total = a, 0.0
    prev_slope = None
    
    # PROFILING DATA
    x_points = []
    step_sizes = []
    
    while x < b:
        x_points.append(x)
        step_sizes.append(h)
        
        x_next = min(x + h, b)
        if x_next <= x + 1e-15:
            x_next = x + h_min
            if x_next > b: break

        y0, y1 = f(x), f(x_next)
        slope = (y1 - y0) / (x_next - x)
        arc = (x_next - x) * math.sqrt(1 + slope * slope)
        total += arc

        if prev_slope is not None:
            curvature = abs(slope - prev_slope)
            if curvature > tol:
                h /= (1 + alpha * curvature)
            else:
                h *= (1 + 0.3 * alpha * (tol - curvature))

        h = min(max(h, h_min), h_max)
        prev_slope = slope
        x = x_next
        
    return total, pd.DataFrame({"x": x_points, "Step Size (h)": step_sizes})

# ==========================================
# 2. STREAMLIT UI CONFIGURATION
# ==========================================

st.set_page_config(page_title="Hylos Analytics", layout="wide")

st.markdown("## Hylos Systems | Adaptive Family Framework")
st.markdown("---")

# Sidebar
st.sidebar.header("Configuration")
st.sidebar.markdown("Define the integration parameters for the function $f(x) = \sqrt{1-x^2}$")

a_val = st.sidebar.number_input("Start (a)", value=0.0, step=0.1)
b_val = st.sidebar.number_input("End (b)", value=1.0, max_value=1.0, step=0.1)
exact_val = math.pi / 2  # Known exact value for unit circle quarter

st.sidebar.subheader("Benchmark Settings")
n_start = st.sidebar.number_input("N Start", value=100)
n_end = st.sidebar.number_input("N End", value=5000)
steps_count = st.sidebar.slider("Data Points", 3, 20, 5)

# Generate N values logarithmically spaced
import numpy as np
n_values = np.logspace(math.log10(n_start), math.log10(n_end), num=steps_count).astype(int)
n_values = sorted(list(set(n_values))) # Deduplicate

# ==========================================
# 3. BENCHMARK EXECUTION
# ==========================================

if st.sidebar.button("Run Profiling Benchmark"):
    
    # Container for results
    results = []
    
    # 1. Run Fixed-Step Methods
    methods = {
        "Std-Trap": lambda a, b, n: standard_trapezoid(fprime, a, b, n),
        "Std-Simp": lambda a, b, n: standard_simpson(fprime, a, b, n),
        "New-Rich": lambda a, b, n: new_calculus_richardson(f, a, b, n),
        "New-Quad": lambda a, b, n: new_calculus_quadratic(f, a, b, n),
    }
    
    # Execution loop
    for n in n_values:
        for name, func in methods.items():
            start_t = time.perf_counter()
            val = func(a_val, b_val, n)
            end_t = time.perf_counter()
            
            err = abs(val - exact_val)
            runtime = (end_t - start_t) * 1000 # ms
            
            results.append({
                "Method": name,
                "N": n,
                "Value": val,
                "Abs Error": err,
                "Runtime (ms)": runtime,
                "Type": "Fixed-Step"
            })

    # 2. Run Adaptive Fast (Single run for profiling)
    tols = [1e-4, 1e-5, 1e-6, 1e-7]
    adaptive_profile = None # Store the most precise one

    for t in tols:
        start_t = time.perf_counter()
        val, profile_df = new_calculus_adaptive_fast_profiled(f, a_val, b_val, tol=t, init_n=100)
        end_t = time.perf_counter()
        
        err = abs(val - exact_val)
        runtime = (end_t - start_t) * 1000
        
        # Save the highest precision profile for the deep dive
        if t == 1e-7:
            adaptive_profile = profile_df

        results.append({
            "Method": "New-AdapFast",
            "N": len(profile_df), # Effective steps
            "Value": val,
            "Abs Error": err,
            "Runtime (ms)": runtime,
            "Type": "Adaptive"
        })

    df = pd.DataFrame(results)

    # ==========================================
    # 4. ANALYTICS DASHBOARD
    # ==========================================
    
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Efficiency Frontier: Runtime vs. Error")
        st.markdown(
            "This plot identifies the **Pareto Optimal** methods. "
            "Ideally, a method is in the **bottom-left** (Fastest Time, Lowest Error)."
        )
        
        # Log-Log Plot
        fig_perf = px.scatter(
            df, 
            x="Abs Error", 
            y="Runtime (ms)", 
            color="Method", 
            symbol="Type",
            log_x=True, 
            log_y=True,
            hover_data=["N"],
            size_max=10
        )
        
        fig_perf.update_layout(xaxis_title="Absolute Error (Lower is Better)", yaxis_title="Runtime ms (Lower is Better)")
        st.plotly_chart(fig_perf, use_container_width=True)

    with col2:
        st.subheader("Benchmark Data")
        st.dataframe(
            df[["Method", "N", "Abs Error", "Runtime (ms)"]].sort_values("Abs Error"), 
            height=400,
            use_container_width=True
        )

    st.markdown("---")

    # ==========================================
    # 5. ADAPTIVE PROFILING
    # ==========================================
    
    st.subheader("👁️ Hylomorphic Insight: Adaptive Step Distribution")
    st.markdown(
        """
        This visualization profiles the **New-AdapFast** algorithm's internal state during execution.
        Notice how the step size ($\Delta x$) is **not constant**. The system automatically contracts its stride 
        as it approaches high-curvature regions ($x \\to 1.0$), maximizing accuracy where it matters most while 
        conserving computational resources in linear regions.
        """
    )
    
    if adaptive_profile is not None:
        fig_adapt = go.Figure()

        # Trace 1: Step Size (h)
        fig_adapt.add_trace(go.Scatter(
            x=adaptive_profile["x"], 
            y=adaptive_profile["Step Size (h)"],
            mode='lines',
            name='Step Size (h)',
            line=dict(color='#00CC96', width=2),
            fill='tozeroy',
            fillcolor='rgba(0, 204, 150, 0.1)'
        ))

        fig_adapt.update_layout(
            title="Dynamic Step Rebalancing Profile",
            xaxis_title="Integration Position (x)",
            yaxis_title="Step Size (h)",
            template="plotly_dark",
            height=350
        )
        
        st.plotly_chart(fig_adapt, use_container_width=True)
        
        # Metric highlights
        c1, c2, c3 = st.columns(3)
        c1.metric("Min Step Size", f"{adaptive_profile['Step Size (h)'].min():.2e}")
        c2.metric("Max Step Size", f"{adaptive_profile['Step Size (h)'].max():.2e}")
        c3.metric("Dynamic Range", f"{adaptive_profile['Step Size (h)'].max() / adaptive_profile['Step Size (h)'].min():.0f}x")

else:
    st.info("Adjust parameters in the sidebar and click **Run Profiling Benchmark**.")
