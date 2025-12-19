import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io

# Read data
df = pd.read_csv("Results/generation_metrics.csv")

# Use a clean style
sns.set_theme(style="whitegrid")

# --- PLOT 1: Duration vs Frames ---
plt.figure(figsize=(10, 6))
sns.lineplot(
    data=df, 
    x='num_frames', 
    y='duration_seconds', 
    hue='pattern', 
    style='pattern', 
    markers=True, 
    dashes=False, 
    linewidth=2.5
)
plt.title('Generation Time vs Frames', fontsize=16, fontweight='bold')
plt.ylabel('Duration (Seconds)', fontsize=14)
plt.xlabel('Number of Frames', fontsize=14)
plt.legend(title='Pattern', fontsize=12)
plt.tight_layout()
plt.savefig('Figures/plot_1_duration.png', dpi=300)
plt.close() # Close to free memory
print("Saved plot_1_duration.png")

# --- PLOT 2: Peak VRAM vs Frames ---
plt.figure(figsize=(10, 6))
sns.lineplot(
    data=df, 
    x='num_frames', 
    y='peak_vram_gb', 
    hue='pattern', 
    style='pattern', 
    markers=True, 
    dashes=False, 
    linewidth=2.5
)
plt.title('Peak VRAM vs Frames', fontsize=16, fontweight='bold')
plt.ylabel('VRAM Usage (GB)', fontsize=14)
plt.xlabel('Number of Frames', fontsize=14)
plt.legend(title='Pattern', fontsize=12)
plt.tight_layout()
plt.savefig('Figures/plot_2_vram.png', dpi=300)
plt.close()
print("Saved plot_2_vram.png")

# --- PLOT 3: Speedup Factor ---
plt.figure(figsize=(10, 6))
pivot_df = df.pivot(index='num_frames', columns='pattern', values='duration_seconds')
pivot_df['speedup'] = pivot_df['dense'] / pivot_df['radial']
pivot_df = pivot_df.dropna()

sns.lineplot(
    x=pivot_df.index, 
    y=pivot_df['speedup'], 
    marker='o', 
    color='green', 
    linewidth=2.5
)
plt.axhline(1.0, color='red', linestyle='--', label='Baseline (1x)')

# Add text labels
for x, y in zip(pivot_df.index, pivot_df['speedup']):
    plt.text(x, y + 0.03, f"{y:.2f}x", ha='center', fontsize=10, fontweight='bold')

plt.title('Speedup Factor (Dense / Radial)', fontsize=16, fontweight='bold')
plt.ylabel('Speedup (x times faster)', fontsize=14)
plt.xlabel('Number of Frames', fontsize=14)
plt.legend()
plt.tight_layout()
plt.savefig('Figures/plot_3_speedup.png', dpi=300)
plt.close()
print("Saved plot_3_speedup.png")