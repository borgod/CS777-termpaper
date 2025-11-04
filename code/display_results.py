
import pandas as pd
import matplotlib.pyplot as plt


# Load the saved results ---
output_path = "feature_comparison_results.csv"  
# or "gs://hw05-bg-5/results/feature_comparison_results.csv"
loaded_results = pd.read_csv(output_path)
print("\n=== Loaded Results ===")
print(loaded_results.to_string(index=False))

# Extract columns for plotting ---
methods = loaded_results["Feature Type"].tolist()
accuracy = loaded_results["Accuracy"].tolist()
runtime = loaded_results["Time (s)"].tolist()

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Subplot 1: Accuracy
axes[0].bar(methods, accuracy, color=['skyblue', 'lightgreen', 'salmon'])
axes[0].set_title("Classification Accuracy")
axes[0].set_ylabel("Accuracy")
axes[0].set_ylim(0, 1)
for i, val in enumerate(accuracy):
    axes[0].text(i, val + 0.02, f"{val:.3f}", ha='center', fontweight='bold')

# Subplot 2: Runtime
axes[1].bar(methods, runtime, color=['skyblue', 'lightgreen', 'salmon'])
axes[1].set_title("Runtime (seconds)")
axes[1].set_ylabel("Time (s)")
for i, val in enumerate(runtime):
    axes[1].text(i, val + (0.05 * max(runtime)), f"{val:.2f}", ha='center', fontweight='bold')

plt.suptitle("Feature Representation Comparison", fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

