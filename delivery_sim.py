import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

# ---- Create dataset: 100 deliveries, a few deliberately slow ones ----
normal_part = np.random.normal(loc=32, scale=7, size=97)
times = np.concatenate([normal_part, [68, 71, 75]])
times = np.round(np.clip(times, 10, None), 1)

pd.DataFrame({
    "OrderID": range(1, 101),
    "DeliveryTime(min)": times
}).to_csv("delivery_times.csv", index=False)

df = pd.read_csv("delivery_times.csv")
col = df["DeliveryTime(min)"]

# (a) mean and std of real data
mean, std = col.mean(), col.std()
print(f"Real delivery times -- Mean: {mean:.1f} min  Std: {std:.1f} min")

# (b) simulate with same mean/std
simulated = np.random.normal(loc=mean, scale=std, size=100)
print(f"Simulated 100 values from Normal(mean={mean:.1f}, std={std:.1f})")

# (c) overlapping histograms
plt.figure(figsize=(7, 4))
plt.hist(col, bins=15, alpha=0.6, color="steelblue", label="Real")
plt.hist(simulated, bins=15, alpha=0.6, color="indianred", label="Simulated")
plt.xlabel("Delivery Time (min)")
plt.ylabel("Frequency")
plt.title("Real vs Simulated Delivery Times")
plt.legend()
plt.tight_layout()
plt.savefig("delivery_comparison_histogram.png")

# (d) EDA: missing values, boxplot, percentiles
print("Missing values:", df["DeliveryTime(min)"].isnull().sum())

plt.figure(figsize=(4, 4))
plt.boxplot(col)
plt.title("Delivery Time Boxplot")
plt.ylabel("Minutes")
plt.tight_layout()
plt.savefig("delivery_boxplot.png")

q1, q2, q3 = np.percentile(col, [25, 50, 75])
print(f"Percentiles -- 25th: {q1:.1f}  50th: {q2:.1f}  75th: {q3:.1f}")

# (e) IQR outliers (unusually LONG deliveries)
iqr = q3 - q1
lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
print(f"IQR outlier bounds: {lower:.1f} - {upper:.1f}")

outliers = col[col > upper]
print(f"Outliers detected: {len(outliers)} deliveries ({outliers.tolist()} minutes)")
print(f"Percentage flagged as outliers: {len(outliers) / len(col) * 100:.1f}%")

print("Saved delivery_comparison_histogram.png, delivery_boxplot.png")