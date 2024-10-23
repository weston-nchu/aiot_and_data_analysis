# Prompt
# give me a sample python code to implements condition below
# 1. give me x list random num between 0 to 1000
# 2. there are threshold 200 and 800
# 3. give me y if in threshold then return 1 else 0
# 4. use pylot draw the final reult

import numpy as np
import matplotlib.pyplot as plt

# Step 1: Generate random numbers between 0 and 1000
n = 100  # Number of random values
x = np.random.randint(0, 1001, n)

# Step 2: Set thresholds
threshold_low = 200
threshold_high = 800

# Step 3: Generate y based on thresholds
y = np.where((x >= threshold_low) & (x <= threshold_high), 1, 0)

# Step 4: Plot the result
plt.figure(figsize=(10, 6))
plt.scatter(x, y, c='blue', label='Data points')
plt.axvline(threshold_low, color='red', linestyle='--', label='Threshold 200')
plt.axvline(threshold_high, color='green', linestyle='--', label='Threshold 800')
plt.title('Random Numbers vs Threshold')
plt.xlabel('Random numbers (x)')
plt.ylabel('Threshold indicator (y)')
plt.legend()
plt.show()
