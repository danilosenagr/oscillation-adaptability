import numpy as np
import matplotlib.pyplot as plt

# Create a simple plot
x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.figure()
plt.plot(x, y)
plt.title("Test Plot")
plt.savefig('/Users/bobbarclay/osolationadaptability/figures/test_plot.png')
print("Test plot saved successfully!")