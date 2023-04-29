import matplotlib.pyplot as plt
import numpy as np

# Create a NumPy array of x values
x = np.linspace(0, 10, 100)

# Create a NumPy array of y values
y = np.sin(x)

# Create a line plot of the data
plt.plot(x, y)

# Add labels to the plot
plt.title('Sine Wave')
plt.xlabel('x')
plt.ylabel('y')

# Show the plot
plt.show()