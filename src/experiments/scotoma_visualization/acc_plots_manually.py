import matplotlib.pyplot as plt
import numpy as np

# Data for the models
scotoma_radii = [0, 8, 10, 12, 14, 16]

# Accuracies for the locally connected model
locally_connected_accuracies = [92.53, 91.92, 92.22, 91.31, 90.70, 88.56]
locally_connected_logp = [94.36, 93.44, 92.37, 92.07, 90.85, 89.25]


# Accuracies for the convolutional model
convolutional_accuracies = [94.51, 92.98, 91.76, 87.95, 86.58, 84.14]

convolutional_accuracies_logpolar = [91.98,91.61, 91.46, 91.68, 89.48, 87.56]

# Create the plot
plt.figure(figsize=(10, 6))

# Plotting the data
plt.plot(scotoma_radii, locally_connected_accuracies, marker='o', linestyle='-', label='Locally Connected Model')
plt.plot(scotoma_radii, convolutional_accuracies, marker='s', linestyle='-', label='Convolutional Model')
plt.plot(scotoma_radii, convolutional_accuracies_logpolar, marker='d', linestyle='-', label='Convolutional Model + Logpolar')
plt.plot(scotoma_radii, locally_connected_logp,marker='o', linestyle='-', label='Locally Connected Model + Logpolar')

# Adding title and labels
plt.title("Accuracy with increasing scotoma sizes for convolutional and for locally connected models")
plt.xlabel("Scotoma Radius")
plt.ylabel("Accuracy (%)")
plt.grid(True)
plt.legend()

# Show the plot
plt.show()
