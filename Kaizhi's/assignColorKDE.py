import matplotlib.pyplot as plt
import numpy as np

def value_to_color(value, min_value, max_value, colorGradients):
    
    # Calculate the normalized value within the range [0, 1]
    normalized_value = (value - min_value) / (max_value - min_value)
    index = int(normalized_value * 512 - 2)
    
    print("norm: ",normalized_value)
    print("\nindex: ", index)
    # Interpolate between blue and red based on the normalized value
    interpolated_color = colorGradients[index]
    print("\nassigned color: ", interpolated_color)
    # Return the interpolated color as an integer tuple (B, G, R)
    return interpolated_color

# Define the number of steps in the gradient
num_steps = 511  # You can adjust this as needed

# Create a gradient from blue to red
gradient = np.zeros((num_steps, 3), dtype=np.uint8)
count = 255

for i in range(num_steps):
    # Interpolate between blue and red
    
    if (i < 255):
        gradient[i][0] = i
        gradient[i][1] = i
        gradient[i][2] = 255
    else:
        gradient[i][0] = 255
        gradient[i][1] = count
        gradient[i][2] = count
        count -= 1

color1 = value_to_color(0.2, 0.001, 0.2, gradient)
color2 = value_to_color(0.001, 0.001, 0.2, gradient)

# Display the gradient
plt.imshow([gradient])
plt.axis('off')
plt.show()
