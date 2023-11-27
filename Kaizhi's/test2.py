# Read data from the file
with open("Kaizhi's/kde.txt", "r") as file:
    lines = file.readlines()

# Initialize a set to keep track of unique third elements
unique_third_elements = set()

# Variable to store the sum
sum_of_unique_third_elements = 0
count = 0

# Iterate through each line in the file
for line in lines:
    # Split the line into elements
    element = float(line)
    if element not in unique_third_elements:
        # Add the third element to the set
        unique_third_elements.add(element)

        # Add the third element to the sum
        sum_of_unique_third_elements += element
    count += 1
    print(str(count) + "\n")

# Print the result
print(f"Sum of unique third elements: {sum_of_unique_third_elements}")
