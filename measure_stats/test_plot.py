def maximizeFinalElement(arr):
    # Sort the array first
    arr.sort()

    # Start from the first element, ensuring it's 1 or reducing the first element to 1
    arr[0] = 1

    # Iterate through the array to adjust values as per the constraints
    for i in range(1, len(arr)):
        # If the current element is more than 1 greater than its predecessor,
        # it is adjusted to be exactly 1 greater than the previous element
        if arr[i] > arr[i - 1] + 1:
            arr[i] = arr[i - 1] + 1

    # The last element now represents the maximum value achievable under the constraints
    return arr[-1]


# Example cases
print(maximizeFinalElement([2, 3, 3, 5]))  # Expected output: 4
print(maximizeFinalElement([3, 1, 3, 4]))