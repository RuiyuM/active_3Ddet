def unboundedKnapsack(W, wt, val):
    """
    Solve the unbounded knapsack problem using dynamic programming.

    Parameters:
    W (int): Maximum capacity of the knapsack.
    wt (list of int): List containing the weights of the items.
    val (list of int): List containing the values of the items.

    Returns:
    int: The maximum value that can be achieved within the given weight W.
    """
    # Number of items
    n = len(val)

    # Initialize the dp array to store maximum value that can be achieved for given capacity
    dp = [0 for _ in range(W + 1)]

    # Build up the dp array
    for w in range(W + 1):
        for i in range(n):
            if wt[i] <= w:
                # If the item can fit in the current capacity,
                # check if choosing it leads to a better value
                dp[w] = max(dp[w], dp[w - wt[i]] + val[i])

    return dp[W]


# Example usage
W = 100  # Maximum weight of knapsack
wt = [10, 20, 30]  # Weights of the items
val = [60, 100, 120]  # Values of the items

# Call the function
max_value = unboundedKnapsack(W, wt, val)
print(f"Maximum value achievable: {max_value}")
