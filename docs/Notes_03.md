# Quick Adam vs SGD optimizer
```bash
# Your manual implementation (yesterday):
W = W - learning_rate * gradient
# Same learning rate for ALL parameters, forever

# Adam (today):
# - Adapts learning rate per parameter
# - Uses momentum (remembers past gradients)
# - Normalizes by gradient variance
# Result: Faster convergence, better accuracy!
```