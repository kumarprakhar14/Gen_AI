# 📚 Phase 0 Consolidation Guide

## 🎯 Goal: Solidify Your Foundation Before Generative AI

**Time Required:** 2-3 hours  
**Format:** Active recall + hands-on practice  
**Outcome:** Rock-solid understanding of fundamentals

---

## 🗓️ Session Structure (Pick Your Path)

### **Path A: Complete Review** (2.5-3 hours)
All topics, maximum retention

### **Path B: Focused Review** (2 hours)
Core concepts + weak areas

### **Path C: Lightning Review** (1.5 hours)
Key takeaways + practice problems

---

## 📊 PART 1: Active Recall Testing (30-45 min)

### Tensors & Operations
**Close all notes, answer from memory:**

1. What's the difference between `tf.Variable` and a regular tensor?
2. Why do we use `float32` instead of `int32` for neural networks?
3. How do you add a channel dimension to MNIST? (shape transformation)
4. What does `.numpy()` do? When does it involve a copy?
5. What's the output shape after: `tf.reshape(tensor_784, (28, 28, 1))`?

**Check your answers** against your Day 1 notes.

---

### Neural Networks from Scratch
**Without looking at code, explain:**

1. **Forward Pass:**
   - Write the equations: z1 = ?, a1 = ?, z2 = ?, a2 = ?
   - What does each variable represent?

2. **Backpropagation:**
   - Why does `dz2 = y_pred - y_true` work? (softmax + cross-entropy)
   - How do you compute `dW2` from `dz2`?
   - What's the purpose of ReLU derivative in backprop?

3. **Training:**
   - What's the weight update rule?
   - Why do we divide by batch_size in gradient computation?
   - What's the difference between epoch and iteration?

**Verify** your understanding by reviewing your Day 2 code.

---

### CNNs Deep Dive
**Draw and explain (use paper!):**

1. **Convolution Operation:**
   - Draw a 5×5 image and 3×3 filter
   - Show how filter slides (stride=1)
   - Calculate output size
   - Why does output shrink?

2. **Architecture Flow:**
   - Draw: Input → Conv → Pool → Conv → Pool → Flatten → Dense
   - Write shapes at each layer for 28×28 input
   - Why does pooling not have parameters?

3. **Feature Learning:**
   - What do Layer 1 filters learn? (edges, lines)
   - What do Layer 2 filters learn? (shapes, curves)
   - Why hierarchical features matter?

**Compare** your drawings to the CNN theory guide.

---

### Keras & Best Practices
**Quick fire questions:**

1. What does `model.compile()` do? What are the 3 key arguments?
2. What's the difference between `fit()` and `fit_generator()`?
3. Name 3 callbacks and their purposes
4. What's the purpose of validation_split?
5. How do you save/load a model?

---

## 📝 PART 2: Concept Connections (20-30 min)

### Create a Mind Map

On paper or digitally, connect these concepts:

```
              NEURAL NETWORKS
                     |
    ┌────────────────┼────────────────┐
    |                |                |
  TENSORS      ARCHITECTURE      TRAINING
    |                |                |
  Shape          Forward          Loss
  Dtype           Layers        Backprop
  Operations     Activation     Optimizer
```

**Expand each branch** with sub-concepts you learned.

---

### The "Why" Questions

Answer these in your own words (this tests true understanding):

1. **Why do we normalize images to [0, 1]?**
   - Not just "it's standard" - what breaks if we don't?

2. **Why do CNNs beat dense networks for images?**
   - List 3 specific reasons with examples

3. **Why does validation accuracy plateau?**
   - Distinguish between convergence vs overfitting

4. **Why Adam > SGD?**
   - What specifically does Adam do differently?

5. **Why do we use ReLU instead of sigmoid in hidden layers?**
   - What problem does ReLU solve?

---

## 💻 PART 3: Hands-On Practice (45-60 min)

### Challenge 1: Build from Memory (25 min)

**Without looking at previous code**, build a CNN for MNIST:

```python
# Goal: Get >98% accuracy
# Constraints: 
# - Use only 2 Conv layers
# - No data augmentation
# - Max 5 epochs

# Can you do it?
```

**Steps:**
1. Load data
2. Preprocess (reshape, normalize, one-hot)
3. Build model (Conv → Pool → Conv → Pool → Dense)
4. Compile (optimizer, loss, metrics)
5. Train (with validation_split)
6. Evaluate

**If stuck**, peek at your Day 4 code, then try again.

---

### Challenge 2: Debug Broken Code (15 min)

```python
# This code has 5 bugs - find and fix them!

import tensorflow as tf
from tensorflow import keras

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Preprocess
x_train = x_train / 255.0  # Bug 1: Missing reshape
y_train = keras.utils.to_categorical(y_train, 10)

# Model
model = keras.Sequential([
    keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28)),  # Bug 2: Missing channel
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(10, activation='relu')  # Bug 3: Wrong activation
])

# Compile
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train
model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.1)

# Evaluate
model.evaluate(x_test, y_test)  # Bug 4 & 5: x_test and y_test not preprocessed
```

**Bugs to find:**
1. x_train not reshaped to (N, 28, 28, 1)
2. input_shape missing channel dimension
3. Output layer using relu instead of softmax
4. x_test not normalized
5. y_test not one-hot encoded

---

### Challenge 3: Experiment (20 min)

Pick ONE experiment:

**Option A: Architecture Exploration**
- Try 3 Conv layers instead of 2
- Try different filter sizes (3×3 vs 5×5)
- Try different pooling (MaxPool vs AvgPool)
- Compare results

**Option B: Regularization**
- Add Dropout(0.5) before final layer
- Add L2 regularization to Conv layers
- Compare with/without regularization
- Which prevents overfitting better?

**Option C: Data Augmentation**
- Implement rotation, zoom, shift
- Train with/without augmentation
- Compare validation accuracy
- Does it help for MNIST?

---

## 📊 PART 4: Self-Assessment (15 min)

### Rate Your Understanding (1-10)

| Topic | Score | Weak Areas |
|-------|-------|------------|
| Tensors & Operations | /10 | |
| Neural Network Math | /10 | |
| Backpropagation | /10 | |
| Keras API | /10 | |
| CNN Theory | /10 | |
| CNN Implementation | /10 | |
| Data Pipelines | /10 | |
| Debugging & Best Practices | /10 | |

**Total Score:** ___/80

**Score Interpretation:**
- 70-80: Excellent! Ready for Generative AI
- 60-69: Good! Review weak areas
- 50-59: Okay. Spend extra time on low-scoring topics
- <50: Need more practice. Consider redoing some exercises

---

## 🎯 PART 5: Create Your Cheat Sheet (20 min)

### On One Page, Summarize:

**1. Essential Code Snippets**
```python
# Data preprocessing
x = x.reshape(-1, 28, 28, 1).astype('float32') / 255.0
y = keras.utils.to_categorical(y, 10)

# Basic CNN
model = keras.Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D((2,2)),
    ...
])

# Training with callbacks
callbacks = [EarlyStopping(...), ModelCheckpoint(...)]
model.fit(x, y, callbacks=callbacks)
```

**2. Key Formulas**
```
Conv output size: (Input - Kernel) / Stride + 1
Pool output size: Input / Pool_size
Parameters: filters × (kernel_h × kernel_w × input_channels + 1)
```

**3. Debugging Checklist**
```
□ Data preprocessed correctly? (shape, normalization)
□ Model architecture matches input/output?
□ Loss function appropriate for task?
□ Learning rate reasonable? (try 0.001 if stuck)
```