# 🚀 Machine Learning Part 3: SVM, Random Forest & XGBoost
## (Advanced Classification Algorithms)

---

## 📚 สารบัญ

1. **Support Vector Machine (SVM)**: Linear & Non-linear Separation
2. **Random Forest**: Ensemble Method with Bagging
3. **Gradient Boosting & XGBoost**: Boosting Method
4. **เปรียบเทียบ 3 algorithms + Previous 4**
5. **Code ตัวอย่าง**

---

# 🔴 Part 1: SUPPORT VECTOR MACHINE (SVM)

## 1.1 ความเข้าใจพื้นฐาน

### Goal: Find Optimal Hyperplane

```
Hyperplane = line/surface ที่ separate 2 classes
Dimension = 1D (line) ใน 2D space
         = 2D (plane) ใน 3D space
         = nD surface ใน (n+1)D space

Example: 2D Data
┌─────────────────────────────────────┐
│  ●●●●                               │
│    ●●●●                             │
│        ___________  ← Hyperplane    │
│                                     │
│              ◆◆◆◆                   │
│                ◆◆◆◆                 │
└─────────────────────────────────────┘

Goal: Find line that separate ● and ◆
```

---

## 1.2 Hard Margin vs Soft Margin

### Hard Margin (Ideal)
```
Assumption: Data completely separable
Problem: ❌ Real data ไม่สมบูรณ์ separable!

┌─────────────────────────────────────┐
│  ●●● ___ ◆◆◆                       │
│      |   |    Margin                │
│    ●●|___|◆◆                       │
│                                     │
│  Perfect separation!                │
└─────────────────────────────────────┘
```

### Soft Margin (Realistic)
```
Allow: Some points inside margin ✓
Penalty: Cases in margin ได้ penalty

┌─────────────────────────────────────┐
│  ●●● ●___ ◆◆◆                     │
│    ●●|_●_|◆◆ ← Some ● inside margin│
│      |   |                          │
│  Real data!                         │
└─────────────────────────────────────┘

Hyperparameter C: Controls penalty
- High C = Hard margin (ตึง)
- Low C = Soft margin (หลวม)
```

---

## 1.3 Non-linear Data: Kernel Trick

### Problem: Data Not Linearly Separable

```
Original 2D:
┌─────────────────────────────────────┐
│    ●    ◆◆◆                        │
│  ●●●●  ◆◆◆◆                       │
│  ●●●●  ◆◆◆◆                       │
│    ●    ◆◆◆                        │
│                                     │
│ Cannot separate with straight line!│
└─────────────────────────────────────┘

Solution: Add extra dimension (Kernel)!
```

### Kernel Trick: Project to Higher Dimension

```
Original 2D Data:
┌─────────────────────────────────────┐
│        ●          ◆◆◆              │
│      ●●●●        ◆◆◆◆             │
│      ●●●●        ◆◆◆◆             │
│        ●          ◆◆◆              │
└─────────────────────────────────────┘

Apply Kernel → 3D Data:
        ◆◆◆
       ◆◆◆◆◆
      ◆◆◆◆◆◆
     /       \
    /         \
   /___●___●___\ ← Separated!
      ●●●●●●

Now separable with plane! ✓
```

---

## 1.4 Kernel Functions

```
┌─────────────┬──────────────┬──────────────────┐
│ Kernel      │ Formula      │ When to Use      │
├─────────────┼──────────────┼──────────────────┤
│ Linear      │ K(x,y) = x·y │ Linearly         │
│             │              │ separable data   │
├─────────────┼──────────────┼──────────────────┤
│ Polynomial  │ K = (x·y+1)^d│ Smooth curves   │
│ (degree d)  │              │ d=2 or 3 common │
├─────────────┼──────────────┼──────────────────┤
│ RBF         │ K = exp(-γ║) │ Most popular!   │
│ (Gaussian)  │ x-y║²)      │ Flexible        │
├─────────────┼──────────────┼──────────────────┤
│ Sigmoid     │ K = tanh(x·y)│ Neural networks │
└─────────────┴──────────────┴──────────────────┘
```

---

## 1.5 Key Hyperparameters

```
C (Cost): Controls margin softness
├─ High C (1000): Hard margin
│  └─ Few misclassifications, overfit risk
└─ Low C (0.1): Soft margin
   └─ More misclassifications, generalize better

Kernel: How to transform data
├─ linear: Linearly separable
├─ polynomial: Curved boundaries
├─ rbf: Most flexible (default)
└─ sigmoid: Neural network-like

gamma (for RBF/Sigmoid): Point influence
├─ High gamma: Each point matters → Overfit
└─ Low gamma: Smooth decision → Underfit

degree (for Polynomial): Complexity
├─ degree=2: Simple curve
└─ degree=3+: Complex curves
```

---

## 1.6 Support Vectors

```
Support Vector = ข้อมูลที่อยู่บนหรือใกล้ margin

Important:
- Only support vectors affect hyperplane
- Non-support vectors: ไม่มีอิทธิพล
- More support vectors = More complex boundary

Example:
┌────────────────────────────┐
│  ●     ◆ ◇ ← support vector│
│ ●● ◇ ◆◆                   │
│  ● ◆◆                     │
│                            │
│ Only ●,◆,◇ matter!        │
│ Other ● ปกติไม่ matter     │
└────────────────────────────┘
```

---

# 🟠 Part 2: RANDOM FOREST

## 2.1 ความเข้าใจพื้นฐาน: Ensemble Methods

### Concept: "Wisdom of Crowds"

```
❌ Single Decision Tree:
   - Tend to overfit
   - High variance
   - Unstable

✅ Multiple Trees (Forest):
   - Lower variance (averaging effect)
   - Less overfit
   - More stable
   - Better predictions!
```

### Bagging: Bootstrap Aggregating

```
Step 1: Create bootstrap samples (sample with replacement)
Step 2: Train tree on each sample
Step 3: Combine predictions (voting/averaging)

Example with 3 trees:
┌─────────────────────────────────────┐
│ Training Data (10 samples):         │
│ A B C D E F G H I J                 │
└─────────────────────────────────────┘
              │
      ┌───────┼───────┬───────┐
      │       │       │       │
   Bootstrap Bootstrap Bootstrap
   A,C,E,F,B A,A,D,G,I E,F,H,I,J
      │       │       │
      ▼       ▼       ▼
    Tree1   Tree2   Tree3
      │       │       │
   Pred A   Pred B  Pred A
      └───────┼───────┘
              │
          Vote: A wins!
              │
              ▼
          Final: A
```

---

## 2.2 Random Forest: Extra Randomness

```
Random Forest = Bagging + Random Feature Selection

At each split in each tree:
- Randomly select subset of features
- Choose best split from that subset

Why?
- Creates uncorrelated trees
- Prevents strong features from dominating
- Better ensemble performance!
```

---

## 2.3 How Random Forest Works

```
1. Create multiple bootstrap samples
2. For each sample:
   a. Build decision tree
   b. At each split, randomly select k features
   c. Find best split from those k features
3. Combine all trees
4. Predict:
   - Classification: Majority vote
   - Regression: Average prediction

Key: Each tree is different!
     Different samples + Different features
     = Low correlation between trees
     = Better ensemble
```

---

## 2.4 Hyperparameters

```
n_trees: Number of trees
├─ 10-100: Usually good
├─ 500: Often default
└─ 1000+: Rarely needed

mtry: Number of features to sample at each split
├─ p/3: Regression default (p = total features)
├─ √p: Classification default
└─ Tune to find best

max_depth: Max tree depth
├─ Unlimited: Default (grow until pure)
└─ Limit: Prevent overfitting

min_samples_leaf: Min samples in leaf node
├─ 1: Default
└─ Higher: Prevent overfitting
```

---

# 🟡 Part 3: GRADIENT BOOSTING & XGBOOST

## 3.1 Boosting Concept

### Different from Bagging

```
BAGGING (Random Forest):
- Train trees in PARALLEL
- Independent of each other
- Reduce variance

BOOSTING:
- Train trees SEQUENTIALLY
- Each tree learns from mistakes of previous
- Reduce BIAS + reduce variance
```

---

## 3.2 How Gradient Boosting Works

```
Step 1: Train Tree 1 on full data
        Predict: ŷ₁
        Error (Residual): r₁ = y - ŷ₁

Step 2: Train Tree 2 on RESIDUALS r₁
        Predict: ŷ₂
        New Error: r₂ = r₁ - ŷ₂

Step 3: Train Tree 3 on RESIDUALS r₂
        Predict: ŷ₃
        ...

Final: ŷ = ŷ₁ + ŷ₂ + ŷ₃ + ...
       (sum all predictions)

Key Idea: Each tree fixes mistakes of previous!
```

---

## 3.3 XGBoost: eXtreme Gradient Boosting

```
Features:
✅ Parallel tree construction (fast)
✅ Handle missing values
✅ Built-in regularization (prevent overfit)
✅ Flexible loss functions
✅ Pruning (remove weak branches)

Why popular?
- Extremely accurate
- Fast training
- Low memory
- Wins many Kaggle competitions!
```

---

## 3.4 Hyperparameters

```
n_rounds/n_estimators: Number of boosting rounds
├─ Start: 100
└─ Tune: 50-500 common

learning_rate (eta): Step size
├─ 0.1 (default): Conservative
├─ 0.3: Moderate
├─ High values: May overshoot
└─ Lower η needs more rounds

max_depth: Tree depth
├─ 3-5: Shallow trees (weak learners)
├─ Default: 6
└─ Higher: Complex interactions

gamma: Min loss to split
├─ 0 (default): Split if any loss reduction
└─ Higher: More conservative splits

colsample_bytree: Feature sampling fraction
├─ 1.0 (default): Use all features
└─ 0.5-0.8: Sample subset
```

---

# 🟣 Part 4: เปรียบเทียบทั้ง 6 Algorithms

## 4.1 Complete Comparison

| Feature | kNN | NB | LDA | QDA | SVM | RF | XGB |
|---------|-----|-----|-----|-----|-----|-----|-----|
| **Speed Train** | None | Fast | Fast | Medium | Slow | Medium | Medium |
| **Speed Predict** | Slow | Fast | Fast | Fast | Medium | Fast | Fast |
| **Data Size** | Small | Any | Medium | Large | Any | Large | Large |
| **Non-linear** | ⭐⭐⭐ | ⭐⭐ | ⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| **Interpretable** | ⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐ | ⭐ | ⭐⭐ | ⭐ |
| **Hyperparams** | 1 | 0 | 0 | 0 | 3-4 | 3-5 | 5+ |
| **Overfitting** | High | Low | Low | Medium | Medium | Low | Low |
| **Bias-Var** | Low Bias | Medium | High Bias | Medium | Medium | Low Bias | Medium |

---

## 4.2 Decision Tree: Which Algorithm?

```
START
  │
  ├─ Linear separable?
  │  ├─ YES → LDA (fastest)
  │  └─ NO → Q2
  │
  ├─ Text/Categorical data?
  │  ├─ YES → Naive Bayes
  │  └─ NO → Q3
  │
  ├─ Need interpretability?
  │  ├─ YES → LDA or Naive Bayes
  │  └─ NO → Q4
  │
  ├─ Large dataset (>10K)?
  │  ├─ YES → Random Forest or XGBoost
  │  └─ NO → Q5
  │
  ├─ Accuracy is CRITICAL?
  │  ├─ YES → XGBoost (best accuracy!)
  │  └─ NO → Q6
  │
  └─ Speed is critical?
     ├─ YES → Naive Bayes or LDA
     └─ NO → Try XGBoost
```

---

## 4.3 Quick Selection Guide

### Small Dataset (< 1000)
```
Best: Naive Bayes > LDA > kNN
Why: Fewer parameters to estimate
```

### Medium Dataset (1K - 10K)
```
Best: LDA > QDA > Random Forest
Why: Balanced complexity
```

### Large Dataset (> 10K)
```
Best: XGBoost > Random Forest > SVM
Why: Handle complexity, good accuracy
```

### Non-linear Pattern
```
Best: XGBoost > SVM > Random Forest
Why: Can learn complex boundaries
```

### Need Probability Scores
```
Best: Naive Bayes > LDA > SVM (Platt scaling)
Why: Directly give probabilities
```

### Need Speed
```
Best: Naive Bayes > LDA > kNN > Random Forest > XGBoost > SVM
Why: Computational efficiency
```

### Text Classification
```
Best: Naive Bayes ⭐⭐⭐
Why: Handles high-dimensional categorical perfectly
```

---

# 🔵 Part 5: R CODE EXAMPLES

## 5.1 Support Vector Machine (SVM)

```r
library(caret)
library(kernlab)

# Load & Split
data(iris)
idx <- createDataPartition(iris$Species, p = 0.8, list = FALSE)
train <- iris[idx, ]
test <- iris[-idx, ]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TRAIN SVM with different kernels
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ctrl <- trainControl(method = "cv", number = 5)

# Linear SVM
svm_linear <- train(
  Species ~ .,
  data = train,
  method = "svmLinear",
  trControl = ctrl,
  tuneGrid = expand.grid(C = c(0.1, 1, 10))
)

# RBF (Radial Basis Function) SVM
svm_rbf <- train(
  Species ~ .,
  data = train,
  method = "svmRadial",
  trControl = ctrl,
  tuneGrid = expand.grid(
    C = c(0.1, 1, 10),
    sigma = c(0.01, 0.1, 1)
  )
)

# Polynomial SVM
svm_poly <- train(
  Species ~ .,
  data = train,
  method = "svmPoly",
  trControl = ctrl,
  tuneGrid = expand.grid(
    C = c(0.1, 1, 10),
    degree = c(2, 3),
    scale = 1
  )
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PREDICT & EVALUATE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

pred_linear <- predict(svm_linear, test)
pred_rbf <- predict(svm_rbf, test)
pred_poly <- predict(svm_poly, test)

confusionMatrix(pred_linear, test$Species)
confusionMatrix(pred_rbf, test$Species)
confusionMatrix(pred_poly, test$Species)
```

---

## 5.2 Random Forest

```r
library(randomForest)
library(caret)

# Train Random Forest
rf_model <- train(
  Species ~ .,
  data = train,
  method = "rf",
  trControl = trainControl(method = "cv", number = 5),
  tuneGrid = expand.grid(mtry = c(2, 3, 4))
  # mtry = number of features to sample at each split
)

# Or using randomForest directly
rf_direct <- randomForest(
  Species ~ .,
  data = train,
  ntree = 500,      # Number of trees
  mtry = 2,         # Features per split
  max.depth = 10,   # Max tree depth
  importance = TRUE # Calculate feature importance
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FEATURE IMPORTANCE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

importance(rf_direct)
# Shows which features matter most!

plot(importance(rf_direct))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PREDICT & EVALUATE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

pred_rf <- predict(rf_model, test)
confusionMatrix(pred_rf, test$Species)
```

---

## 5.3 XGBoost

```r
library(xgboost)
library(caret)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# METHOD 1: Using caret
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

xgb_model <- train(
  Species ~ .,
  data = train,
  method = "xgbTree",
  trControl = trainControl(method = "cv", number = 5),
  tuneGrid = expand.grid(
    nrounds = c(50, 100, 200),
    max_depth = c(2, 3, 4),
    eta = c(0.1, 0.3),
    gamma = 0,
    colsample_bytree = 1,
    min_child_weight = 1,
    subsample = 1
  ),
  verbose = FALSE
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# METHOD 2: Using xgboost directly (for regression/advanced)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Convert data to xgb.DMatrix
train_matrix <- xgb.DMatrix(
  data = as.matrix(train[, -5]),
  label = as.numeric(train$Species) - 1  # 0-based indexing
)

test_matrix <- xgb.DMatrix(
  data = as.matrix(test[, -5]),
  label = as.numeric(test$Species) - 1
)

# Parameters
params <- list(
  booster = "gbtree",
  objective = "multi:softmax",  # Multi-class classification
  num_class = 3,                # Number of classes
  max_depth = 3,
  eta = 0.1,
  gamma = 0
)

# Train
xgb_direct <- xgb.train(
  params = params,
  data = train_matrix,
  nrounds = 100,
  eval_metric = "mlogloss"
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PREDICT & EVALUATE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

pred_xgb <- predict(xgb_model, test)
confusionMatrix(pred_xgb, test$Species)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FEATURE IMPORTANCE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

importance <- xgb.importance(
  feature_names = colnames(train[, -5]),
  model = xgb_direct
)
print(importance)

xgb.plot.importance(importance)
```

---

## 5.4 Complete Comparison: All 6 on Same Data

```r
library(tidyverse)
library(caret)
library(naivebayes)
library(MASS)
library(kernlab)
library(randomForest)
library(xgboost)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SETUP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

set.seed(42)
idx <- createDataPartition(iris$Species, p = 0.8, list = FALSE)
train <- iris[idx, ]
test <- iris[-idx, ]

ctrl <- trainControl(method = "cv", number = 5, verboseIter = FALSE)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TRAIN ALL 6
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

knn <- train(Species ~ ., data = train, method = "knn",
             tuneGrid = expand.grid(k = 5), trControl = ctrl)

nb <- train(Species ~ ., data = train, method = "naive_bayes", trControl = ctrl)

lda <- train(Species ~ ., data = train, method = "lda", trControl = ctrl)

qda <- train(Species ~ ., data = train, method = "qda", trControl = ctrl)

svm <- train(Species ~ ., data = train, method = "svmRadial",
             tuneGrid = expand.grid(C = 1, sigma = 0.1), trControl = ctrl)

rf <- train(Species ~ ., data = train, method = "rf",
            tuneGrid = expand.grid(mtry = 2), trControl = ctrl)

xgb <- train(Species ~ ., data = train, method = "xgbTree",
             tuneGrid = expand.grid(
               nrounds = 100, max_depth = 3, eta = 0.1,
               gamma = 0, colsample_bytree = 1,
               min_child_weight = 1, subsample = 1
             ), trControl = ctrl, verbose = FALSE)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PREDICT & COMPARE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

models <- list(knn, nb, lda, qda, svm, rf, xgb)
names_models <- c("kNN", "Naive Bayes", "LDA", "QDA", "SVM", "Random Forest", "XGBoost")

results <- tibble()

for (i in 1:length(models)) {
  pred <- predict(models[[i]], test)
  cm <- confusionMatrix(pred, test$Species)
  
  results <- results %>%
    bind_rows(tibble(
      Algorithm = names_models[i],
      Accuracy = cm$overall['Accuracy'],
      Kappa = cm$overall['Kappa']
    ))
}

cat("\n╔══════════════════════════════════════╗\n")
cat("║       6 ALGORITHMS COMPARISON        ║\n")
cat("╚══════════════════════════════════════╝\n")
print(results %>% arrange(desc(Accuracy)))

# Plot
ggplot(results, aes(x = reorder(Algorithm, Accuracy), y = Accuracy, fill = Algorithm)) +
  geom_col() +
  coord_flip() +
  ylim(0, 1) +
  geom_text(aes(label = round(Accuracy, 4)), hjust = -0.1) +
  labs(title = "6 ML Algorithms Comparison",
       x = "", y = "Test Accuracy") +
  theme_bw() +
  theme(legend.position = "none")
```

---

# 🎓 Learning Summary

## All 6 Algorithms at a Glance

```
┌──────────────┬─────────┬──────────┬──────────┐
│ Algorithm    │ Type    │ Speed    │ Accuracy │
├──────────────┼─────────┼──────────┼──────────┤
│ kNN          │ Distance│ Slow     │ Medium   │
│ Naive Bayes  │ Prob    │ Fast     │ Medium   │
│ LDA          │ Linear  │ Fast     │ Good     │
│ QDA          │ Quad    │ Medium   │ Good     │
│ SVM          │ Kernel  │ Slow     │ Excellent│
│ Random Forest│ Ensemble│ Medium   │ Excellent│
│ XGBoost      │ Boost   │ Medium   │ Best!    │
└──────────────┴─────────┴──────────┴──────────┘
```

---

## ✅ Workflow for choosing:

```
1. Load & Explore data
2. Preprocess (scale, handle NA)
3. Split train/test
4. TRY ALL 6 with same CV
5. Compare results
6. Pick winner
7. Fine-tune hyperparameters
8. Report final results
```

---

**ตอนนี้คุณรู้ 6 algorithms แล้ว! 🚀**
