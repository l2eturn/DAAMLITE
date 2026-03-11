# 📋 MASTER CHEAT SHEET: All 4 Classification Algorithms
## (kNN, Naive Bayes, LDA, QDA)

---

## 🎯 Quick Algorithm Selector

```
START HERE: "Which algorithm should I use?"

Q1: High-dimensional data (>100 features)?
├─ YES  → Naive Bayes ⭐
└─ NO   → Q2

Q2: Data is normally distributed?
├─ NO   → Naive Bayes ⭐
└─ YES  → Q3

Q3: Need probability scores?
├─ YES  → Naive Bayes ⭐
└─ NO   → Q4

Q4: Small dataset (<1000 rows)?
├─ YES  → kNN or Naive Bayes
└─ NO   → Q5

Q5: Classes have similar covariance?
├─ YES  → LDA (simple & fast)
└─ NO   → QDA (flexible)

Q6: Speed critical?
├─ YES  → Naive Bayes > LDA > kNN > QDA
└─ NO   → Use CV to select best

DEFAULT: Try all 4 with CV, pick winner!
```

---

## 📊 Comparison Table

| Feature | kNN | Naive Bayes | LDA | QDA |
|---------|-----|-------------|-----|-----|
| **Type** | Distance-based | Probabilistic | Linear | Quadratic |
| **Speed** | Slow | Fast | Fast | Medium |
| **Training** | Store data | Learn probs | Learn means | Learn covariances |
| **Prediction** | Distance → Vote | P(class\|data) | Linear combo | Quadratic combo |
| **Data Type** | Any | Mixed | Continuous | Continuous |
| **Scalability** | Poor (curse) | High | High | Medium |
| **Interpretability** | ⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐ |
| **Parameters** | k | Priors/Likelihoods | Means/Covariance | Covariances |
| **Assumptions** | None | Independence | Normal + Equal Cov | Normal |

---

## 🔴 kNN (k-Nearest Neighbors)

### How It Works
```
1. Calculate distance to all training data
2. Find k nearest neighbors
3. Majority vote determines class
```

### When to Use
```
✅ Small to medium datasets
✅ Non-linear patterns
✅ No assumptions about distribution
✅ Simple to understand
```

### When NOT to Use
```
❌ Large datasets (slow prediction)
❌ High-dimensional data (curse of dimensionality)
❌ Need probability scores
❌ Categorical variables only
```

### Key Parameters
```r
k = 5  # Usually best (odd number)
     # Try: 1, 3, 5, 7, 9, 11

distance = "euclidean"  # Default
                        # Others: manhattan, minkowski

scale = TRUE  # ALWAYS!
```

### Code
```r
library(caret)

# Train with CV
knn_model <- train(
  class ~ .,
  data = train,
  method = "knn",
  tuneGrid = expand.grid(k = seq(1, 21, 2)),
  trControl = trainControl(method = "cv", number = 5)
)

# Predict
predict(knn_model, test)
```

---

## 🟠 Naive Bayes

### How It Works
```
1. Calculate P(class) = class proportions
2. Calculate P(features|class) = feature likelihoods
3. Use Bayes' Theorem: P(class|features) ∝ P(features|class) × P(class)
4. Predict highest probability class

Assumption: Features are independent (often wrong, but works anyway!)
```

### When to Use
```
✅ High-dimensional data (text classification!)
✅ Categorical & continuous mix
✅ Need probability scores
✅ Small training dataset
✅ Fast prediction needed
✅ Can handle missing values
```

### When NOT to Use
```
❌ Strong feature dependencies (hurts performance)
❌ Need complex decision boundaries
❌ Few features where LDA better
```

### Code
```r
library(naivebayes)

# Train
nb_model <- train(
  class ~ .,
  data = train,
  method = "naive_bayes",
  trControl = trainControl(method = "cv", number = 5)
)

# Predict with probabilities
predict(nb_model, test, type = "prob")

# Use confidence for filtering
prob <- predict(nb_model, test, type = "prob")
max_prob <- apply(prob, 1, max)
confident <- which(max_prob > 0.8)
```

---

## 🟡 Linear Discriminant Analysis (LDA)

### How It Works
```
1. Find linear combinations of features (discriminant functions)
2. Maximize separation between class centroids
3. Minimize variance within each class
4. Project data onto these new axes
5. Classify based on projected position

Number of functions = min(classes-1, features)
Example: 1000 features + 3 classes → 2 discriminant functions!
```

### When to Use
```
✅ Normally distributed data
✅ Similar covariance between classes
✅ Need interpretable coefficients
✅ Fast prediction
✅ Dimension reduction bonus
✅ Medium-sized datasets
```

### When NOT to Use
```
❌ Non-normal data
❌ Different covariances (use QDA)
❌ Categorical variables (must be continuous)
❌ Non-linear patterns
```

### Code
```r
library(MASS)

# Train
lda_model <- train(
  class ~ .,
  data = train,
  method = "lda",
  trControl = trainControl(method = "cv", number = 5)
)

# View coefficients
lda_mass <- lda(class ~ ., data = train)
lda_mass$scaling  # Linear discriminant coefficients

# Predict
predict(lda_model, test)
```

---

## 🟣 Quadratic Discriminant Analysis (QDA)

### How It Works
```
Like LDA, but:
1. Allows different covariance matrices for each class
2. Results in curved decision boundaries
3. More flexible than LDA
4. More parameters to estimate (needs more data)
```

### When to Use
```
✅ Classes have different covariances
✅ Curved decision boundaries needed
✅ LDA underperforms significantly
✅ Enough data for more parameters
✅ Non-linear patterns
```

### When NOT to Use
```
❌ Small dataset (overfitting risk)
❌ Computational speed critical
❌ Need simple, interpretable model
❌ Equal covariances (LDA better)
```

### Code
```r
library(MASS)

# Train
qda_model <- train(
  class ~ .,
  data = train,
  method = "qda",
  trControl = trainControl(method = "cv", number = 5)
)

# Predict
predict(qda_model, test)
```

---

## ⚖️ Quick Comparison Examples

### Example 1: Tweet Classification (Politics/Sports/Movies)
```
Data: Text → words (categorical), high-dimensional
Features: 1000+ words
Classes: 4
Size: 10,000 tweets

BEST: Naive Bayes ⭐⭐⭐
Why:
- High-dimensional (curse of dimensionality for kNN)
- Categorical features (Naive Bayes handles)
- Fast training/prediction needed
- Text classification is perfect for NB

NOT GOOD:
- kNN: Too slow, curse of dimensionality
- LDA: Requires continuous features, can't handle categorical words
- QDA: Too many features, overfitting risk
```

### Example 2: Iris Flower Classification
```
Data: Sepal Length, Sepal Width, Petal Length, Petal Width
Features: 4 (continuous)
Classes: 3 (setosa, versicolor, virginica)
Size: 150 samples

BEST: LDA ⭐⭐
Also good: Naive Bayes (similar performance)

Why:
- Small dataset → ok for all
- Normal distribution ≈ met
- Continuous features ✓
- LDA simple and interpretable

NOT GOOD:
- QDA: Only 150 samples, too many parameters
- kNN: Works ok but k selection tricky
```

### Example 3: Diabetes Classification (Normal/Chemical/Overt)
```
Data: glucose, insulin, sspg levels
Features: 3 (continuous)
Classes: 3 (3 levels of diabetes)
Size: 145 samples

TRY ALL FOUR with CV!
```
- kNN: Good if non-linear
- Naive Bayes: Good fast baseline
- LDA: Good if assumptions met
- QDA: If LDA underperforms

Use CV to select → usually LDA or kNN wins
```

---

## 🔑 Key Concepts Summary

### Distance vs Probability vs Linear
```
kNN:
- Uses DISTANCE (Euclidean)
- Measures how close to other points

Naive Bayes:
- Uses PROBABILITY (Bayes' Theorem)
- P(class | data) = how likely is this class?

LDA:
- Uses LINEAR COMBINATION of features
- DF = a₁×var₁ + a₂×var₂ + ...
- Straight line separates classes

QDA:
- Uses QUADRATIC COMBINATION
- Curved lines separate classes
```

### Assumptions
```
kNN:
- None! Super flexible

Naive Bayes:
- Features are INDEPENDENT (often wrong, still works)

LDA:
- Data NORMALLY DISTRIBUTED in each class
- Same COVARIANCE for all classes

QDA:
- Data NORMALLY DISTRIBUTED in each class
- (Different covariance ok)
```

---

## 💻 Complete Workflow

```r
library(tidyverse)
library(caret)
library(MASS)
library(naivebayes)

# 1. Load & Explore
data <- read_csv("data.csv")
str(data)
summary(data)

# 2. Preprocess
data <- data %>%
  drop_na() %>%
  mutate(across(where(is.numeric), scale))  # Scale!

# 3. Split
set.seed(42)
idx <- createDataPartition(data$class, p = 0.8, list = FALSE)
train <- data[idx, ]
test <- data[-idx, ]

# 4. Define CV
ctrl <- trainControl(method = "cv", number = 5)

# 5. Train all 4
knn <- train(class ~ ., data = train, method = "knn",
             tuneGrid = expand.grid(k = seq(1, 21, 2)), trControl = ctrl)
nb <- train(class ~ ., data = train, method = "naive_bayes", trControl = ctrl)
lda <- train(class ~ ., data = train, method = "lda", trControl = ctrl)
qda <- train(class ~ ., data = train, method = "qda", trControl = ctrl)

# 6. Predict
pred_knn <- predict(knn, test)
pred_nb <- predict(nb, test)
pred_lda <- predict(lda, test)
pred_qda <- predict(qda, test)

# 7. Evaluate
cm_knn <- confusionMatrix(pred_knn, test$class)
cm_nb <- confusionMatrix(pred_nb, test$class)
cm_lda <- confusionMatrix(pred_lda, test$class)
cm_qda <- confusionMatrix(pred_qda, test$class)

# 8. Compare
tibble(
  Algorithm = c("kNN", "Naive Bayes", "LDA", "QDA"),
  Accuracy = c(cm_knn$overall[1], cm_nb$overall[1],
               cm_lda$overall[1], cm_qda$overall[1])
) %>%
  arrange(desc(Accuracy))
```

---

## 🚀 Pro Tips

1. **Always scale for kNN & LDA/QDA** (not needed for Naive Bayes)
2. **Try all 4 algorithms** with cross-validation
3. **Use stratified sampling** with imbalanced data
4. **Choose k (odd)** for kNN: 1, 3, 5, 7, ...
5. **Check assumptions** before using LDA/QDA
6. **Use probabilities** from Naive Bayes for confidence
7. **Look at coefficients** from LDA for interpretability
8. **Monitor CV results** not just test results
9. **Compare multiple metrics** (accuracy + kappa + F1)
10. **Understand why algorithm won** (don't just pick blindly!)

---

## ❌ Common Mistakes

| Mistake | ❌ Wrong | ✅ Correct |
|---------|---------|-----------|
| No scaling | `knn(..., X)` | `knn(..., scale(X))` |
| Tune on test set | `findBestK(test)` | `findBestK(train w/ CV)` |
| No train/test split | `train_on_all_data` | `80/20 split` |
| Using kNN for text | kNN with 1000 features | Naive Bayes |
| LDA with categorical | `lda(y ~ factor(x))` | Convert to numeric or use NB |
| Accuracy only | Use accuracy alone | Use kappa, F1, confusion matrix |
| Only 1 algorithm | Try kNN only | Try all 4 with CV |

---

## 📚 Learning Order

```
Day 1: Understand kNN
      └─ Distance, k selection, scaling importance

Day 2: Implement kNN
      └─ Code, cross-validation, confusion matrix

Day 3: Understand Naive Bayes
      └─ Bayes theorem, probability, independence

Day 4: Implement Naive Bayes
      └─ Code, probability scores, comparison with kNN

Day 5: Understand LDA
      └─ Discriminant functions, dimension reduction

Day 6: Implement LDA & QDA
      └─ Code both, compare coefficients

Day 7: Compare All 4
      └─ Run on multiple datasets, choose algorithm
```

---

## 🎯 Final Checklist

Before choosing algorithm:
- [ ] Understand your data (EDA)
- [ ] Check data types (continuous/categorical)
- [ ] Check distributions (normal?)
- [ ] Check for missing values
- [ ] Check for class imbalance
- [ ] Scale if needed (not for NB)
- [ ] Split train/test
- [ ] Set random seed
- [ ] Use cross-validation
- [ ] Try ALL 4 algorithms
- [ ] Compare metrics (not just accuracy)
- [ ] Understand why algorithm won

---

**ตอนนี้คุณพร้อมเลือกและใช้ 4 algorithms แล้ว! 🚀**
