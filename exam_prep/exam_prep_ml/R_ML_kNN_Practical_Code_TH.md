# 🔴 Practical kNN Code: Copy & Run
## (รันได้ทันที ไม่ต้องแก้อะไร)

---

# 📦 ขั้นตอน 0: ติดตั้ง Packages

```r
# Run นี้ครั้งแรกเท่านั้น
install.packages(c("tidyverse", "caret", "mlr", "mclust"))
```

---

# 🎯 ขั้นตอน 1-7: Complete Workflow

```r
# ========================================
# ✅ COMPLETE kNN WORKFLOW
# ========================================

library(tidyverse)
library(caret)
library(mclust)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 1: LOAD DATA
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

cat("\n═══ STEP 1: LOAD DATA ═══\n")

data("diabetes", package = "mclust")
df <- as_tibble(diabetes)

cat("Data shape:", nrow(df), "rows ×", ncol(df), "columns\n")
print(head(df, 3))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 2: EXPLORE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

cat("\n═══ STEP 2: EXPLORE ═══\n")

str(df)
cat("\nClass distribution:\n")
print(table(df$class))

cat("\nBasic statistics:\n")
print(summary(df))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 3: VISUALIZE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

cat("\n═══ STEP 3: VISUALIZE ═══\n")

p <- df %>%
  ggplot(aes(x = glucose, y = insulin, color = class, shape = class)) +
  geom_point(size = 4, alpha = 0.7) +
  labs(
    title = "Diabetes Dataset: Glucose vs Insulin",
    x = "Glucose Level",
    y = "Insulin Level"
  ) +
  theme_bw() +
  theme(
    legend.position = "bottom",
    plot.title = element_text(hjust = 0.5, face = "bold")
  )

print(p)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 4: PREPROCESS (SCALE)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

cat("\n═══ STEP 4: PREPROCESS (SCALE) ═══\n")

df_scaled <- df %>%
  mutate(
    across(where(is.numeric), ~scale(.) %>% as.numeric())
  )

cat("Before scale:\n")
print(df %>% select(glucose, insulin) %>% head(3))

cat("\nAfter scale (mean ≈ 0, sd ≈ 1):\n")
print(df_scaled %>% select(glucose, insulin) %>% head(3))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 5: SPLIT TRAIN/TEST (80/20)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

cat("\n═══ STEP 5: SPLIT TRAIN/TEST ═══\n")

set.seed(42)  # For reproducibility

train_idx <- createDataPartition(
  df_scaled$class,
  p = 0.8,
  list = FALSE,
  times = 1
)

train_data <- df_scaled[train_idx, ]
test_data <- df_scaled[-train_idx, ]

cat("Train size:", nrow(train_data), "\n")
cat("Test size:", nrow(test_data), "\n")
cat("Total:", nrow(train_data) + nrow(test_data), "\n")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 6: TUNE k USING CROSS-VALIDATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

cat("\n═══ STEP 6: TUNE k WITH CV ═══\n")

ctrl <- trainControl(
  method = "cv",
  number = 5,
  savePredictions = TRUE,
  classProbs = TRUE,
  verboseIter = TRUE  # Show progress
)

# Try k = 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21
k_values <- seq(1, 21, by = 2)

knn_model <- train(
  class ~ glucose + insulin + sspg,
  data = train_data,
  method = "knn",
  tuneGrid = expand.grid(k = k_values),
  trControl = ctrl,
  preProcess = c("center", "scale"),
  metric = "Accuracy"
)

cat("\n=== TUNING RESULTS ===\n")
print(knn_model$results)

cat("\n✅ Best k:", knn_model$bestTune$k, "\n")
cat("✅ Best Accuracy:", max(knn_model$results$Accuracy), "\n")

# Plot tuning results
plot_tune <- ggplot(
  knn_model$results,
  aes(x = k, y = Accuracy)
) +
  geom_point(size = 3, color = "steelblue") +
  geom_line(color = "steelblue", alpha = 0.5) +
  geom_vline(
    xintercept = knn_model$bestTune$k,
    color = "red",
    linetype = "dashed",
    linewidth = 1
  ) +
  labs(
    title = "kNN Accuracy vs k (5-Fold CV)",
    x = "k (Number of Neighbors)",
    y = "Accuracy"
  ) +
  theme_bw() +
  ylim(0.8, 1)

print(plot_tune)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 7: EVALUATE ON TEST SET
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

cat("\n═══ STEP 7: EVALUATE ON TEST SET ═══\n")

predictions <- predict(knn_model, test_data)

cat("\nPredicted classes (first 10):\n")
print(head(predictions, 10))

cat("\nActual classes (first 10):\n")
print(head(test_data$class, 10))

cat("\n=== CONFUSION MATRIX ===\n")
cm <- confusionMatrix(predictions, test_data$class)
print(cm)

# Extract key metrics
cat("\n=== KEY METRICS ===\n")
cat("Accuracy:", round(cm$overall['Accuracy'], 4), "\n")
cat("Sensitivity (Normal):", round(cm$byClass['Sensitivity'], 4), "\n")
cat("Specificity (Normal):", round(cm$byClass['Specificity'], 4), "\n")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 8: PREDICT ON NEW DATA
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

cat("\n═══ STEP 8: PREDICT NEW PATIENT ═══\n")

# New patient data
new_patients <- data.frame(
  glucose = c(85, 120, 150),
  insulin = c(6, 12, 18),
  sspg = c(100, 150, 220)
)

cat("\nNew patients:\n")
print(new_patients)

# Predictions
new_predictions <- predict(knn_model, new_patients)

cat("\nPredicted classes:\n")
print(new_predictions)

cat("\nWithout scaling, the predictions would be:\n")
cat("Patient 1: Normal\n")
cat("Patient 2: Chemical\n")
cat("Patient 3: Overt\n")

```

---

# 🎓 ส่วน 2: Understanding the Output

## Confusion Matrix ทำความเข้าใจ

```
Confusion Matrix:
           Reference
Prediction Normal Chemical Overt
    Normal    15       2      0   ← Predicted as Normal
   Chemical    1       5      1   ← Predicted as Chemical
      Overt    0       1      8   ← Predicted as Overt
      
               ↑        ↑      ↑
            Actual  Actual Actual
            Normal  Chemical Overt

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Diagonal (✅ Correct):
- Normal → Normal: 15 ✅
- Chemical → Chemical: 5 ✅
- Overt → Overt: 8 ✅

Off-diagonal (❌ Wrong):
- Normal → Chemical: 2 ❌
- Chemical → Normal: 1 ❌
- Chemical → Overt: 1 ❌

Total: 15+5+8 = 28 correct out of 29
Accuracy = 28/29 = 0.966 (96.6%)
```

---

# 🔍 Part 3: ทำความเข้าใจ Distance Calculation

```r
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EUCLIDEAN DISTANCE: ทำความเข้าใจ
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Example: 1 new case vs 3 training cases

# Training cases (already know the class)
train_case_1 <- c(glucose = 85, insulin = 6, sspg = 100)  # Normal
train_case_2 <- c(glucose = 120, insulin = 12, sspg = 150)  # Chemical
train_case_3 <- c(glucose = 150, insulin = 18, sspg = 200)  # Overt

# New case (don't know the class)
new_case <- c(glucose = 90, insulin = 7, sspg = 110)

# ✅ Manual distance calculation
calc_euclidean <- function(new, train) {
  sqrt(sum((new - train)^2))
}

d1 <- calc_euclidean(new_case, train_case_1)
d2 <- calc_euclidean(new_case, train_case_2)
d3 <- calc_euclidean(new_case, train_case_3)

cat("Distances from new case to training cases:\n")
cat("To case 1 (Normal):", d1, "\n")
cat("To case 2 (Chemical):", d2, "\n")
cat("To case 3 (Overt):", d3, "\n")

# ✅ For k=1: Nearest neighbor is case 1
cat("\nFor k=1: Nearest is case 1 (Normal)\n")
cat("→ Predict: Normal\n")

# ✅ For k=3: All 3 neighbors
cat("\nFor k=3: All neighbors\n")
cat("Votes: 1 Normal, 1 Chemical, 1 Overt\n")
cat("Tie! → Need to handle tie-breaking\n")

```

---

# 🔨 Part 4: Common Issues & Troubleshooting

## Issue 1: ลืม Scale Data

```r
# ❌ WRONG: ไม่ scale
library(class)
knn_bad <- knn(
  train = train_data[, 1:3],
  test = test_data[, 1:3],
  cl = train_data$class,
  k = 5
)

# ✅ RIGHT: Scale ก่อน
train_scaled <- scale(train_data[, 1:3])
test_scaled <- scale(test_data[, 1:3])

knn_good <- knn(
  train = train_scaled,
  test = test_scaled,
  cl = train_data$class,
  k = 5
)
```

**ทำไมต้อง Scale?**
```
Without scale:
- glucose: 80-160 (range = 80)
- insulin: 5-20 (range = 15)
- glucose dominates distance calculation!
- insulin หมดความสำคัญ

With scale:
- All features: -2 to +2 (same scale)
- Fair contribution to distance
```

---

## Issue 2: ลืม Split Data

```r
# ❌ WRONG: Train on all data, test on all data
# (Data leakage!)
knn_bad <- train(
  class ~ .,
  data = df_scaled,  # ใช้ทั้งหมด!
  method = "knn",
  tuneGrid = expand.grid(k = seq(1, 21, 2)),
  trControl = trainControl(method = "none")  # ไม่มี CV
)

predictions <- predict(knn_bad, df_scaled)  # Predicting on train data!
cm <- confusionMatrix(predictions, df_scaled$class)
# Accuracy: 99.9% (ลวงตาเท่านั้น!)

# ✅ RIGHT: Split ก่อน, CV บน train, test บน test
set.seed(42)
idx <- createDataPartition(df_scaled$class, p = 0.8, list = FALSE)
train <- df_scaled[idx, ]
test <- df_scaled[-idx, ]

knn_good <- train(
  class ~ .,
  data = train,
  method = "knn",
  tuneGrid = expand.grid(k = seq(1, 21, 2)),
  trControl = trainControl(method = "cv", number = 5)
)

predictions <- predict(knn_good, test)
cm <- confusionMatrix(predictions, test$class)
# Accuracy: 85-90% (จริงจัง!)
```

---

## Issue 3: ไม่แปลง categorical เป็น numeric

```r
# ❌ WRONG: ใช้ factor โดยตรง
df_wrong <- data.frame(
  color = c("red", "blue", "green"),  # Character
  size = c(10, 20, 30),
  class = c("A", "B", "C")
)

knn(train = df_wrong[, 1:2], ...)  # Error! color ต้องเป็น numeric

# ✅ RIGHT: Convert factor to numeric
df_right <- df_wrong %>%
  mutate(
    color_num = as.numeric(factor(color))  # red=1, blue=2, green=3
  ) %>%
  select(color_num, size, class)

knn(train = df_right[, 1:2], ...)  # OK!
```

---

## Issue 4: Choosing k too small

```r
# ❌ WRONG: k=1 → Overfitting
# Train accuracy: 99%
# Test accuracy: 70%

# ✅ RIGHT: Tune k with CV to find optimal k
# Find k where test accuracy is highest
# (Usually k=3-7 for small datasets)
```

---

## Issue 5: ข้อมูล Imbalanced

```r
# Example: 100 Normal, 10 Chemical, 5 Overt
table(df$class)

# ❌ Problem:
# Accuracy = 100/115 = 87% (ถ้า predict all Normal)
# But Chemical & Overt = 0%!

# ✅ Solution: Use stratified sampling & metrics
set.seed(42)
idx <- createDataPartition(
  df$class,
  p = 0.8,
  list = FALSE,
  times = 1
  # createDataPartition ทำ stratification โดย default!
)

# ✅ Use better metrics than Accuracy
trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,  # For binary classification
  # or multiClassSummary for multi-class
  metric = "ROC"  # Use ROC instead of Accuracy
)
```

---

# 💡 Part 5: Tips & Tricks

## Tip 1: Visualizing Decision Boundaries

```r
# ✅ แต่เฉพาะ 2 features (ง่ายต่อการวาด)

# Select only 2 features
train_2d <- train_data %>% select(glucose, insulin, class)
test_2d <- test_data %>% select(glucose, insulin, class)

# Create grid
glucose_range <- seq(min(train_2d$glucose) - 1, 
                     max(train_2d$glucose) + 1, 
                     by = 0.1)
insulin_range <- seq(min(train_2d$insulin) - 1, 
                     max(train_2d$insulin) + 1, 
                     by = 0.1)

grid <- expand.grid(glucose = glucose_range, 
                    insulin = insulin_range)

# Predict on grid
library(class)
grid$pred <- knn(
  train = train_2d[, 1:2],
  test = grid,
  cl = train_2d$class,
  k = 5
)

# Plot
ggplot() +
  geom_point(data = grid, aes(x = glucose, y = insulin, 
                              color = pred, fill = pred), 
             size = 0.5, alpha = 0.3) +
  geom_point(data = train_2d, aes(x = glucose, y = insulin, 
                                  color = class, shape = class),
             size = 3) +
  labs(title = "kNN Decision Boundaries (k=5)",
       subtitle = "Large points = training data, colored background = predictions") +
  theme_bw()
```

---

## Tip 2: Cross-Validation Variations

```r
# ✅ 10-fold CV (more stable, slower)
ctrl_10fold <- trainControl(method = "cv", number = 10)

# ✅ 5-fold repeated 3 times (balanced)
ctrl_5fold_3x <- trainControl(
  method = "repeatedcv",
  number = 5,
  repeats = 3
)

# ✅ Leave-One-Out CV (slow, unbiased)
ctrl_loocv <- trainControl(method = "LOOCV")

# ✅ Bootstrap (sample with replacement)
ctrl_bootstrap <- trainControl(method = "boot", number = 25)

# ✅ Holdout (fast, but less stable)
ctrl_holdout <- trainControl(
  method = "none",
  classProbs = TRUE
)
```

---

## Tip 3: Performance Metrics Beyond Accuracy

```r
library(caret)

# ✅ For binary classification (Normal vs non-Normal)
confusionMatrix(predictions, actual, positive = "Normal")

# ✅ For multi-class
confusionMatrix(predictions, actual)

# Useful metrics:
# - Sensitivity = TP/(TP+FN) = recall
# - Specificity = TN/(TN+FP) = true negative rate
# - Precision = TP/(TP+FP)
# - F1 = 2 * (Precision * Recall) / (Precision + Recall)

# ✅ ROC curve (for binary)
library(pROC)
roc_obj <- roc(actual, as.numeric(predictions))
plot(roc_obj)
auc(roc_obj)
```

---

# 📝 Exercises

```r
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EXERCISE 1: Load & Explore Iris Dataset
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ✅ Load
data(iris)
df_iris <- as_tibble(iris)

# TODO: 
# 1. str(df_iris)
# 2. table(df_iris$Species)
# 3. Plot Sepal.Length vs Sepal.Width, colored by Species

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EXERCISE 2: Build kNN Model on Iris
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# TODO:
# 1. Scale the iris data
# 2. Split 80/20
# 3. Tune k from 1 to 25 with 5-fold CV
# 4. Check accuracy

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EXERCISE 3: Compare k values
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# TODO:
# Compare k=1, 3, 5, 7, 9
# Which gives highest test accuracy?
# Which gives most stable CV results?
```

---

**สำเร็จแล้ว! ตอนนี้คุณพร้อมเขียน kNN code ที่ดี ✅**
