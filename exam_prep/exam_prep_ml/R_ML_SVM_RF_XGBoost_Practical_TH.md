# 💻 Practical Code: SVM, Random Forest & XGBoost
## (Ready to Run - Complete Workflows)

---

# 📦 Install Packages

```r
install.packages(c("kernlab", "randomForest", "xgboost", "tidyverse", "caret"))
```

---

# 🎯 COMPLETE IRIS ANALYSIS: All 6 Algorithms

```r
# ════════════════════════════════════════════════════════════════
# ✅ COMPREHENSIVE ML COMPARISON: All 6 Algorithms
# ════════════════════════════════════════════════════════════════

library(tidyverse)
library(caret)
library(naivebayes)
library(MASS)
library(kernlab)
library(randomForest)
library(xgboost)

cat("\n════════════════════════════════════════════════════════════\n")
cat("   ULTIMATE ML COMPARISON: 6 ALGORITHMS ON IRIS DATASET\n")
cat("════════════════════════════════════════════════════════════\n")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 1: LOAD & EXPLORE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

cat("\n✅ STEP 1: LOAD & EXPLORE\n")

data(iris)
df <- as_tibble(iris)

cat("Dataset: Iris (150 samples, 5 variables)\n")
cat("Classes:", nlevels(df$Species), "-", 
    paste(levels(df$Species), collapse = ", "), "\n")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 2: SPLIT DATA
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

cat("\n✅ STEP 2: SPLIT DATA (80/20)\n")

set.seed(42)
idx <- createDataPartition(df$Species, p = 0.8, list = FALSE)
train <- df[idx, ]
test <- df[-idx, ]

cat("Train:", nrow(train), "| Test:", nrow(test), "\n")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 3: SETUP CROSS-VALIDATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

cat("\n✅ STEP 3: SETUP 5-FOLD CROSS-VALIDATION\n")

ctrl <- trainControl(
  method = "cv",
  number = 5,
  savePredictions = FALSE,
  classProbs = FALSE,
  verboseIter = FALSE
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 4: TRAIN MODEL 1 - KNN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

cat("\n✅ STEP 4: TRAIN kNN (k=5)\n")

knn_model <- train(
  Species ~ .,
  data = train,
  method = "knn",
  tuneGrid = expand.grid(k = 5),
  trControl = ctrl
)

cat("kNN CV Accuracy:", round(max(knn_model$results$Accuracy), 4), "\n")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 5: TRAIN MODEL 2 - NAIVE BAYES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

cat("\n✅ STEP 5: TRAIN NAIVE BAYES\n")

nb_model <- train(
  Species ~ .,
  data = train,
  method = "naive_bayes",
  trControl = ctrl
)

cat("Naive Bayes CV Accuracy:", round(max(nb_model$results$Accuracy), 4), "\n")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 6: TRAIN MODEL 3 - LDA
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

cat("\n✅ STEP 6: TRAIN LDA\n")

lda_model <- train(
  Species ~ .,
  data = train,
  method = "lda",
  trControl = ctrl
)

cat("LDA CV Accuracy:", round(max(lda_model$results$Accuracy), 4), "\n")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 7: TRAIN MODEL 4 - QDA
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

cat("\n✅ STEP 7: TRAIN QDA\n")

qda_model <- train(
  Species ~ .,
  data = train,
  method = "qda",
  trControl = ctrl
)

cat("QDA CV Accuracy:", round(max(qda_model$results$Accuracy), 4), "\n")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 8: TRAIN MODEL 5 - SVM (RBF Kernel)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

cat("\n✅ STEP 8: TRAIN SVM (RBF Kernel)\n")

svm_model <- train(
  Species ~ .,
  data = train,
  method = "svmRadial",
  trControl = ctrl,
  tuneGrid = expand.grid(C = 1, sigma = 0.1)
  # C = cost, sigma = kernel parameter
)

cat("SVM CV Accuracy:", round(max(svm_model$results$Accuracy), 4), "\n")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 9: TRAIN MODEL 6 - RANDOM FOREST
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

cat("\n✅ STEP 9: TRAIN RANDOM FOREST\n")

rf_model <- train(
  Species ~ .,
  data = train,
  method = "rf",
  trControl = ctrl,
  tuneGrid = expand.grid(mtry = c(2, 3, 4)),
  ntree = 200
)

cat("Random Forest CV Accuracy:", round(max(rf_model$results$Accuracy), 4), "\n")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 10: TRAIN MODEL 7 - XGBOOST
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

cat("\n✅ STEP 10: TRAIN XGBOOST\n")

xgb_model <- train(
  Species ~ .,
  data = train,
  method = "xgbTree",
  trControl = ctrl,
  tuneGrid = expand.grid(
    nrounds = c(50, 100),
    max_depth = 3,
    eta = 0.1,
    gamma = 0,
    colsample_bytree = 1,
    min_child_weight = 1,
    subsample = 1
  ),
  verbose = FALSE
)

cat("XGBoost CV Accuracy:", round(max(xgb_model$results$Accuracy), 4), "\n")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 11: PREDICT ON TEST SET
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

cat("\n✅ STEP 11: PREDICT ON TEST SET\n")

predictions <- list(
  knn = predict(knn_model, test),
  nb = predict(nb_model, test),
  lda = predict(lda_model, test),
  qda = predict(qda_model, test),
  svm = predict(svm_model, test),
  rf = predict(rf_model, test),
  xgb = predict(xgb_model, test)
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 12: EVALUATE & COMPARE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

cat("\n✅ STEP 12: EVALUATE & COMPARE\n")

results <- tibble()

for (i in 1:length(predictions)) {
  cm <- confusionMatrix(predictions[[i]], test$Species)
  
  results <- results %>%
    bind_rows(tibble(
      Algorithm = names(predictions)[i],
      Accuracy = cm$overall['Accuracy'],
      Kappa = cm$overall['Kappa'],
      Sensitivity = mean(cm$byClass[, 1]),
      Specificity = mean(cm$byClass[, 2])
    ))
}

cat("\n┌────────────────────────────────────────────────────────────────┐\n")
cat("│        TEST SET RESULTS (FINAL COMPARISON)                   │\n")
cat("├────────────────────────────────────────────────────────────────┤\n")

results_sorted <- results %>% arrange(desc(Accuracy))
print(results_sorted)

cat("└────────────────────────────────────────────────────────────────┘\n")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 13: VISUALIZE RESULTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

cat("\n✅ STEP 13: VISUALIZE RESULTS\n")

p1 <- ggplot(results_sorted, aes(x = reorder(Algorithm, Accuracy), 
                                 y = Accuracy, fill = Algorithm)) +
  geom_col(alpha = 0.7) +
  geom_text(aes(label = round(Accuracy, 4)), vjust = -0.3, size = 3) +
  coord_flip() +
  ylim(0, 1.1) +
  labs(title = "Test Accuracy Comparison",
       x = "", y = "Accuracy") +
  theme_bw() +
  theme(legend.position = "none")

p2 <- ggplot(results_sorted, aes(x = reorder(Algorithm, Kappa), 
                                 y = Kappa, fill = Algorithm)) +
  geom_col(alpha = 0.7) +
  geom_text(aes(label = round(Kappa, 4)), vjust = -0.3, size = 3) +
  coord_flip() +
  ylim(0, 1.1) +
  labs(title = "Kappa Comparison",
       x = "", y = "Kappa") +
  theme_bw() +
  theme(legend.position = "none")

print(p1)
print(p2)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 14: DETAILED RESULTS FOR WINNER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

cat("\n✅ STEP 14: DETAILED RESULTS FOR BEST MODEL\n")

best_algo <- results_sorted$Algorithm[1]
best_pred <- predictions[[best_algo]]

cat("\n🏆 BEST ALGORITHM:", best_algo, "\n")
cat("Accuracy:", round(results_sorted$Accuracy[1], 4), "\n\n")

cat("Confusion Matrix:\n")
cm_best <- confusionMatrix(best_pred, test$Species)
print(cm_best)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SUMMARY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

cat("\n════════════════════════════════════════════════════════════\n")
cat("RANKING (by Test Accuracy):\n")
cat("════════════════════════════════════════════════════════════\n")

for (i in 1:nrow(results_sorted)) {
  cat(sprintf("%d. %-15s Accuracy: %.4f\n",
              i, results_sorted$Algorithm[i], results_sorted$Accuracy[i]))
}

cat("\n════════════════════════════════════════════════════════════\n")
```

---

## 🔬 Random Forest Feature Importance

```r
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ANALYZE RANDOM FOREST: FEATURE IMPORTANCE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

cat("\n═══════════════════════════════════════════════════════════\n")
cat("  RANDOM FOREST: FEATURE IMPORTANCE\n")
cat("═══════════════════════════════════════════════════════════\n")

# Train RF with importance calculation
rf_importance <- randomForest(
  Species ~ .,
  data = train,
  ntree = 500,
  importance = TRUE,
  mtry = 2
)

# Get importance scores
imp <- importance(rf_importance)
print(imp)

# Visualize
varImpPlot(rf_importance, 
           main = "Random Forest: Feature Importance",
           type = 1)

cat("\nInterpretation:\n")
cat("- Mean Decrease in Accuracy: How much accuracy drops if feature removed\n")
cat("- Higher value = More important feature\n")

# Get top features
top_features <- names(sort(imp[, 1], decreasing = TRUE)[1:3])
cat("\nTop 3 important features:", paste(top_features, collapse = ", "), "\n")
```

---

## 🚀 XGBoost Hyperparameter Tuning

```r
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# XGBOOST: HYPERPARAMETER TUNING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

cat("\n═══════════════════════════════════════════════════════════\n")
cat("  XGBOOST: HYPERPARAMETER TUNING\n")
cat("═══════════════════════════════════════════════════════════\n")

# Tune with different parameters
xgb_tuned <- train(
  Species ~ .,
  data = train,
  method = "xgbTree",
  trControl = trainControl(method = "cv", number = 5, verboseIter = TRUE),
  tuneGrid = expand.grid(
    nrounds = seq(50, 300, 50),      # 50, 100, 150, ...
    max_depth = c(2, 3, 4, 5),       # Tree depth
    eta = c(0.01, 0.05, 0.1, 0.3),   # Learning rate
    gamma = c(0, 0.5, 1),            # Min loss to split
    colsample_bytree = 1,
    min_child_weight = 1,
    subsample = 1
  ),
  verbose = FALSE
)

# Best parameters
cat("\nBest Parameters Found:\n")
print(xgb_tuned$bestTune)

# Plot tuning results
plot(xgb_tuned)

# Repredict with best model
pred_xgb_tuned <- predict(xgb_tuned, test)
cm_xgb_tuned <- confusionMatrix(pred_xgb_tuned, test$Species)
cat("\nXGBoost (Tuned) Accuracy:", round(cm_xgb_tuned$overall['Accuracy'], 4), "\n")
```

---

## ⚡ SVM: Try All Kernels

```r
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SVM: COMPARE DIFFERENT KERNELS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

cat("\n═══════════════════════════════════════════════════════════\n")
cat("  SVM: COMPARING KERNELS\n")
cat("═══════════════════════════════════════════════════════════\n")

ctrl_svm <- trainControl(method = "cv", number = 5)

# Linear SVM
svm_linear <- train(
  Species ~ ., data = train, method = "svmLinear",
  tuneGrid = expand.grid(C = c(0.1, 1, 10)), trControl = ctrl_svm
)

# Polynomial SVM
svm_poly <- train(
  Species ~ ., data = train, method = "svmPoly",
  tuneGrid = expand.grid(
    C = c(0.1, 1, 10),
    degree = c(2, 3),
    scale = 1
  ), trControl = ctrl_svm
)

# RBF SVM
svm_rbf <- train(
  Species ~ ., data = train, method = "svmRadial",
  tuneGrid = expand.grid(
    C = c(0.1, 1, 10),
    sigma = c(0.01, 0.1, 1)
  ), trControl = ctrl_svm
)

# Compare
svm_results <- tibble(
  Kernel = c("Linear", "Polynomial", "RBF"),
  CV_Accuracy = c(
    max(svm_linear$results$Accuracy),
    max(svm_poly$results$Accuracy),
    max(svm_rbf$results$Accuracy)
  ),
  Best_C = c(
    svm_linear$bestTune$C,
    svm_poly$bestTune$C,
    svm_rbf$bestTune$C
  )
)

cat("\nSVM Kernel Comparison:\n")
print(svm_results)

# Test predictions
pred_linear <- predict(svm_linear, test)
pred_poly <- predict(svm_poly, test)
pred_rbf <- predict(svm_rbf, test)

cat("\nTest Accuracy by Kernel:\n")
cat("Linear:    ", round(sum(pred_linear == test$Species) / nrow(test), 4), "\n")
cat("Polynomial:", round(sum(pred_poly == test$Species) / nrow(test), 4), "\n")
cat("RBF:       ", round(sum(pred_rbf == test$Species) / nrow(test), 4), "\n")
```

---

**ตอนนี้คุณมี practical code สำหรับทั้ง 6 algorithms! 🚀**
