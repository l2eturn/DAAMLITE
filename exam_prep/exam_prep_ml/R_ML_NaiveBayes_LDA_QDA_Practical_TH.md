# 💻 Practical Code: Naive Bayes, LDA & QDA
## (Copy & Run - No Modifications Needed)

---

# 📦 INSTALL PACKAGES (First time only)

```r
install.packages(c("tidyverse", "caret", "mlbench", "naivebayes", "MASS"))
```

---

# 🎯 COMPLETE WORKFLOW: All 3 Algorithms on Same Data

```r
# ════════════════════════════════════════════════════════════════
# ✅ COMPLETE IRIS ANALYSIS: Naive Bayes vs LDA vs QDA
# ════════════════════════════════════════════════════════════════

library(tidyverse)
library(caret)
library(naivebayes)
library(MASS)

cat("═══════════════════════════════════════════════════════════\n")
cat("  ML COMPARISON: Naive Bayes vs LDA vs QDA\n")
cat("═══════════════════════════════════════════════════════════\n")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 1: LOAD & EXPLORE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

cat("\n✅ STEP 1: LOAD & EXPLORE\n")

data(iris)
df <- as_tibble(iris)

cat("\nData shape:", nrow(df), "×", ncol(df), "\n")
cat("\nClass distribution:\n")
print(table(df$Species))

# Visualize
p <- ggplot(df, aes(x = Sepal.Length, y = Petal.Length, 
                     color = Species, shape = Species)) +
  geom_point(size = 3, alpha = 0.7) +
  labs(title = "Iris Dataset: Features vs Species") +
  theme_bw() +
  theme(legend.position = "bottom")
print(p)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 2: SPLIT DATA
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

cat("\n✅ STEP 2: SPLIT DATA\n")

set.seed(42)
idx <- createDataPartition(df$Species, p = 0.8, list = FALSE)
train <- df[idx, ]
test <- df[-idx, ]

cat("Train size:", nrow(train), "\n")
cat("Test size:", nrow(test), "\n")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 3: DEFINE TRAINING CONTROL
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

cat("\n✅ STEP 3: SETUP CROSS-VALIDATION\n")

ctrl <- trainControl(
  method = "cv",
  number = 5,
  savePredictions = TRUE,
  classProbs = TRUE,
  verboseIter = FALSE
)

cat("Using 5-fold Cross-Validation\n")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 4: TRAIN NAIVE BAYES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

cat("\n✅ STEP 4: TRAIN NAIVE BAYES\n")

nb_model <- train(
  Species ~ .,
  data = train,
  method = "naive_bayes",
  trControl = ctrl,
  metric = "Accuracy"
)

cat("\nNaive Bayes CV Accuracy:", 
    round(max(nb_model$results$Accuracy), 4), "\n")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 5: TRAIN LDA
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

cat("\n✅ STEP 5: TRAIN LDA\n")

lda_model <- train(
  Species ~ .,
  data = train,
  method = "lda",
  trControl = ctrl,
  metric = "Accuracy"
)

cat("LDA CV Accuracy:", 
    round(max(lda_model$results$Accuracy), 4), "\n")

# Also train with MASS to get coefficients
lda_mass <- lda(Species ~ ., data = train)
cat("\nLDA Coefficients:\n")
print(lda_mass$scaling)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 6: TRAIN QDA
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

cat("\n✅ STEP 6: TRAIN QDA\n")

qda_model <- train(
  Species ~ .,
  data = train,
  method = "qda",
  trControl = ctrl,
  metric = "Accuracy"
)

cat("QDA CV Accuracy:", 
    round(max(qda_model$results$Accuracy), 4), "\n")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 7: PREDICT ON TEST SET
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

cat("\n✅ STEP 7: PREDICT ON TEST SET\n")

pred_nb <- predict(nb_model, test)
pred_lda <- predict(lda_model, test)
pred_qda <- predict(qda_model, test)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 8: EVALUATE & COMPARE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

cat("\n✅ STEP 8: EVALUATE & COMPARE\n")

cm_nb <- confusionMatrix(pred_nb, test$Species)
cm_lda <- confusionMatrix(pred_lda, test$Species)
cm_qda <- confusionMatrix(pred_qda, test$Species)

cat("\n┌─────────────────────────────────────────┐\n")
cat("│         TEST SET RESULTS                │\n")
cat("├─────────────────────────────────────────┤\n")
cat("│ Algorithm         │ Accuracy │ Kappa   │\n")
cat("├───────────────────┼──────────┼─────────┤\n")
cat(sprintf("│ Naive Bayes       │ %0.4f    │ %0.4f   │\n",
            cm_nb$overall['Accuracy'], cm_nb$overall['Kappa']))
cat(sprintf("│ LDA               │ %0.4f    │ %0.4f   │\n",
            cm_lda$overall['Accuracy'], cm_lda$overall['Kappa']))
cat(sprintf("│ QDA               │ %0.4f    │ %0.4f   │\n",
            cm_qda$overall['Accuracy'], cm_qda$overall['Kappa']))
cat("└─────────────────────────────────────────┘\n")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 9: DETAILED CONFUSION MATRICES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

cat("\n✅ STEP 9: DETAILED RESULTS\n")

cat("\n🔵 NAIVE BAYES\n")
print(cm_nb)

cat("\n🔴 LDA\n")
print(cm_lda)

cat("\n🟡 QDA\n")
print(cm_qda)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 10: VISUALIZE RESULTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

cat("\n✅ STEP 10: VISUALIZE COMPARISON\n")

comparison <- tibble(
  Algorithm = c("Naive Bayes", "LDA", "QDA"),
  Accuracy = c(
    cm_nb$overall['Accuracy'],
    cm_lda$overall['Accuracy'],
    cm_qda$overall['Accuracy']
  ),
  Kappa = c(
    cm_nb$overall['Kappa'],
    cm_lda$overall['Kappa'],
    cm_qda$overall['Kappa']
  )
)

p_acc <- ggplot(comparison, aes(x = Algorithm, y = Accuracy, 
                                fill = Algorithm)) +
  geom_col(alpha = 0.7) +
  ylim(0, 1) +
  geom_text(aes(label = round(Accuracy, 4)), 
            vjust = -0.3, size = 4) +
  labs(title = "Algorithm Accuracy Comparison") +
  theme_bw() +
  theme(legend.position = "none")

p_kappa <- ggplot(comparison, aes(x = Algorithm, y = Kappa, 
                                   fill = Algorithm)) +
  geom_col(alpha = 0.7) +
  ylim(0, 1) +
  geom_text(aes(label = round(Kappa, 4)), 
            vjust = -0.3, size = 4) +
  labs(title = "Algorithm Kappa Comparison") +
  theme_bw() +
  theme(legend.position = "none")

print(p_acc)
print(p_kappa)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SUMMARY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

cat("\n═══════════════════════════════════════════════════════════\n")
cat("  SUMMARY\n")
cat("═══════════════════════════════════════════════════════════\n")

best <- comparison %>%
  arrange(desc(Accuracy)) %>%
  slice(1)

cat(sprintf("\n✅ BEST ALGORITHM: %s (Accuracy = %.4f)\n",
            best$Algorithm, best$Accuracy))

cat("\n📊 Ranking by Accuracy:\n")
comparison %>%
  arrange(desc(Accuracy)) %>%
  print()

cat("\n════════════════════════════════════════════════════════════\n")
```

---

# 🎓 Part 2: Understanding Naive Bayes Probabilities

```r
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SEE PROBABILITIES FROM NAIVE BAYES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

cat("\n═══════════════════════════════════════════════════════════\n")
cat("  NAIVE BAYES PROBABILITIES\n")
cat("═══════════════════════════════════════════════════════════\n")

# Get probability estimates
prob_nb <- predict(nb_model, test, type = "prob")

cat("\nProbabilities for first 10 test cases:\n")
print(head(prob_nb, 10))

# Example interpretation
cat("\nFirst test case:\n")
cat("Actual Species:", as.character(test$Species[1]), "\n")
cat("Predicted Species:", as.character(pred_nb[1]), "\n")
cat(sprintf("  P(setosa)     = %.4f\n", prob_nb[1, 1]))
cat(sprintf("  P(versicolor) = %.4f\n", prob_nb[1, 2]))
cat(sprintf("  P(virginica)  = %.4f\n", prob_nb[1, 3]))
cat("  → Highest probability = Predicted class ✓\n")

# Distribution of max probabilities
max_probs <- apply(prob_nb, 1, max)

cat("\nDistribution of prediction confidence:\n")
cat(sprintf("  Min confidence:  %.4f\n", min(max_probs)))
cat(sprintf("  Max confidence:  %.4f\n", max(max_probs)))
cat(sprintf("  Mean confidence: %.4f\n", mean(max_probs)))

# Histogram
hist(max_probs, breaks = 20, 
     main = "Naive Bayes Prediction Confidence",
     xlab = "Max Probability",
     ylab = "Frequency",
     col = "steelblue", alpha = 0.7)

cat("\n✅ Interpretation:\n")
cat("- Higher confidence (closer to 1) = More confident prediction\n")
cat("- Lower confidence (closer to 1/3) = Less confident prediction\n")
cat("- Can use confidence as threshold: only trust predictions > 0.8?\n")
```

---

# 🔬 Part 3: Understanding LDA Coefficients

```r
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ANALYZE LDA COEFFICIENTS & DISCRIMINANT FUNCTIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

cat("\n═══════════════════════════════════════════════════════════\n")
cat("  LDA DISCRIMINANT FUNCTION ANALYSIS\n")
cat("═══════════════════════════════════════════════════════════\n")

# Train LDA with MASS
lda_mass <- lda(Species ~ ., data = train)

cat("\nLDA Scaling (Canonical Coefficients):\n")
scaling <- lda_mass$scaling
print(scaling)

cat("\n📌 Interpretation:\n")
cat("- Each column (LD1, LD2) is a discriminant function\n")
cat("- Each row is a variable's weight in that function\n")
cat("- Larger absolute values = more important for separation\n")

cat("\nVariable Importance for LD1:\n")
ld1_importance <- abs(scaling[, 1])
ld1_importance <- ld1_importance[order(ld1_importance, decreasing = TRUE)]
print(ld1_importance)

cat("\nVariable Importance for LD2:\n")
ld2_importance <- abs(scaling[, 2])
ld2_importance <- ld2_importance[order(ld2_importance, decreasing = TRUE)]
print(ld2_importance)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PROJECT DATA ONTO DISCRIMINANT AXES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

cat("\n📊 Projecting data onto discriminant axes...\n")

# Get discriminant scores for train and test
train_scaled <- as.matrix(scale(train[, -5]))  # Standardize
test_scaled <- as.matrix(scale(test[, -5]))

# Project onto discriminant axes
train_ld <- train_scaled %*% scaling
test_ld <- test_scaled %*% scaling

# Combine with class labels
train_df <- as_tibble(train_ld) %>%
  mutate(Species = train$Species)

test_df <- as_tibble(test_ld) %>%
  mutate(Species = test$Species)

# Visualize
p_lda <- ggplot(test_df, aes(x = LD1, y = LD2, 
                             color = Species, shape = Species)) +
  geom_point(size = 3, alpha = 0.7) +
  labs(title = "LDA Projection: Test Set",
       x = "Linear Discriminant 1",
       y = "Linear Discriminant 2") +
  theme_bw() +
  theme(legend.position = "bottom")

print(p_lda)

cat("\n✅ Key insight:\n")
cat("- LD1 separates setosa from others (large separation)\n")
cat("- LD2 separates versicolor from virginica\n")
cat("- Clear clusters = good separation!\n")
```

---

# ⚖️ Part 4: Naive Bayes vs LDA vs QDA Decision Guide

```r
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DECISION GUIDE: WHICH ALGORITHM TO USE?
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

cat("\n═══════════════════════════════════════════════════════════\n")
cat("  DECISION GUIDE: WHICH ALGORITHM?\n")
cat("═══════════════════════════════════════════════════════════\n")

cat("\n1️⃣  HIGH-DIMENSIONAL DATA (>100 features)?")
cat("\n   → Choose NAIVE BAYES")
cat("\n   Why: Scales better, fewer parameters to learn\n")

cat("\n2️⃣  CATEGORICAL & CONTINUOUS MIX?")
cat("\n   → Choose NAIVE BAYES")
cat("\n   Why: Handles both naturally\n")

cat("\n3️⃣  NEED PROBABILITY SCORES?")
cat("\n   → Choose NAIVE BAYES")
cat("\n   Why: Gives P(class | data) directly\n")

cat("\n4️⃣  SMALL TO MEDIUM DATA (1K-10K rows)?")
cat("\n   → Choose LDA or QDA")
cat("\n   Why: More stable estimates\n")

cat("\n5️⃣  NEED INTERPRETABILITY?")
cat("\n   → Choose NAIVE BAYES > LDA > QDA")
cat("\n   Why: Naive Bayes easiest to explain\n")

cat("\n6️⃣  CLASSES HAVE DIFFERENT SHAPES?")
cat("\n   → Choose QDA over LDA")
cat("\n   Why: QDA allows different covariances\n")

cat("\n7️⃣  FAST PREDICTION NEEDED?")
cat("\n   → Choose NAIVE BAYES or LDA")
cat("\n   Why: Fastest algorithms\n")

cat("\n╔═════════════════════════════════════════════════════════╗\n")
cat("║  PRACTICAL ADVICE                                       ║\n")
cat("╠═════════════════════════════════════════════════════════╣\n")
cat("║  1. Always try all 3 algorithms                         ║\n")
cat("║  2. Use cross-validation to select                      ║\n")
cat("║  3. Pick based on YOUR dataset characteristics          ║\n")
cat("║  4. Naive Bayes = good default for text/high-dim       ║\n")
cat("║  5. LDA = good default for structured numeric data     ║\n")
cat("║  6. QDA = if LDA underperforms significantly            ║\n")
cat("╚═════════════════════════════════════════════════════════╝\n")
```

---

# 🔧 Part 5: Common Issues & Troubleshooting

```r
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ISSUE 1: NAs (Missing Values) in Data
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

cat("\n❌ ISSUE 1: Missing Values\n")

# Create data with NAs
df_with_na <- iris %>%
  as_tibble() %>%
  mutate(
    Sepal.Length = ifelse(row_number() <= 5, NA, Sepal.Length)
  )

cat("Data with NAs:\n")
print(sum(is.na(df_with_na)))

# Solution 1: Remove rows with NAs
df_clean <- df_with_na %>% drop_na()
cat("✅ After drop_na():", nrow(df_clean), "rows\n")

# Solution 2: Replace with mean
df_imputed <- df_with_na %>%
  mutate(
    Sepal.Length = ifelse(is.na(Sepal.Length),
                          mean(Sepal.Length, na.rm = TRUE),
                          Sepal.Length)
  )
cat("✅ After imputation:", sum(is.na(df_imputed)), "NAs\n")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ISSUE 2: Categorical Variables
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

cat("\n❌ ISSUE 2: Categorical Variables (LDA/QDA only accept continuous)\n")

# Create data with categorical
df_cat <- iris %>%
  as_tibble() %>%
  mutate(
    Size = ifelse(Sepal.Length > 5.8, "Large", "Small")
  )

# ❌ This will error:
# lda_bad <- lda(Species ~ ., data = df_cat)
# Error: variables must be continuous

# ✅ Solution 1: Remove categorical
df_cat_clean <- df_cat %>% select(-Size)
lda_good <- lda(Species ~ ., data = df_cat_clean)
cat("✅ LDA trained after removing categorical\n")

# ✅ Solution 2: Convert categorical to numeric
df_cat_encoded <- df_cat %>%
  mutate(Size_num = as.numeric(factor(Size)))
lda_good2 <- lda(Species ~ . - Size, data = df_cat_encoded)
cat("✅ LDA trained with encoded categorical\n")

# ✅ Solution 3: Use Naive Bayes (handles categorical naturally!)
nb_cat <- train(Species ~ ., data = df_cat, method = "naive_bayes")
cat("✅ Naive Bayes handles categorical directly!\n")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ISSUE 3: Class Imbalance
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

cat("\n❌ ISSUE 3: Imbalanced Classes\n")

# Create imbalanced data
df_imbal <- iris %>%
  as_tibble() %>%
  slice(-c(51:100))  # Remove versicolor

cat("Class distribution:\n")
print(table(df_imbal$Species))

# ❌ Problem: Accuracy can be misleading
# If 75% are setosa, predicting "all setosa" = 75% accuracy!

# ✅ Solution 1: Use stratified sampling
set.seed(42)
idx <- createDataPartition(df_imbal$Species, p = 0.8, 
                           list = FALSE, times = 1)
# createDataPartition DOES stratification by default!

# ✅ Solution 2: Use appropriate metrics
ctrl <- trainControl(
  method = "cv",
  number = 5,
  summaryFunction = multiClassSummary,  # Better for imbalanced
  metric = "F1"  # Or use ROC, not just Accuracy
)

# ✅ Solution 3: Use class weights
# (Different packages have different syntax)
cat("✅ Use metrics beyond Accuracy for imbalanced data!\n")
```

---

**สำเร็จแล้ว! ตอนนี้คุณมี practical code สำหรับทั้ง 3 algorithms! 🚀**
