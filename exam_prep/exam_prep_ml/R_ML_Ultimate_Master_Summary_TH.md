# 📚 ULTIMATE MASTER CHEAT SHEET
## Complete Guide: Statistics Tests + 6 ML Algorithms

---

## 🎓 What You've Learned

```
┌─────────────────────────────────────────────────────────────┐
│         STATISTICAL TESTS & MACHINE LEARNING               │
│                   (Before your exam)                       │
└─────────────────────────────────────────────────────────────┘

PART 1: STATISTICAL TESTS (Para, Nonpara, Nominal)
├─ Parametric Tests (assume normal distribution)
│  ├─ One-sample t-test
│  ├─ Two-sample t-test
│  ├─ One-way ANOVA
│  └─ Two-way ANOVA
│
├─ Nonparametric Tests (no assumptions)
│  ├─ Sign Test
│  ├─ Mann-Whitney U
│  ├─ Wilcoxon Signed-Rank
│  └─ Kruskal-Wallis
│
└─ Nominal Tests (categorical data)
   ├─ Goodness-of-Fit (Chi-square, G-test)
   ├─ Association Tests (Chi-square)
   └─ McNemar Test (paired)

PART 2: MACHINE LEARNING - 6 ALGORITHMS

1. kNN (k-Nearest Neighbors)
   └─ Simple distance-based classifier

2. Naive Bayes
   └─ Probabilistic classifier (Bayes Theorem)

3. LDA (Linear Discriminant Analysis)
   └─ Linear classifier with dimension reduction

4. QDA (Quadratic Discriminant Analysis)
   └─ Flexible curved boundaries

5. SVM (Support Vector Machine)
   └─ Hyperplane with kernel trick

6. Random Forest + XGBoost
   └─ Ensemble methods (bagging + boosting)
```

---

## 🔀 Decision Flow: Statistics Tests

```
START: "Which test should I use?"
  │
  ├─ Is data CATEGORICAL?
  │  ├─ YES → NOMINAL TESTS
  │  │  ├─ 1 variable → Goodness-of-Fit (Chi-square, G-test)
  │  │  ├─ 2+ variables → Association (Chi-square)
  │  │  └─ Paired → McNemar Test
  │  │
  │  └─ NO → Continue...
  │
  ├─ Is data INTERVAL/RATIO CONTINUOUS?
  │  ├─ NO → ORDINAL → Check Normality
  │  └─ YES → Check Normality
  │
  ├─ Is data NORMALLY DISTRIBUTED?
  │  │
  │  ├─ YES → PARAMETRIC TESTS
  │  │  ├─ 1 group → One-sample t-test
  │  │  ├─ 2 groups → Two-sample t-test
  │  │  └─ 3+ groups → One-way ANOVA (+ post-hoc TukeyHSD)
  │  │
  │  └─ NO → NONPARAMETRIC TESTS
  │     ├─ 1 group → Sign Test
  │     ├─ 2 groups → Mann-Whitney U
  │     ├─ 2 groups (paired) → Wilcoxon
  │     └─ 3+ groups → Kruskal-Wallis (+ post-hoc Dunn)
  │
  └─ Done!
```

---

## 🤖 Decision Flow: ML Algorithm

```
START: "Which ML algorithm?"
  │
  ├─ Text/High-dimensional (>100 features)?
  │  ├─ YES → Naive Bayes ⭐⭐⭐
  │  └─ NO → Continue...
  │
  ├─ Small dataset (<1K)?
  │  ├─ YES → Naive Bayes or kNN
  │  └─ NO → Continue...
  │
  ├─ Need interpretability?
  │  ├─ YES → Naive Bayes > LDA > SVM
  │  └─ NO → Continue...
  │
  ├─ Accuracy is CRITICAL?
  │  ├─ YES → XGBoost ⭐⭐⭐
  │  └─ NO → Continue...
  │
  ├─ Speed critical?
  │  ├─ YES → Naive Bayes > LDA
  │  └─ NO → Try all 6 with CV
  │
  └─ Done! (Usually try all 6)
```

---

## 📊 Quick Comparison: All 11 Topics

### Statistical Tests

| Test | Data Type | Groups | Assumption | When |
|------|-----------|--------|-----------|------|
| **t-test** | Continuous | 1-2 | Normal | Parametric comparison |
| **ANOVA** | Continuous | 3+ | Normal | Multiple groups |
| **Sign Test** | Ordinal | 1 | None | No normality |
| **Mann-Whitney** | Continuous/Ordinal | 2 | None | Non-normal |
| **Kruskal-Wallis** | Continuous/Ordinal | 3+ | None | Non-normal |
| **Chi-square** | Categorical | Any | None | Count data |
| **McNemar** | Categorical | 2 paired | None | Before-after |

### ML Algorithms

| Algorithm | Type | Speed | Data | Accuracy | Notes |
|-----------|------|-------|------|----------|-------|
| **kNN** | Distance | Slow | Small | Medium | Simple baseline |
| **NB** | Probability | Fast | Any | Medium | Text ⭐ |
| **LDA** | Linear | Fast | Medium | Good | Interpretable |
| **QDA** | Quadratic | Medium | Large | Good | Flexible |
| **SVM** | Kernel | Slow | Any | Excellent | Complex tuning |
| **RF** | Ensemble | Medium | Large | Excellent | Feature importance |
| **XGB** | Boosting | Medium | Large | Best! | Most accurate |

---

## ✅ Complete R Workflow

```r
# ════════════════════════════════════════════════════════════
# UNIVERSAL ML WORKFLOW
# ════════════════════════════════════════════════════════════

library(tidyverse)
library(caret)

# 1️⃣ LOAD & EXPLORE
data <- read.csv("data.csv")
str(data)
summary(data)
table(data$class)

# 2️⃣ VISUALIZE
ggplot(data, aes(x = feature1, y = feature2, color = class)) +
  geom_point() + theme_bw()

# 3️⃣ CLEAN & PREPROCESS
data <- data %>%
  drop_na() %>%
  mutate(across(where(is.numeric), scale))  # SCALE!

# 4️⃣ SPLIT TRAIN/TEST
set.seed(42)
idx <- createDataPartition(data$class, p = 0.8, list = FALSE)
train <- data[idx, ]
test <- data[-idx, ]

# 5️⃣ CROSS-VALIDATION
ctrl <- trainControl(method = "cv", number = 5)

# 6️⃣ TRAIN ALL 6 (or relevant ones)
knn_model <- train(class ~ ., data = train, 
                   method = "knn", tuneGrid = expand.grid(k = 5), 
                   trControl = ctrl)

nb_model <- train(class ~ ., data = train, 
                  method = "naive_bayes", trControl = ctrl)

lda_model <- train(class ~ ., data = train, 
                   method = "lda", trControl = ctrl)

svm_model <- train(class ~ ., data = train, 
                   method = "svmRadial", trControl = ctrl)

rf_model <- train(class ~ ., data = train, 
                  method = "rf", trControl = ctrl)

xgb_model <- train(class ~ ., data = train, 
                   method = "xgbTree", trControl = ctrl)

# 7️⃣ PREDICT
predictions <- list(
  knn = predict(knn_model, test),
  nb = predict(nb_model, test),
  lda = predict(lda_model, test),
  svm = predict(svm_model, test),
  rf = predict(rf_model, test),
  xgb = predict(xgb_model, test)
)

# 8️⃣ EVALUATE & COMPARE
results <- tibble()
for (i in 1:length(predictions)) {
  cm <- confusionMatrix(predictions[[i]], test$class)
  results <- results %>%
    bind_rows(tibble(
      Algorithm = names(predictions)[i],
      Accuracy = cm$overall['Accuracy'],
      Kappa = cm$overall['Kappa']
    ))
}

print(results %>% arrange(desc(Accuracy)))

# 9️⃣ WINNER ANALYSIS
best <- results$Algorithm[which.max(results$Accuracy)]
cat("✅ BEST ALGORITHM:", best, "\n")
```

---

## 🎯 Critical Things to Remember

### ALWAYS:
```
✅ 1. Split data (train/test) FIRST
✅ 2. Use cross-validation for tuning
✅ 3. SCALE features (except Naive Bayes)
✅ 4. Try multiple algorithms
✅ 5. Compare metrics (not just accuracy)
✅ 6. Check assumptions (for parametric tests)
✅ 7. Visualize data first
✅ 8. Use stratified sampling for imbalanced data
```

### NEVER:
```
❌ 1. Tune on test set (data leakage!)
❌ 2. Use mean/SD for categorical data
❌ 3. Skip preprocessing
❌ 4. Use kNN for high-dimensional data
❌ 5. Apply parametric test without checking normality
❌ 6. Forget to scale for distance-based algorithms
❌ 7. Report only accuracy for imbalanced data
❌ 8. Trust a single train/test split
```

---

## 📋 Quick Command Reference

### Statistical Tests

```r
# Parametric
shapiro.test(x)           # Test normality
t.test(x, mu = 0)         # One-sample t
t.test(x ~ group)         # Two-sample t
aov(y ~ x)                # One-way ANOVA
TukeyHSD(anova_result)     # Post-hoc

# Nonparametric
SIGN.test(x, m = 3)       # Sign test
wilcox.test(x ~ group)    # Mann-Whitney U
wilcox.test(x, y, paired=TRUE)  # Wilcoxon
kruskal.test(y ~ x)       # Kruskal-Wallis
dunnTest(y ~ x)           # Post-hoc

# Nominal
chisq.test(counts)        # Chi-square
mcnemar.test(table)       # McNemar
```

### ML Training

```r
# Train
train(y ~ ., data = train,
      method = "knn",           # algorithm name
      tuneGrid = ...,           # parameters to try
      trControl = ctrl)         # CV settings

# Predict
predict(model, test)           # Classification
predict(model, test, type="prob")  # Probabilities

# Evaluate
confusionMatrix(pred, actual)  # Full metrics
```

---

## 🚀 Exam Preparation Checklist

```
WEEK 1: Statistical Tests
  □ Understand Para vs Nonpara vs Nominal
  □ Practice decision tree
  □ Run examples on your data
  □ Know when to use each test

WEEK 2: kNN & Naive Bayes
  □ Understand distance metrics
  □ Know k selection importance
  □ Understand Bayes Theorem
  □ Practice hyperparameter tuning

WEEK 3: LDA & QDA
  □ Understand discriminant functions
  □ Know assumptions of each
  □ Visualize decision boundaries
  □ Compare performance

WEEK 4: SVM & Ensemble Methods
  □ Understand kernel trick
  □ Know hyperparameters (C, gamma)
  □ Understand bagging vs boosting
  □ Know RF & XGBoost differences

WEEK 5: Integration & Practice
  □ Run all 6 algorithms on 3 datasets
  □ Create comparison reports
  □ Know when to use each algorithm
  □ Practice code from memory
```

---

## 💡 Pro Tips

### For Statistical Tests
- **Always visualize** before testing
- **Check assumptions** (normality, homogeneity)
- **Use appropriate metric** (accuracy, F1, etc.)
- **Consider effect size**, not just p-value
- **Report confidence intervals**

### For ML Algorithms
- **Standardize features** (except tree-based, NB)
- **Use cross-validation** (5 or 10 fold)
- **Tune hyperparameters** systematically
- **Compare multiple algorithms**
- **Understand why algorithm wins** (don't blindly pick)

### For Exams
- **Know the theory** behind each method
- **Practice coding** from scratch
- **Understand outputs** (confusion matrix, etc.)
- **Know when to use which** (decision trees!)
- **Be ready to explain** your choices

---

## 🎓 Final Summary

```
You now know:

📊 STATISTICS (7 tests):
   ✓ When to use parametric vs nonparametric vs nominal
   ✓ How to conduct each test
   ✓ How to interpret results
   ✓ How to do post-hoc analysis

🤖 MACHINE LEARNING (6 algorithms):
   ✓ How each algorithm works (theory)
   ✓ When to use each one (decision guide)
   ✓ How to code each one (practical)
   ✓ How to compare them (benchmarking)

📚 TOTAL: 13 MAJOR TOPICS
   ✓ Hundreds of lines of commented code
   ✓ Multiple examples per method
   ✓ Decision guides for selection
   ✓ Complete workflows ready to use
```

---

## 📞 Quick Q&A

### "Which test/algorithm should I use?"
→ Use the decision trees above!

### "How do I know if my code is correct?"
→ Compare output with textbook examples
→ Check confusion matrix makes sense
→ Verify accuracy is reasonable

### "What if I get different results?"
→ Check random seed (set.seed())
→ Check data scaling
→ Check train/test split
→ Check cross-validation folds

### "Should I always scale data?"
```
✅ YES for: kNN, SVM, LDA, QDA
❌ NO for: Tree-based (RF, XGB), Naive Bayes
```

### "How many folds for cross-validation?"
→ Default: 5-fold
→ Small data: 10-fold
→ Very small: Leave-One-Out CV

### "What if training is slow?"
→ Reduce number of folds
→ Reduce hyperparameter grid
→ Use simpler algorithm (NB, LDA)
→ Sample data for initial testing

---

## 🏁 Before Your Exam

```
✅ Download all 12 files
✅ Read each guide thoroughly
✅ Run ALL practical code
✅ Modify code with your own data
✅ Try different parameters
✅ Document your findings
✅ Practice explaining results
✅ Know decision trees by heart
✅ Get comfortable with metrics
✅ Be ready for any question!
```

---

**สำเร็จแล้ว! คุณพร้อมสอบแล้ว! 🎉**

```
📚 12 Complete Files Created
💻 400+ Lines of R Code
🎯 11 Major Topics Covered
✅ Decision Guides Included
🚀 Ready to Use Workflows
```

**Good Luck on Your Exam! 🍀**
