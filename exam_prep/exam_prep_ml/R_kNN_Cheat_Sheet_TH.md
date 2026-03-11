# 📋 CHEAT SHEET: R Basics & kNN at a Glance

---

## 🔴 R Basics: ตัวการที่ต้องใช้ทั้งวัน

### Data Types & Structures

```r
# Vector: ทีม "ลูกศร" 
x <- c(1, 2, 3)
x[1]        # ตัวแรก → 1
x[2:3]      # ตัวที่ 2-3 → 2 3
length(x)   # นับทั้งหมด → 3

# Data Frame: ตารางหลัก (ใช้บ่อยที่สุด!)
df <- data.frame(
  name = c("A", "B", "C"),
  age = c(25, 30, 35),
  score = c(80, 85, 90)
)
df$age              # เลือก column → 25 30 35
df[1, ]             # เลือก row 1
nrow(df)            # จำนวน rows → 3
ncol(df)            # จำนวน columns → 3

# Tibble: Data frame แล้วแต่ดี (จาก tidyverse)
library(tidyverse)
df <- tibble(
  name = c("A", "B", "C"),
  age = c(25, 30, 35),
  score = c(80, 85, 90)
)
```

---

### Common Functions

| Function | ทำอะไร | ตัวอย่าง |
|----------|--------|---------|
| `head(x)` | แสดง 6 rows แรก | `head(iris)` |
| `str(x)` | โครงสร้างข้อมูล | `str(iris)` |
| `summary(x)` | สถิติพื้นฐาน | `summary(iris)` |
| `nrow(x)` | นับ rows | `nrow(iris)` → 150 |
| `ncol(x)` | นับ columns | `ncol(iris)` → 5 |
| `dim(x)` | Dimensions (rows, cols) | `dim(iris)` → 150 5 |
| `unique(x)` | ค่า unique | `unique(iris$Species)` |
| `table(x)` | นับจำนวน | `table(iris$Species)` |
| `mean(x)` | ค่าเฉลี่ย | `mean(iris$Sepal.Length)` |
| `sd(x)` | ส่วนเบี่ยงเบนมาตรฐาน | `sd(iris$Sepal.Length)` |
| `sort(x)` | เรียงลำดับ | `sort(c(3, 1, 2))` |

---

### Pipes (%>%): เชื่อมคำสั่ง

```r
# ❌ Nested (ยุ่น):
result <- select(filter(arrange(iris, Species), Sepal.Length > 5), Sepal.Length)

# ✅ With pipes (สวย):
result <- iris %>%
  arrange(Species) %>%
  filter(Sepal.Length > 5) %>%
  select(Sepal.Length)

# แปลว่า:
# 1. เอา iris
# 2. เรียงตาม Species
# 3. เลือก Sepal.Length > 5
# 4. เลือก column Sepal.Length
```

---

### dplyr verbs: จัดการข้อมูล

```r
library(tidyverse)

iris %>%
  # ✅ filter: เลือก rows
  filter(Species == "setosa") %>%
  
  # ✅ select: เลือก columns
  select(Sepal.Length, Sepal.Width, Species) %>%
  
  # ✅ mutate: สร้าง column ใหม่
  mutate(
    size_category = ifelse(Sepal.Length > 5.5, "Large", "Small"),
    ratio = Sepal.Length / Sepal.Width
  ) %>%
  
  # ✅ arrange: เรียงลำดับ
  arrange(desc(Sepal.Length)) %>%
  
  # ✅ group_by + summarise: รวมข้อมูล
  group_by(size_category) %>%
  summarise(
    count = n(),
    mean_width = mean(Sepal.Width),
    min_length = min(Sepal.Length)
  )
```

---

### Plotting with ggplot2

```r
library(ggplot2)

# ✅ Basic scatter plot
ggplot(iris, aes(x = Sepal.Length, y = Sepal.Width, color = Species)) +
  geom_point(size = 3) +
  labs(title = "Iris Flowers", x = "Sepal Length", y = "Sepal Width") +
  theme_bw()

# ✅ Box plot
ggplot(iris, aes(x = Species, y = Sepal.Length, fill = Species)) +
  geom_boxplot(alpha = 0.5) +
  theme_bw()

# ✅ Histogram
ggplot(iris, aes(x = Sepal.Length, fill = Species)) +
  geom_histogram(bins = 20, alpha = 0.7) +
  facet_wrap(~Species) +  # Separate plots by Species
  theme_bw()
```

---

## 🟠 Machine Learning Workflow: 8 Steps

```
INPUT: Raw Data
  ↓
1️⃣  LOAD → read.csv(), read_xlsx()
  ↓
2️⃣  EXPLORE → str(), summary(), table(), plot()
  ↓
3️⃣  CLEAN → handle NA, outliers
  ↓
4️⃣  PREPROCESS → scale(), encode categorical
  ↓
5️⃣  SPLIT → 80% train, 20% test
  ↓
6️⃣  TUNE → Cross-validation, tune hyperparameters
  ↓
7️⃣  PREDICT → Make predictions on test set
  ↓
8️⃣  EVALUATE → Accuracy, Confusion Matrix
  ↓
OUTPUT: Trained Model + Performance Metrics
```

---

## 🟡 kNN Algorithm at a Glance

### How kNN Works

```
NEW DATA
  ↓
1. Calculate distance to ALL training data
   distance = √[(x₁-x₂)² + (y₁-y₂)² + ...]
  ↓
2. Find k NEAREST neighbors (k=5 → top 5)
  ↓
3. Get the class of k neighbors
   Normal, Normal, Chemical, Normal, Overt
  ↓
4. VOTE (majority rule)
   Normal: 3 ✅ WINNER!
   Chemical: 1
   Overt: 1
  ↓
PREDICTION: Normal
```

---

### kNN Parameters

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| **k** | ? | 1-25+ | Odd number preferred |
| **distance** | Euclidean | Various | Most common |
| **scale** | NO ⚠️ | YES/NO | Always YES! |
| **train/test split** | ? | 70/30 or 80/20 | 80/20 common |
| **CV folds** | ? | 3-10 | 5 is balanced |

---

### Strengths & Weaknesses of kNN

| Strength ✅ | Weakness ❌ |
|-----------|-----------|
| Very simple | Slow on large data |
| Easy to understand | Can't handle categorical |
| No assumptions | Poor in high dimensions |
| Works with small data | Sensitive to outliers |
| | Can't interpret |

---

## 🔵 Complete kNN Code Skeleton

```r
# ════════════════════════════════════════════════════════
# kNN COMPLETE PIPELINE
# ════════════════════════════════════════════════════════

library(tidyverse)
library(caret)
library(mclust)

# 1️⃣ LOAD
data("diabetes", package = "mclust")
df <- as_tibble(diabetes)

# 2️⃣ EXPLORE
str(df)
summary(df)
table(df$class)
ggplot(df, aes(x = glucose, y = insulin, color = class)) + 
  geom_point() + theme_bw()

# 3️⃣ PREPROCESS: SCALE (важно!)
df_scaled <- df %>%
  mutate(across(where(is.numeric), scale))

# 4️⃣ SPLIT
set.seed(42)
idx <- createDataPartition(df_scaled$class, p = 0.8, list = FALSE)
train <- df_scaled[idx, ]
test <- df_scaled[-idx, ]

# 5️⃣ TUNE WITH CV
ctrl <- trainControl(method = "cv", number = 5)
knn_model <- train(
  class ~ glucose + insulin + sspg,
  data = train,
  method = "knn",
  tuneGrid = expand.grid(k = seq(1, 21, 2)),
  trControl = ctrl
)

# 6️⃣ PREDICT
predictions <- predict(knn_model, test)

# 7️⃣ EVALUATE
cm <- confusionMatrix(predictions, test$class)
print(cm)
cat("Accuracy:", cm$overall['Accuracy'], "\n")

# 8️⃣ NEW DATA
new <- data.frame(glucose = 100, insulin = 10, sspg = 100)
predict(knn_model, new)  # → Normal/Chemical/Overt
```

---

## ⚫ Common Mistakes & Fixes

| ❌ Mistake | ✅ Fix |
|-----------|--------|
| ลืม scale | `mutate(across(where(is.numeric), scale))` |
| Train on all data | Use `createDataPartition()` |
| Tune on test set | Use `trainControl(method = "cv")` |
| Choose k=1 | Tune with CV, usually k=3-7 better |
| Don't handle NA | Use `drop_na()` or `na.omit()` |
| Categorical variables | Convert to numeric first |
| No train/test split | DO THIS FIRST! |

---

## 📊 Confusion Matrix Quick Reference

```
              Actual
            Neg  Pos
Pred   Neg   TN  FN
       Pos   FP  TP

TP (True Positive): Predicted Pos, Actually Pos ✅
TN (True Negative): Predicted Neg, Actually Neg ✅
FP (False Positive): Predicted Pos, Actually Neg ❌
FN (False Negative): Predicted Neg, Actually Pos ❌

Accuracy = (TP + TN) / Total
Sensitivity = TP / (TP + FN)  ← Focus on Pos
Specificity = TN / (TN + FP)  ← Focus on Neg
```

---

## 🎯 Quick Decision Guide

```
Q1: What's your data?
  → Categorical? Use Nominal Tests (Chi-square, McNemar)
  → Continuous? Continue...

Q2: How many records?
  → < 1000? kNN works fine
  → > 10000? kNN might be slow
  → > 100000? Consider other algorithms

Q3: How many features?
  → 2-3? Plot them! Visualize easily
  → 4-20? Typical, kNN works
  → > 100? High dimensional, kNN struggles ("curse of dimensionality")

Q4: Ready to code?
  → Load → Explore → Scale → Split → Tune → Predict → Evaluate
```

---

## 🔧 Useful Code Snippets

### Check for Missing Values
```r
sum(is.na(df))          # Count NAs
is.na(df)               # TRUE/FALSE for each cell
df %>% drop_na()        # Remove rows with NA
df %>% drop_na(column)  # Remove NA in specific column
```

### Handle Categorical Variables
```r
# Convert factor to numeric
df <- df %>%
  mutate(
    color_num = as.numeric(factor(color))
  )

# One-hot encoding (multiple binary columns)
library(fastDummies)
df <- dummy_columns(df, select_columns = "color", remove_first_dummy = TRUE)
```

### Create Training/Test Split
```r
# Method 1: caret
idx <- createDataPartition(df$class, p = 0.8, list = FALSE)
train <- df[idx, ]
test <- df[-idx, ]

# Method 2: Base R
set.seed(42)
idx <- sample(1:nrow(df), size = 0.8 * nrow(df))
train <- df[idx, ]
test <- df[-idx, ]
```

### Scale/Normalize
```r
# Z-score normalization (mean=0, sd=1)
df_scaled <- df %>%
  mutate(across(where(is.numeric), ~scale(.) %>% as.numeric()))

# Min-Max normalization (0-1)
df_norm <- df %>%
  mutate(across(where(is.numeric), ~(. - min(.)) / (max(.) - min(.))))
```

---

## 📚 Key Concepts Explained Simply

| Term | ความหมาย | Analogy |
|------|---------|---------|
| **Feature** | Input variable | 📊 column ในตาราง |
| **Target** | Output variable (class) | 🎯 หลักการพยากรณ์ |
| **Training Set** | ข้อมูลที่สอน model | 📖 ตัวอักษร |
| **Test Set** | ข้อมูลที่วัดความดี | 📝 สอบ |
| **Overfitting** | Fit train เกินไป | 🎨 จำจมูกศิลปหน้าแล้ว |
| **Hyperparameter** | ค่าที่ตั้งเอง (k) | ⚙️ สวิทช์ที่ปรับ |
| **Cross-Validation** | Split data k ครั้ง | 🔄 ทดลอง k ครั้ง |
| **Accuracy** | ถูก / ทั้งหมด | ✅ 90% ตอบถูก |
| **Distance** | ระยะห่างระหว่าง points | 📏 กี่นิ้ว? |
| **Vote** | ลงคะแนนเลือก class | 🗳️ 5 เสียง → normal 3 เสียง |

---

## 🚀 Learning Path

```
Week 1:
  □ Master R basics (vectors, data frames, pipes)
  □ Learn dplyr (filter, select, mutate, group_by)
  □ Practice ggplot2 (scatter, box, histogram)

Week 2:
  □ Understand ML workflow (8 steps)
  □ Learn about data preprocessing
  □ Practice train/test split

Week 3:
  □ Learn kNN algorithm deeply
  □ Code kNN step by step
  □ Understand hyperparameter tuning

Week 4:
  □ Practice with multiple datasets (iris, diamonds, etc.)
  □ Try different k values
  □ Understand cross-validation

Week 5:
  □ Learn about model evaluation metrics
  □ Try nested cross-validation
  □ Visualize decision boundaries

Week 6+:
  □ Learn other algorithms (Logistic Regression, Trees, etc.)
  □ Compare algorithms
  □ Build real projects
```

---

## 💡 Pro Tips

1. **Always plot first** - ตรวจสอบข้อมูลด้วยกราฟ
2. **Always scale** - เด็กใหม่มักลืม → นี่คือ #1 mistake!
3. **Use random seed** - `set.seed(42)` → reproducible results
4. **Stratified sampling** - `createDataPartition()` ทำ automatically
5. **Cross-validate** - ไม่เพียงแค่ simple train/test split
6. **Check confusion matrix** - accuracy ไม่พอ!
7. **Consider class imbalance** - 100 vs 10 → metrics matter
8. **Profile your code** - ช้าไหม? `Sys.time()` ตรวจ
9. **Document everything** - comments บอกทำไม ไม่ใช่ว่า
10. **Version your data** - `seed`, `date`, `package versions`

---

**เตรียมตัวสอบแล้ว! 🎓 ลงมือเขียน code กันเลย!**
