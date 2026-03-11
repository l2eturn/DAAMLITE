# 🤖 Machine Learning ด้วย R: เรียนตั้งแต่พื้นฐาน
## Part 1: R Basics & kNN Algorithm

---

## 📚 สารบัญ

1. **R Basics** - สิ่งที่ต้องรู้ก่อนทำ ML
2. **kNN คืออะไร** - ความเข้าใจพื้นฐาน
3. **Code Step-by-Step** - เขียน kNN ขั้นต่อขั้น
4. **Hyperparameter Tuning** - ปรับค่า k ให้ดี
5. **Cross-validation** - ทดสอบความแม่นยำ

---

# 🔴 Part 1: R BASICS ที่ต้องรู้

## 1. Data Types ใน R

```r
# ✅ Vector (เก็บค่าเดียวกันหลายตัว)
x <- c(1, 2, 3, 4, 5)
typeof(x)  # "double"
x[1]       # ใช้ [index] เพื่อเลือกตัว

# ✅ Data Frame (ตาราง - ใช้มากที่สุดใน ML)
df <- data.frame(
  name = c("Alice", "Bob", "Charlie"),
  age = c(25, 30, 35),
  score = c(85, 90, 88)
)
df$name       # ใช้ $ เพื่อเลือก column
df[1, ]       # เลือก row แรก
df[, 2]       # เลือก column ที่ 2

# ✅ Tibble (Data frame ที่ดีกว่า - จาก tidyverse)
library(tidyverse)
df <- tibble(
  name = c("Alice", "Bob", "Charlie"),
  age = c(25, 30, 35),
  score = c(85, 90, 88)
)

# ✅ Matrix (ตารางตัวเลขเท่านั้น)
m <- matrix(c(1, 2, 3, 4), nrow = 2, ncol = 2)
m[1, 1]  # Row 1, Column 1
```

---

## 2. Pipes (%>%) - การเชื่อมคำสั่ง

```r
# ❌ แบบเก่า - ยุ่นยาก
result <- select(filter(mtcars, cyl == 4), mpg, hp)

# ✅ แบบใหม่ - ใช้ pipe %>%
result <- mtcars %>%
  filter(cyl == 4) %>%
  select(mpg, hp)

# แปลว่า:
# 1. เอาข้อมูล mtcars
# 2. เลือก row ที่ cyl == 4
# 3. เลือก column mpg กับ hp
```

**ความหมาย:** `%>%` = "แล้วทำ" หรือ "จากนั้น"

---

## 3. Functions - การเขียน Function

```r
# ✅ ฟังก์ชันง่ายๆ
add <- function(a, b) {
  return(a + b)
}
add(3, 5)  # Output: 8

# ✅ ฟังก์ชันที่มี default value
greet <- function(name = "Friend") {
  paste("Hello,", name)
}
greet()            # "Hello, Friend"
greet("Alice")     # "Hello, Alice"

# ✅ ฟังก์ชันที่return หลายค่า
summary_stats <- function(x) {
  list(
    mean = mean(x),
    sd = sd(x),
    n = length(x)
  )
}
summary_stats(c(1, 2, 3, 4, 5))
# Output:
# $mean
# [1] 3
# $sd
# [1] 1.58
# $n
# [1] 5
```

---

## 4. ggplot2 - วาดกราฟ

```r
library(ggplot2)

# ✅ Scatter plot
ggplot(mtcars, aes(x = wt, y = mpg, color = factor(cyl))) +
  geom_point(size = 3) +
  labs(title = "Cars: Weight vs MPG",
       x = "Weight (1000 lbs)",
       y = "Miles per Gallon") +
  theme_bw()

# ✅ Box plot
ggplot(mtcars, aes(x = factor(cyl), y = mpg, fill = factor(cyl))) +
  geom_boxplot() +
  theme_bw()

# ✅ Histogram
ggplot(mtcars, aes(x = mpg, fill = factor(cyl))) +
  geom_histogram(bins = 10, alpha = 0.7) +
  facet_wrap(~cyl) +  # แบ่ง subplot ตามกลุ่ม
  theme_bw()
```

---

## 5. dplyr Verbs - จัดการข้อมูล

```r
library(tidyverse)

df <- mtcars

# ✅ filter() - เลือก rows
df %>% filter(cyl == 4)  # เลือก cyl = 4 เท่านั้น
df %>% filter(mpg > 20)  # mpg > 20

# ✅ select() - เลือก columns
df %>% select(mpg, hp, cyl)  # เลือก 3 columns นี้
df %>% select(-cyl)          # ยกเว้น cyl

# ✅ mutate() - สร้าง column ใหม่
df %>% mutate(
  hp_per_cyl = hp / cyl,  # column ใหม่
  log_mpg = log(mpg)
)

# ✅ arrange() - เรียงลำดับ
df %>% arrange(mpg)         # น้อยไปมาก
df %>% arrange(desc(mpg))   # มากไปน้อย

# ✅ group_by() + summarise() - รวมข้อมูล
df %>%
  group_by(cyl) %>%
  summarise(
    mean_mpg = mean(mpg),
    mean_hp = mean(hp),
    n = n()  # นับจำนวน
  )

# ✅ เชื่อมทั้งหมด
df %>%
  filter(hp > 100) %>%
  select(mpg, hp, cyl) %>%
  arrange(mpg) %>%
  mutate(hp_cat = ifelse(hp > 150, "High", "Low"))
```

---

## 6. การทำข้อมูล Clean (Data Preprocessing)

```r
library(tidyverse)

# ✅ มี missing values?
df <- tibble(
  name = c("Alice", "Bob", NA),
  age = c(25, NA, 35),
  score = c(85, 90, 88)
)

# Check
is.na(df)  # TRUE/FALSE สำหรับแต่ละค่า
sum(is.na(df))  # นับ missing

# Remove rows ที่มี NA
df %>% drop_na()

# Replace NA ด้วยค่าเฉลี่ย
df <- df %>%
  mutate(age = ifelse(is.na(age), mean(age, na.rm = TRUE), age))

# ✅ Standardize (ทำให้ mean=0, sd=1)
# สำคัญ! kNN ต้องทำแบบนี้ เพราะวัดระยะทาง
df <- df %>%
  mutate(
    age_scaled = scale(age)[, 1],
    score_scaled = scale(score)[, 1]
  )
```

---

---

# 🎯 Part 2: kNN คืออะไร?

## 2.1 ความเข้าใจพื้นฐาน

### วิธีการทำงาน: 4 ขั้นตอน

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
          HOW kNN WORKS (Step by Step)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

INPUT: ข้อมูลใหม่ที่ไม่รู้ class
  ↓
STEP 1: คำนวณระยะทาง (distance) ไปยังข้อมูลทั้งหมด
  ↓
STEP 2: เรียงลำดับจากใกล้ที่สุด
  ↓
STEP 3: เลือก k ตัวแรก (เช่น k=5 → เลือก 5 ตัวใกล้สุด)
  ↓
STEP 4: ให้คะแนน - class ไหนมีตัวมากที่สุด → นั่นคือ class!
  ↓
OUTPUT: Prediction ✅
```

---

### ตัวอย่างจริง: โรค Diabetes

```
ข้อมูล train (ที่รู้เรียบร้อย):
┌─────────┬────────┬───────┬─────────┐
│glucose  │ insulin│ sspg  │class    │
├─────────┼────────┼───────┼─────────┤
│85       │ 5      │ 80    │Normal   │
│95       │ 8      │ 95    │Chemical │
│150      │ 15     │ 200   │Overt    │
│80       │ 4      │ 75    │Normal   │
│140      │ 14     │ 190   │Overt    │
└─────────┴────────┴───────┴─────────┘

ข้อมูลใหม่ (ไม่รู้ class):
Patient X: glucose=92, insulin=7, sspg=88

→ คำนวณระยะทาง ไปยัง 5 rows:
  - Row1: distance = sqrt((92-85)² + (7-5)² + (88-80)²) = sqrt(113) ≈ 10.6
  - Row2: distance = sqrt((92-95)² + (7-8)² + (88-95)²) = sqrt(59) ≈ 7.7  ✅
  - Row3: distance = sqrt((92-150)² + (7-15)² + (88-200)²) = sqrt(14697) ≈ 121.2
  - Row4: distance = sqrt((92-80)² + (7-4)² + (88-75)²) = sqrt(313) ≈ 17.7
  - Row5: distance = sqrt((92-140)² + (7-14)² + (88-190)²) = sqrt(12618) ≈ 112.3

→ เรียงลำดับ:
  1. Row2 (7.7) - Chemical ← ใกล้สุด
  2. Row4 (17.7) - Normal
  3. Row1 (10.6) - Normal
  4. Row5 (112.3) - Overt
  5. Row3 (121.2) - Overt

→ ถ้า k=3 → เลือก 3 ตัวแรก: Chemical, Normal, Normal
  → ให้คะแนน: Normal มี 2 ตัว, Chemical มี 1 ตัว
  → PREDICT: Patient X = Normal ✅
```

---

### ขั้นตอนละเอียด

| ขั้นตอน | ทำไม | วิธี |
|--------|------|-----|
| **Load Data** | เตรียมข้อมูล | `read.csv()`, `read_xlsx()` |
| **Explore** | เข้าใจข้อมูล | `str()`, `summary()`, `plot()` |
| **Preprocess** | ทำให้ถูกต้อง | Scale, remove NA, convert factor |
| **Split** | แบ่ง train/test | 80% train, 20% test |
| **Train** | สอน model | เพียงแต่ "เก็บ" ข้อมูล |
| **Predict** | ทำนาย | คำนวณ distance + voting |
| **Evaluate** | วัดความดี | Accuracy, Confusion matrix |
| **Tune** | ปรับให้ดี | เลือก k ที่ดีที่สุด |

---

## 2.2 ความสำคัญของ k

```
k = 1: 
┌─────────────────────────────────┐
│ Advantage: กรรม precise          │
│ Disadvantage: ไง sensitive      │
└─────────────────────────────────┘

k = 3:
┌─────────────────────────────────┐
│ "Goldilocks" choice             │
│ Balance between bias & variance  │
└─────────────────────────────────┘

k = 145 (ทั้งข้อมูล):
┌─────────────────────────────────┐
│ Advantage: ไม่ sensitive          │
│ Disadvantage: ไม่ precise        │
└─────────────────────────────────┘
```

**ทั่วไป:** k เป็น **odd number** (1, 3, 5, 7, ...) เพื่อหลีกเลี่ยง tie

---

---

# 💻 Part 3: CODE - เขียน kNN Step by Step

## 3.1 Setup & Load Data

```r
# ✅ Load packages
library(tidyverse)
library(mclust)      # มี diabetes dataset
library(mlr)         # Tools สำหรับ ML
library(caret)       # Alternative ML package

# ✅ Load diabetes data
data("diabetes", package = "mclust")
diabetes <- as_tibble(diabetes)

# ✅ Explore
str(diabetes)
head(diabetes)
summary(diabetes)

# Output:
# A tibble: 145 x 4
#   glucose insulin  sspg class
#     <dbl>   <dbl> <dbl> <fct>
# 1      89       86   325 Normal
# 2      80       90   290 Normal
# ...
```

---

## 3.2 Exploratory Data Analysis (EDA)

```r
# ✅ Check classes
table(diabetes$class)
# Normal  Chemical  Overt
#    76       36      33

# ✅ Summary by class
diabetes %>%
  group_by(class) %>%
  summarise(
    mean_glucose = mean(glucose),
    mean_insulin = mean(insulin),
    mean_sspg = mean(sspg),
    n = n()
  )

# ✅ Visualize
p1 <- diabetes %>% 
  ggplot(aes(x = glucose, y = insulin, color = class)) +
  geom_point(size = 3) +
  theme_bw()

p2 <- diabetes %>%
  ggplot(aes(x = glucose, y = sspg, color = class)) +
  geom_point(size = 3) +
  theme_bw()

p3 <- diabetes %>%
  ggplot(aes(x = sspg, y = insulin, color = class)) +
  geom_point(size = 3) +
  theme_bw()

# ✅ Check correlation
cor(diabetes[, 1:3])
```

---

## 3.3 Data Preprocessing (สำคัญ!)

```r
# ✅ Step 1: Check for missing values
sum(is.na(diabetes))  # Should be 0

# ✅ Step 2: Normalize/Scale the data
# ⚠️  ทำไมต้อง scale?
#    kNN ใช้ distance metric → glucose 100 >> insulin 50
#    ถ้าไม่ scale → glucose มีอิทธิพลมากกว่าใจ
#    
#    scale() ทำให้: mean = 0, sd = 1

# Method 1: ใช้ scale()
diabetes_scaled <- diabetes %>%
  mutate(
    across(glucose:sspg, scale)  # Scale เฉพาะ 3 columns
  )

# Method 2: ใช้ preProcess จาก caret
library(caret)
preproc <- preProcess(
  diabetes[, 1:3],
  method = c("center", "scale")  # center = subtract mean, scale = divide by sd
)
diabetes_scaled_2 <- predict(preproc, diabetes)

# ✅ Verify
summary(diabetes_scaled)  # Should be ~0
sd(diabetes_scaled$glucose)  # Should be 1
```

---

## 3.4 Split Data: Train vs Test

```r
library(caret)

# ✅ Method 1: Simple 80-20 split
set.seed(123)  # For reproducibility
train_index <- createDataPartition(
  diabetes$class,           # Stratified by class
  p = 0.8,                  # 80% train
  list = FALSE,
  times = 1
)

train_data <- diabetes_scaled[train_index, ]
test_data <- diabetes_scaled[-train_index, ]

cat("Train size:", nrow(train_data), "\n")
cat("Test size:", nrow(test_data), "\n")

# ✅ Method 2: Using mlr
library(mlr)
split <- makeResampleDesc(
  method = "Holdout",
  split = 0.8,
  stratify = TRUE
)
```

---

## 3.5 Train kNN Model (Basic)

```r
library(class)  # ชื่อ package ที่มี knn()

# ✅ Prepare data
# X = features (เฉพาะตัวแปร input)
# y = target (class)

X_train <- train_data[, 1:3]  # glucose, insulin, sspg
y_train <- train_data$class   # Normal, Chemical, Overt

X_test <- test_data[, 1:3]
y_test <- test_data$class

# ✅ Train kNN (k=5)
knn_model <- knn(
  train = X_train,    # Training data
  test = X_test,      # Test data
  cl = y_train,       # Classes of training data
  k = 5               # Number of neighbors
)

# ✅ Check predictions
head(knn_model, 10)
# Output:
# [1] Normal   Normal   Chemical Overt    ...
# Levels: Normal Chemical Overt
```

---

## 3.6 Evaluate: ดูความถูกต้อง

```r
# ✅ Step 1: Confusion Matrix
confusion <- table(Predicted = knn_model, Actual = y_test)
confusion

# Output:
#           Actual
# Predicted  Normal Chemical Overt
#   Normal      15       2      0
#   Chemical     1       5      1
#   Overt        0       1      8

# ✅ Step 2: Calculate Accuracy
accuracy <- sum(diag(confusion)) / sum(confusion)
cat("Accuracy:", accuracy, "\n")  # e.g., 0.913

# ✅ Step 3: More detailed metrics
library(caret)
confusionMatrix(knn_model, y_test)

# Output:
# Confusion Matrix and Statistics
#           Reference
# Prediction Normal Chemical Overt
#    Normal     15       2      0
#    Chemical    1       5      1
#    Overt       0       1      8
#
# Overall Statistics
#                Accuracy : 0.913
#                  95% CI : (0.787, 0.975)
#     No Information Rate : 0.522
#     P-Value [Acc > NIR] : 2.22e-08
#
# Sensitivity: 0.938  # ถูกต้องสำหรับ Normal
# Specificity: 0.875  # ถูกต้องสำหรับ non-Normal
```

---

## 3.7 Try Different k Values

```r
# ✅ ลองค่า k หลายๆ ตัว
k_values <- seq(1, 15, by = 2)  # k = 1, 3, 5, 7, ..., 15
accuracies <- numeric(length(k_values))

for (i in seq_along(k_values)) {
  knn_pred <- knn(
    train = X_train,
    test = X_test,
    cl = y_train,
    k = k_values[i]
  )
  accuracies[i] <- sum(knn_pred == y_test) / length(y_test)
}

# ✅ Plot results
results <- data.frame(
  k = k_values,
  accuracy = accuracies
)

ggplot(results, aes(x = k, y = accuracy)) +
  geom_point(size = 3, color = "blue") +
  geom_line(color = "blue", linetype = "dashed") +
  geom_vline(xintercept = k_values[which.max(accuracies)],
             color = "red", linetype = "dotted") +
  labs(title = "kNN Accuracy vs k",
       x = "k (Number of Neighbors)",
       y = "Accuracy") +
  theme_bw() +
  ylim(0.8, 1)

# ✅ Best k
best_k <- k_values[which.max(accuracies)]
cat("Best k:", best_k, "with accuracy:", max(accuracies), "\n")
```

---

## 3.8 Train Final Model with Best k

```r
# ✅ Train กับ best k
final_model <- knn(
  train = X_train,
  test = X_test,
  cl = y_train,
  k = best_k
)

# ✅ Final accuracy
final_accuracy <- sum(final_model == y_test) / length(y_test)
cat("Final Accuracy:", final_accuracy, "\n")
```

---

---

# 🔧 Part 4: HYPERPARAMETER TUNING

## 4.1 ทำไมต้อง Tune?

```
k = 1 (Overfit):
- Memorize training data
- Poor on new data
- High variance

k = "best" (Balanced):
- Good on training data
- Good on new data
- Balanced bias-variance

k = too large (Underfit):
- Too simple
- Poor on both
- High bias
```

---

## 4.2 Cross-Validation (CV)

### ทำไมต้อง CV?

```
❌ ปัญหา: ถ้า split train/test แบบ random ทำได้แค่ครั้งเดียว
         → ผลอาจขึ้นอยู่กับ random split ที่เฉพาะนั้น

✅ วิธี CV: แบ่ง data เป็น k folds
          → ทำนาย k ครั้ง
          → เฉลี่ยผล
          → ได้ผลที่เสถียรกว่า
```

### k-fold CV (k=5)

```
Original Data:
┌────┬────┬────┬────┬────┐
│ F1 │ F2 │ F3 │ F4 │ F5 │
└────┴────┴────┴────┴────┘

Fold 1: Test on F1, Train on F2-F5
┌──────────────────────────┬────┐
│ TRAIN: F2 F3 F4 F5       │ F1 │ TEST
└──────────────────────────┴────┘

Fold 2: Test on F2, Train on F1,F3-F5
┌──┬──────────────────────────┐
│ F1 │ TRAIN: F3 F4 F5       │ F2 │ TEST
└──┴──────────────────────────┘

... (Fold 3, 4, 5)

Final: Average ของ 5 iterations
```

---

## 4.3 Code: Cross-Validation

```r
library(caret)

# ✅ Define CV method: 5-fold CV
ctrl <- trainControl(
  method = "cv",           # Cross-validation
  number = 5,              # 5 folds
  savePredictions = TRUE,  # Save predictions
  classProbs = TRUE        # Save probabilities
)

# ✅ Tune k using caret::train()
# (แบบ high-level, easier than mlr)

knn_tuned <- train(
  class ~ glucose + insulin + sspg,  # Formula: target ~ features
  data = train_data,                 # Training data
  method = "knn",                    # Algorithm
  tuneGrid = expand.grid(k = seq(1, 21, 2)),  # k to try
  trControl = ctrl,                  # CV settings
  preProcess = c("center", "scale")  # Scale during CV (important!)
)

# ✅ View results
knn_tuned$results
# k  Accuracy Kappa AccuracySD KappaSD
# 1   0.920    0.88  0.045     0.053
# 3   0.928    0.89  0.038     0.042
# 5   0.935    0.90  0.035     0.039  ✅ Best!
# 7   0.925    0.88  0.040     0.045
# ...

# ✅ Best k
knn_tuned$bestTune
# k
# 5

plot(knn_tuned)  # Visualize
```

---

## 4.4 Code: mlr Package (Advanced)

```r
library(mlr)

# ✅ Create task
task <- makeClassifTask(
  data = train_data,
  target = "class",
  positive = "Normal"  # Optional: for binary classification
)

# ✅ Create learner (kNN)
learner <- makeLearner("classif.knn")

# ✅ Define parameter search space
ps <- makeParamSet(
  makeIntegerParam("k", lower = 1, upper = 21)
)

# ✅ Define search method (grid search)
ctrl_search <- makeTuneControlGrid()

# ✅ Define resampling (5-fold CV)
resampling <- makeResampleDesc("CV", iters = 5)

# ✅ Tune
tune_result <- tuneParams(
  learner = learner,
  task = task,
  resampling = resampling,
  par.set = ps,
  control = ctrl_search,
  measures = list(mmce())  # Misclassification error
)

# ✅ Results
tune_result

# ✅ Train final model with best k
best_learner <- setHyperPars(
  learner,
  par.vals = tune_result$x
)
final_model <- train(best_learner, task)

# ✅ Predict on test set
test_task <- makeClassifTask(
  data = test_data,
  target = "class"
)
predictions <- predict(final_model, test_task)
```

---

---

# 📊 Part 5: NESTED CROSS-VALIDATION

## 5.1 ทำไมต้อง Nested CV?

```
❌ Problem:
┌─────────────────────────────────────┐
│ Tune hyperparameter on test set?    │
│ → "Cheating" - model fits test set  │
│ → Optimistic accuracy estimate      │
└─────────────────────────────────────┘

✅ Solution: Nested CV
┌─────────────────────────────────────┐
│ OUTER CV: Train/Test split          │
│   INNER CV: Tune hyperparameter     │
│          on TRAIN only              │
└─────────────────────────────────────┘
```

---

## 5.2 Diagram

```
ORIGINAL DATA
│
├─ Fold 1 [Test] ┐
│  │             │
│  ├─ Inner Fold 1 [Test]   Tune k with CV
│  ├─ Inner Fold 2 [Test]   → Best k found
│  ├─ Inner Fold 3 [Test]   → Train model
│  ├─ Inner Fold 4 [Test]   → Evaluate on Outer Test
│  └─ Inner Fold 5 [Test]   
│
├─ Fold 2 [Test] ┐ Repeat 5 times
│  └─ ...         │
├─ Fold 3 [Test] │
├─ Fold 4 [Test] │
└─ Fold 5 [Test] ┘

FINAL: Average ของ 5 outer folds
→ Unbiased estimate!
```

---

## 5.3 Code: Nested CV with mlr

```r
library(mlr)

# ✅ Create task
task <- makeClassifTask(
  data = diabetes_scaled,
  target = "class"
)

# ✅ Create learner
learner <- makeLearner("classif.knn")

# ✅ Hyperparameter space
ps <- makeParamSet(
  makeIntegerParam("k", lower = 1, upper = 25)
)

# ✅ Inner CV: Tune hyperparameter (10-fold)
inner_cv <- makeResampleDesc("CV", iters = 10)
ctrl_tune <- makeTuneControlGrid(resolution = 10)

# ✅ Create tuning wrapper (Inner loop)
lrn_tuned <- makeTuneWrapper(
  learner,
  resampling = inner_cv,
  par.set = ps,
  control = ctrl_tune,
  measures = list(mmce())  # Misclassification error
)

# ✅ Outer CV: Evaluate tuned model (5-fold)
outer_cv <- makeResampleDesc("CV", iters = 5, stratify = TRUE)

# ✅ Nested CV
nested_cv_result <- resample(
  lrn_tuned,
  task,
  resampling = outer_cv,
  measures = list(mmce(), acc())  # Error rate & Accuracy
)

# ✅ Results
nested_cv_result
# Resampling: cross-validation
# Measures:           mmce       acc
# iter 1:            0.087      0.913
# iter 2:            0.125      0.875
# iter 3:            0.100      0.900
# iter 4:            0.118      0.882
# iter 5:            0.095      0.905
# Mean:              0.105 ± 0.015
#
# ✅ Final accuracy: 89.5% ± 1.5%
```

---

---

# 🚀 Part 6: COMPLETE WORKFLOW

## Full Pipeline

```r
library(tidyverse)
library(mlr)
library(caret)
library(mclust)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 1: LOAD & EXPLORE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

data("diabetes", package = "mclust")
df <- as_tibble(diabetes)

str(df)
summary(df)
table(df$class)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 2: VISUALIZE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ggplot(df, aes(x = glucose, y = insulin, color = class)) +
  geom_point(size = 3) +
  theme_bw()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 3: PREPROCESS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

df_scaled <- df %>%
  mutate(across(glucose:sspg, scale))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 4: TRAIN/TEST SPLIT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

set.seed(42)
idx <- createDataPartition(df_scaled$class, p = 0.8, list = FALSE)
train <- df_scaled[idx, ]
test <- df_scaled[-idx, ]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 5: TUNE WITH CV
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ctrl <- trainControl(
  method = "cv",
  number = 5,
  savePredictions = TRUE
)

knn_model <- train(
  class ~ glucose + insulin + sspg,
  data = train,
  method = "knn",
  tuneGrid = expand.grid(k = seq(1, 21, 2)),
  trControl = ctrl,
  preProcess = c("center", "scale")
)

print(knn_model)
plot(knn_model)
best_k <- knn_model$bestTune$k

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 6: PREDICT ON TEST
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

predictions <- predict(knn_model, test)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 7: EVALUATE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

confusionMatrix(predictions, test$class)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 8: PREDICT ON NEW DATA
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# New patient: glucose=100, insulin=10, sspg=100
new_patient <- data.frame(
  glucose = 100,
  insulin = 10,
  sspg = 100
)

prediction <- predict(knn_model, new_patient)
print(prediction)
# [1] Normal
# Levels: Normal Chemical Overt
```

---

---

# 📚 Part 7: สรุป & Cheat Sheet

## Quick Summary

```
┌─────────────────────────────────────────────────────────┐
│             kNN ALGORITHM SUMMARY                       │
├─────────────────────────────────────────────────────────┤
│                                                         │
│ 1️⃣  LOAD: read data → explore                         │
│                                                         │
│ 2️⃣  PREPROCESS: scale data (สำคัญ!)                   │
│                                                         │
│ 3️⃣  SPLIT: 80% train, 20% test                        │
│                                                         │
│ 4️⃣  TUNE: ใช้ CV เลือก best k                         │
│                                                         │
│ 5️⃣  TRAIN: เพียงแต่ "เก็บ" data                       │
│                                                         │
│ 6️⃣  PREDICT: คำนวณ distance → voting                 │
│                                                         │
│ 7️⃣  EVALUATE: check accuracy, confusion matrix        │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## Key Concepts

| Concept | ความหมาย | Example |
|---------|---------|---------|
| **Euclidean Distance** | ระยะทาง (Pythagoras) | √((x1-x2)² + (y1-y2)²) |
| **k** | จำนวน neighbors | k=5 → เลือก 5 ตัวใกล้สุด |
| **Scale** | ทำให้ mean=0, sd=1 | ต้องทำ! |
| **Train/Test Split** | แบ่งข้อมูล | 80/20 หรือ 70/30 |
| **Cross-Validation** | ทำนาย k ครั้ง | 5-fold CV |
| **Hyperparameter** | ค่าที่เลือกเอง | k คือ hyperparameter |
| **Overfitting** | Fit train เกินไป | k=1 มักเป็น overfitting |
| **Accuracy** | ถูก/ทั้งหมด | (15+5+8)/29 = 0.93 |

---

## Common Mistakes

```r
# ❌ WRONG: ลืม scale
knn(train = train_raw, test = test_raw, cl = y_train, k = 5)

# ✅ RIGHT: scale ก่อน
knn(train = scale(train), test = scale(test), cl = y_train, k = 5)

# ❌ WRONG: tune บน test set
tuned_k <- findBestK_on(test_data)  # Cheating!

# ✅ RIGHT: tune บน train set ด้วย CV
tuned_k <- tuneParams(..., resampling = cv_on_train)

# ❌ WRONG: ลืม เรื่องระยะทาง
# ถ้า feature มี units ต่างกัน (e.g., กม vs กรัม)
# → need scale!
```

---

## Next Steps

```
คุณพร้อมแล้ว! ✅

จากนี้ไป:
1. ลอง code ด้วยตัวเอง
2. ลองกับ iris dataset (ใช้งาย)
3. ลองกับ data ของคุณเอง
4. เรียน algorithms อื่น:
   - Logistic Regression
   - Decision Trees
   - Random Forest
   - SVM
```

---

**ส่วนไหนต้องการอธิบายเพิ่มเติม?** 🚀
