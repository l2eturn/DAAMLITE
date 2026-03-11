# 🤖 Machine Learning Part 2: Naive Bayes, LDA & QDA
## (Classification Algorithms ต่างจาก kNN)

---

## 📚 สารบัญ

1. **Naive Bayes**: ใช้ Bayes Theorem + Probability
2. **Linear Discriminant Analysis (LDA)**: ใช้ Linear Functions
3. **Quadratic Discriminant Analysis (QDA)**: ใช้ Curved Functions
4. **เปรียบเทียบ 3 algorithms**
5. **Code ตัวอย่าง**

---

# 🔴 Part 1: NAIVE BAYES

## 1.1 ความเข้าใจพื้นฐาน: Bayes' Theorem

### ตัวอย่างจริง: การทดสอบโรค

```
ประชากร:
- 0.2% มีโรค (prior probability) = 0.002
- 99.8% ไม่มีโรค = 0.998

Test ของเรา:
- ถ้าจริงมีโรค → ตรวจเห็นว่ามี 90% (true positive rate)
- ถ้าไม่มีโรค → ตรวจเห็นว่ามี 5% (false positive rate) ← ผล false positive!

คำถาม:
ถ้าทำ test แล้วเห็นว่า "มีโรค" → ความน่าจะเป็นจริงมีโรค = ?

สัญชาตญาณ: 90% (ราวๆ เดียวกับ true positive rate)

จริงๆ คำตอบ: 3.6% เท่านั้น! 🤯
```

### Bayes' Theorem Formula

```
P(Disease | Positive Test) = P(Positive | Disease) × P(Disease)
                              ───────────────────────────────
                                  P(Positive Test)

แปลว่า:
Posterior = (Likelihood × Prior) / Evidence

เมื่อ:
- P(Disease | Test) = Posterior = ความน่าจะเป็นจริงมีโรคหลังได้ผล test
- P(Test | Disease) = Likelihood = ความน่าจะเป็นได้ผล test นี้ถ้ามีโรค
- P(Disease) = Prior = ความน่าจะเป็นมีโรคจากประชากร
- P(Test) = Evidence = ความน่าจะเป็นได้ผล test นี้
```

### ตัวเลข:

```
P(Disease | +) = 0.9 × 0.002 / 0.05
                = 0.0018 / 0.05
                = 0.036 = 3.6% ✅

ทำไมน้อยกว่า 90%?
→ เพราะ false positive มีเยอะ!
  ในประชากร 10,000:
  - 20 คนมีโรคจริง
  - 9980 คนไม่มี แต่ 499 คนได้ผล false positive
  - Total positive: 20 + 499 = 519
  - True positives: 20 / 519 = 3.8% ≈ 3.6%
```

---

## 1.2 Naive Bayes สำหรับ Classification

### ตัวอย่าง: Twitter Classification

```
ต้องการ: Classify tweet เป็น Politics / Sports / Movies / Other

Features (Categorical):
- Has word "opinion" → Y/N
- Has word "score" → Y/N
- Has word "game" → Y/N
- Has word "cinema" → Y/N

ข้อมูล Training (ตัวอย่าง):
┌───────────┬─────────┬───────┬───────┬────────┐
│Category   │ opinion │ score │ game  │ cinema │
├───────────┼─────────┼───────┼───────┼────────┤
│Politics   │  80%    │  10%  │  5%   │  2%    │
│Sports     │  20%    │  70%  │ 90%   │  1%    │
│Movies     │  15%    │  10%  │  5%   │ 95%    │
│Other      │  10%    │  10%  │ 10%   │  2%    │
└───────────┴─────────┴───────┴───────┴────────┘

นั่นคือ likelihood ของแต่ละ word สำหรับแต่ละ class

New Tweet: "opinion score game"

Bayes' Theorem:
P(Politics | words) ∝ P(words | Politics) × P(Politics)
                    = 0.8 × 0.1 × 0.05 × P(Politics)
                    = 0.004 × P(Politics)

P(Sports | words) ∝ 0.2 × 0.7 × 0.9 × P(Sports)
                  = 0.126 × P(Sports)  ← HIGHEST!

P(Movies | words) ∝ 0.15 × 0.1 × 0.05 × P(Movies)
                  = 0.00075 × P(Movies)

P(Other | words) ∝ 0.1 × 0.1 × 0.1 × P(Other)
                 = 0.001 × P(Other)

→ Sports มี highest probability!
→ Predict: Sports ✅
```

---

## 1.3 ความหมาย "Naive"

```
❌ Naive Assumption:
   Features เป็น INDEPENDENT กัน
   
   Meaning: ความน่าจะเป็น "score" ไม่ขึ้นกับ "game"
            แต่จริงๆ ถ้า tweet มี "score" → น่าจะมี "game" มากขึ้น
   
   → Assumption นี้ผิดบ่อย!

✅ แต่ในทางปฏิบัติ:
   Naive Bayes ทำงานดีได้ถึงแม้ assumption ผิด!
   → ผลที่ได้ยังดีเพราะว่า:
     1. ไม่ต้องเก็บ dependency ที่ซับซ้อน
     2. ง่ายต่อการ compute
     3. ต้องข้อมูล training น้อยกว่า
```

---

## 1.4 Continuous vs Categorical Predictors

### Categorical (Yes/No, True/False)
```
Likelihood = Proportion ของ training cases ที่ match

Example:
P(Cinema=Y | Movies) = (# movies with Cinema=Y) / (# total movies)
                      = 95% = 0.95
```

### Continuous (ตัวเลข)
```
Assume: Data เป็น Normal Distribution ใน class นั้น

คำนวณ: Probability density function (PDF) ของ normal distribution

Example: Height
P(Height=175 | Female) = 1/√(2π×SD²) × e^(-0.5×((175-mean)/SD)²)
                        ← Bell curve!
```

---

# 🟠 Part 2: LINEAR DISCRIMINANT ANALYSIS (LDA)

## 2.1 ความเข้าใจพื้นฐาน

### Visual Intuition

```
Original Data (2D):
┌─────────────────────────────────┐
│  ●●●●  (Class A)                │
│      ●●●●                       │
│            ◆◆◆◆  (Class B)      │
│              ◆◆◆◆               │
│                   ■■■■  (Class C)
│                     ■■■■        │
└─────────────────────────────────┘

Problem:
- 3 dimensions คือ 3 variables
- ต้องการ separate classes ให้ชัด

LDA ทำ:
1. สร้าง DiscriminantFunction (DF1)
2. Project data ลงบนแกนใหม่นี้
3. Classes แยกออกจากกันชัดเจน

Result (1D):
┌───────────────────────────────────┐
│●●●●   ◆◆◆◆   ■■■■               │
│      A  B       C                  │
└───────────────────────────────────┘

✅ ลด dimension จาก 2 → 1 แต่ยังแยก classes ได้!
```

---

## 2.2 วิธีการ LDA

### Goal

```
Maximize: (Distance between class means)² / (Variance within each class)

Simple English:
- Make classes as FAR from each other as possible
- Make each class as TIGHT as possible
- Result: Clear separation!
```

### Linear Discriminant Function

```
DF = coefficient₁ × variable₁ + coefficient₂ × variable₂ + ...

Example:
DF = -0.5 × glucose + 1.2 × insulin + 0.85 × sspg

Interpretation:
- glucose: coefficient = -0.5 (ความสำคัญ = 0.5)
- insulin: coefficient = 1.2 (ความสำคัญ = 1.2) ← มีอิทธิพลมากสุด!
- sspg: coefficient = 0.85 (ความสำคัญ = 0.85)
```

---

## 2.3 กี่ Discriminant Functions?

```
Number of DFs = min(Number of Classes - 1, Number of Variables)

Examples:
┌─────────────────────────────────────────┐
│ Classes │ Variables │ DFs    │ Example  │
├─────────┼───────────┼────────┼──────────┤
│ 2       │ 3         │ min(1,3)=1 │ Line   │
│ 3       │ 3         │ min(2,3)=2 │ Plane  │
│ 4       │ 10        │ min(3,10)=3│ 3D     │
│ 100     │ 1000      │ min(99,1000)=99 │ 99D!│
└─────────┴───────────┴────────┴──────────┘

Amazing: 1000 variables → 99 DFs! (Dimension reduction!)
```

---

## 2.4 Assumptions ของ LDA

```
✅ MUST HAVE:
1. Continuous predictor variables
2. Data normally distributed in each class
3. Same covariance matrix for all classes (KEY!)

❌ If violated:
- Still works okay, but less optimal
- QDA might be better
```

---

# 🟡 Part 3: QUADRATIC DISCRIMINANT ANALYSIS (QDA)

## 3.1 ความแตกต่างจาก LDA

```
LDA:
┌─────────────────────────────┐
│ Assumption: Covariance      │
│           เหมือนกันทุก class │
│                             │
│ Decision boundary: STRAIGHT│
└─────────────────────────────┘

QDA:
┌─────────────────────────────┐
│ Assumption: Covariance      │
│           ต่างกันแต่ละ class │
│                             │
│ Decision boundary: CURVED   │
└─────────────────────────────┘
```

### Visual

```
LDA (Straight lines):
┌─────────────────────────────┐
│ ●●●●|◆◆◆◆|■■■■             │ straight line
│     |    |                 │ separates
└─────────────────────────────┘

QDA (Curved lines):
┌─────────────────────────────┐
│ ●●●●    ◆◆◆◆    ■■■■       │ curved line
│    \  /    \  /            │ fits better
│     \/      \/             │
└─────────────────────────────┘
```

---

## 3.2 Covariance คืออะไร?

```
Covariance = "How much do variables move together?"

Example:
- Height & Weight: Positive covariance
  (ตัวสูง → อักษรมี น้ำหนักมากขึ้น)

- Height & Test Score: No covariance
  (ตัวสูง → ไม่เกี่ยวกับคะแนน)

Visual:
┌──────────────────────────────┐
│ ●          ◆◆◆               │
│  ●●       ◆◆ ◆◆              │ LDA assumes
│    ●●    ◆◆   ◆◆             │ same "shape"
│      ●●                      │
│                              │
│ Both ellipses same size/shape│
└──────────────────────────────┘

vs

┌──────────────────────────────┐
│ ●          ◆◆◆               │
│  ●●       ◆◆ ◆◆              │ QDA allows
│    ●●    ◆◆◆◆◆◆             │ different
│                              │ "shapes"
│ Ellipses different sizes/shapes│
└──────────────────────────────┘
```

---

## 3.3 Strengths & Weaknesses

### LDA
```
✅ Pros:
- Simple (straight decision boundary)
- ต้องข้อมูลน้อยกว่า
- Fast

❌ Cons:
- Assumes covariance เหมือนกัน
- Might underfit
```

### QDA
```
✅ Pros:
- More flexible (curved boundary)
- Better when covariance ต่างกัน
- Less restrictive

❌ Cons:
- More complex (curved boundaries)
- ต้องข้อมูลมากกว่า (more parameters)
- Slower
- Might overfit
```

---

---

# 🔵 Part 4: เปรียบเทียบทั้ง 3 Algorithms

## 4.1 Comparison Table

| Feature | Naive Bayes | LDA | QDA |
|---------|------------|-----|-----|
| **Type** | Probabilistic | Linear | Quadratic |
| **Decision Boundary** | Complex | Straight | Curved |
| **Assumptions** | Independence | Normal + Equal Cov | Normal |
| **Speed** | Very Fast ⚡ | Fast | Medium |
| **Data Required** | Small ✓ | Medium | Large |
| **Interpretability** | ⭐⭐⭐ | ⭐⭐ | ⭐ |
| **Scalability** | High | High | Medium |
| **Best for** | Text, High-dim | Structured | Non-linear |

---

## 4.2 Decision Tree: เลือก Algorithm

```
START
  ↓
Q1: High-dimensional data? (> 100 features)
├─ YES → Try Naive Bayes first
└─ NO  → Continue...

Q2: Data normally distributed?
├─ NO  → Naive Bayes or QDA
└─ YES → Continue...

Q3: Classes have similar covariance?
├─ YES → LDA (simple & fast)
└─ NO  → QDA (flexible)

Q4: Data size?
├─ Small  → Naive Bayes or LDA
├─ Medium → LDA or QDA
└─ Large  → Naive Bayes (fewer parameters)

Q5: Need interpretability?
├─ YES → Naive Bayes > LDA > QDA
└─ NO  → Any is fine, choose by performance
```

---

# 🟢 Part 5: R CODE EXAMPLES

## 5.1 Naive Bayes

### Simple Example: Congressional Voting

```r
library(tidyverse)
library(caret)
library(mlbench)
library(naivebayes)  # For Naive Bayes

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LOAD DATA
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

data("HouseVotes84", package = "mlbench")
df <- as_tibble(HouseVotes84)

# Check structure
str(df)
# 435 obs. of 17 variables
# Class: democrat, republican
# V1-V16: voting patterns (y, n, NA)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EXPLORE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

table(df$Class)
# democrat republican
#      267        168

# Check missing values
map_dbl(df, ~sum(is.na(.)))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SPLIT DATA
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

set.seed(42)
idx <- createDataPartition(df$Class, p = 0.8, list = FALSE)
train <- df[idx, ]
test <- df[-idx, ]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TRAIN NAIVE BAYES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Method 1: Using caret
ctrl <- trainControl(method = "cv", number = 5)
nb_model <- train(
  Class ~ .,
  data = train,
  method = "naive_bayes",
  trControl = ctrl
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PREDICT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

predictions <- predict(nb_model, test)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EVALUATE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

cm <- confusionMatrix(predictions, test$Class)
print(cm)
cat("Accuracy:", cm$overall['Accuracy'], "\n")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PROBABILITIES (Interesting!)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Get probability for each class
prob <- predict(nb_model, test, type = "prob")
head(prob)
#   democrat republican
# 1    0.95       0.05
# 2    0.02       0.98
# ...

# ✅ Advantage of Naive Bayes: ให้ probability!
#    Can use probability as confidence score
```

---

## 5.2 Linear Discriminant Analysis (LDA)

### Example: Iris Dataset

```r
library(MASS)  # For LDA/QDA
library(caret)
library(tidyverse)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LOAD DATA
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

data(iris)
df <- as_tibble(iris)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EXPLORE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

str(df)
table(df$Species)
# setosa versicolor virginica
#     50         50        50

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SPLIT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

set.seed(42)
idx <- createDataPartition(df$Species, p = 0.8, list = FALSE)
train <- df[idx, ]
test <- df[-idx, ]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TRAIN LDA
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Using caret
lda_model <- train(
  Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width,
  data = train,
  method = "lda",
  trControl = trainControl(method = "cv", number = 5)
)

# Or using MASS directly
lda_model_direct <- lda(
  Species ~ .,
  data = train
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# INSPECT LDA RESULTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# View LDA coefficients
lda_model_direct$scaling
#                      LD1        LD2
# Sepal.Length   0.8293776   0.02410215
# Sepal.Width   -1.5344731  -2.16452123
# Petal.Length   2.2012117  -0.93192121
# Petal.Width    2.8104603   2.83918785

# Interpretation:
# - LD1 (Linear Discriminant 1): ต้อง focus บน Petal.Width (2.81)
# - LD2 (Linear Discriminant 2): Sepal.Width (-2.16) important

# Visualize discriminant functions
plot_data <- data.frame(
  lda_model_direct$scaling %*% t(train[, -5] %>% scale()),
  Species = train$Species
) %>% t() %>% as_tibble()

ggplot(plot_data, aes(x = LD1, y = LD2, color = Species)) +
  geom_point(size = 3) +
  theme_bw()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PREDICT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

pred <- predict(lda_model, test)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EVALUATE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

cm <- confusionMatrix(pred, test$Species)
print(cm)
cat("Accuracy:", cm$overall['Accuracy'], "\n")
```

---

## 5.3 Quadratic Discriminant Analysis (QDA)

```r
library(MASS)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TRAIN QDA (Same as LDA, just change method)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

qda_model <- train(
  Species ~ .,
  data = train,
  method = "qda",
  trControl = trainControl(method = "cv", number = 5)
)

# Or using MASS directly
qda_model_direct <- qda(
  Species ~ .,
  data = train
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PREDICT & EVALUATE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

pred_qda <- predict(qda_model, test)
cm_qda <- confusionMatrix(pred_qda, test$Species)
print(cm_qda)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# COMPARE LDA vs QDA
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

cat("LDA Accuracy:", cm$overall['Accuracy'], "\n")
cat("QDA Accuracy:", cm_qda$overall['Accuracy'], "\n")
# Usually similar on iris, QDA might be slightly better or worse
```

---

## 5.4 Complete Comparison: All 3 Algorithms

```r
library(tidyverse)
library(caret)
library(naivebayes)
library(MASS)

set.seed(42)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PREPARE DATA
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

data(iris)
df <- as_tibble(iris)

idx <- createDataPartition(df$Species, p = 0.8, list = FALSE)
train <- df[idx, ]
test <- df[-idx, ]

ctrl <- trainControl(method = "cv", number = 5)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TRAIN ALL THREE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

nb_model <- train(Species ~ ., data = train, 
                  method = "naive_bayes", trControl = ctrl)

lda_model <- train(Species ~ ., data = train, 
                   method = "lda", trControl = ctrl)

qda_model <- train(Species ~ ., data = train, 
                   method = "qda", trControl = ctrl)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PREDICT ON TEST
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

pred_nb <- predict(nb_model, test)
pred_lda <- predict(lda_model, test)
pred_qda <- predict(qda_model, test)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# COMPARE RESULTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

cm_nb <- confusionMatrix(pred_nb, test$Species)
cm_lda <- confusionMatrix(pred_lda, test$Species)
cm_qda <- confusionMatrix(pred_qda, test$Species)

results <- tibble(
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

print(results)
# Algorithm       Accuracy Kappa
# Naive Bayes        0.95  0.93
# LDA                0.97  0.95
# QDA                0.95  0.93

ggplot(results, aes(x = Algorithm, y = Accuracy, fill = Algorithm)) +
  geom_col() +
  ylim(0, 1) +
  theme_bw()
```

---

# 🟣 Part 6: สรุปและเปรียบเทียบ

## 6.1 Algorithm Selection Guide

```
Your data:
├─ High-dimensional (>100 features)?
│  └─ YES → Naive Bayes (handles well)
│
├─ Normal distribution?
│  ├─ YES → LDA (faster) or QDA (flexible)
│  └─ NO  → Naive Bayes
│
├─ Large dataset (>10,000)?
│  └─ YES → Naive Bayes (fewer parameters to estimate)
│
└─ Need interpretability?
   └─ YES → Naive Bayes > LDA > QDA
```

---

## 6.2 Advantages & Disadvantages

### Naive Bayes
```
✅ Pros:
- Very fast
- Handles high dimensions
- Works with categorical & continuous
- Good for text classification
- Gives probabilities
- Fewer parameters to learn

❌ Cons:
- Independence assumption often wrong
- Can underperform if assumptions badly violated
- Less interpretable than LDA
```

### LDA
```
✅ Pros:
- Straight decision boundary (simple)
- Dimension reduction
- Interpretable coefficients
- Medium speed

❌ Cons:
- Assumes normal distribution
- Assumes equal covariance (big assumption!)
- Must be continuous variables
```

### QDA
```
✅ Pros:
- Flexible curved boundaries
- No equal covariance assumption
- Better when classes have different shapes
- Dimension reduction

❌ Cons:
- More parameters (needs more data)
- Slower
- More likely to overfit
- Less interpretable
```

---

## 6.3 Key Differences

| Aspect | Naive Bayes | LDA | QDA |
|--------|------------|-----|-----|
| **Covariance** | Doesn't assume | Equal per class | Different per class |
| **Boundary** | Any shape | Straight line | Quadratic curve |
| **Data Type** | Flex | Continuous | Continuous |
| **Parameters** | Few | Medium | Many |
| **Overfitting Risk** | Low | Low-Medium | Medium-High |
| **Speed** | Fastest | Fast | Slower |

---

# 🎓 Learning Path

```
Week 1:
  □ Understand Bayes' Theorem (disease example)
  □ Learn Naive Bayes basics
  □ Code Naive Bayes on HouseVotes84

Week 2:
  □ Understand LDA concept
  □ Learn about discriminant functions
  □ Code LDA on Iris dataset

Week 3:
  □ Understand QDA differences
  □ Learn about covariance matrices
  □ Code QDA

Week 4:
  □ Compare all 3 on same dataset
  □ Choose based on data characteristics
  □ Practice with your own data
```

---

**ตอนนี้คุณพร้อมเขียน code สำหรับ 3 algorithms นี้แล้ว! 🚀**
