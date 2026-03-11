# 🧪 Practical Guide: R Code สำหรับการทดสอบทางสถิติ
## (Parametric, Nonparametric, Nominal)

---

## 📦 ติดตั้ง Packages ก่อน

```r
# ติดตั้งแบบครั้งเดียว (ถ้ายังไม่ได้ติดตั้ง)
install.packages("readxl")
install.packages("tidyverse")
install.packages("ggplot2")
install.packages("ggpubr")
install.packages("rcompanion")
install.packages("FSA")
install.packages("BSDA")
install.packages("DescTools")

# Load packages ทุกครั้งที่ใช้
library(readxl)
library(tidyverse)
library(ggplot2)
library(ggpubr)
library(rcompanion)
library(FSA)
library(BSDA)
library(DescTools)
```

---

# PART A: PARAMETRIC TESTS 📊

## 1. ONE-SAMPLE T-TEST

**คำถาม:** คะแนน sodium เฉลี่ยเท่ากับ 1500 mg หรือไม่?

### Step 1: Load Data
```r
# สมมติว่าข้อมูลใน Excel
MyData <- read_xlsx("parametric_data.xlsx", sheet = "Sheet1", range = "A1:C21")

# Check data
str(MyData)
head(MyData)
summary(MyData)

# ดูสถิติพื้นฐาน
mean(MyData$Sodium)   # ค่าเฉลี่ย
sd(MyData$Sodium)     # ส่วนเบี่ยงเบนมาตรฐาน
```

### Step 2: Check Normality

```r
# 📊 Histogram
ggplot(MyData, aes(x = Sodium)) + 
  geom_histogram(bins = 6, fill = "lightblue") + 
  ggtitle("Distribution of Sodium Intake (mg)") +
  theme_bw()

# 📈 Normal Q-Q Plot
ggqqplot(MyData$Sodium)

# 🧪 Shapiro-Wilk Test
shapiro.test(MyData$Sodium)
# Output: 
#   W = 0.97, p-value = 0.76
#   → p > 0.05 → ข้อมูลปกติ ✅
```

### Step 3: Run t-test

```r
# One-sample t-test
# H0: mu = 1500
# H1: mu ≠ 1500

result <- t.test(MyData$Sodium, 
                 mu = 1500,
                 alternative = "two.sided",
                 conf.level = 0.95)

result
# Output:
#   t = -0.82, df = 20, p-value = 0.42
#   95% CI: [1320, 1450]
#   → p > 0.05 → Fail to reject H0
#   → ไม่มีหลักฐานว่า mean ≠ 1500
```

### Step 4: Report Results

```
✅ RESULT:
นักเรียน (n=21) มีค่าเฉลี่ยการกินเกลือแค่ 1385 mg 
ไม่แตกต่างจากค่าแนะนำ 1500 mg อย่างมีนัยสำคัญ 
(t(20) = -0.82, p = 0.42)
```

---

## 2. TWO-SAMPLE T-TEST

**คำถาม:** เด็กชายและเด็กหญิงมีส่วนสูงต่างกันหรือไม่?

### Step 1: Load & Explore

```r
# สมมติข้อมูลมี: ส่วนสูง, เพศ
MyData2 <- read_xlsx("parametric_data.xlsx", sheet = "Sheet2")

# Check by group
MyData2 %>%
  group_by(Gender) %>%
  summarise(
    n = n(),
    mean_height = mean(Height),
    sd_height = sd(Height),
    median_height = median(Height)
  )
```

### Step 2: Visualize

```r
# Box plot
ggplot(MyData2, aes(x = Gender, y = Height, fill = Gender)) +
  geom_boxplot(alpha = 0.5) +
  geom_jitter(width = 0.2) +  # เพิ่ม scatter plot
  theme_bw()
```

### Step 3: Check Assumptions

```r
# 1. Normality ทั้ง 2 กลุ่ม
shapiro.test(MyData2$Height[MyData2$Gender == "Male"])
shapiro.test(MyData2$Height[MyData2$Gender == "Female"])

# 2. Homogeneity of Variance (Levene's Test)
library(car)
leveneTest(Height ~ Gender, data = MyData2)
# H0: variance เท่ากัน
# p > 0.05 → variances เท่ากัน ✅
```

### Step 4: Run t-test

```r
# Two-sample t-test
# วิธีที่ 1: ใช้ formula
result_t <- t.test(Height ~ Gender, data = MyData2)

# วิธีที่ 2: ระบุค่า explicitly
male_heights <- MyData2$Height[MyData2$Gender == "Male"]
female_heights <- MyData2$Height[MyData2$Gender == "Female"]
result_t <- t.test(male_heights, female_heights)

result_t
# Output: t = 2.45, p-value = 0.018
# → p < 0.05 → Reject H0 ✅
# → ชายสูงกว่าหญิง
```

---

## 3. ONE-WAY ANOVA

**คำถาม:** นักเรียนของครูต่างคน (A, B, C) กินโซเดียมต่างกันหรือไม่?

### Step 1: Load & Check

```r
MyData4 <- read_xlsx("parametric_data.xlsx", sheet = "Sheet4")

# Convert to factor
MyData4$Instructor <- factor(MyData4$Instructor,
                             levels = unique(MyData4$Instructor))

# Summarize
MyData4 %>%
  group_by(Instructor) %>%
  summarise(
    n = n(),
    mean_sodium = mean(Sodium),
    sd_sodium = sd(Sodium)
  )

# Output:
#   Instructor n mean_sodium sd_sodium
#   A         10  1450       150
#   B          8  1300       180
#   C          6  1200       120
```

### Step 2: Visualize

```r
# Box plot
ggplot(MyData4, aes(x = Instructor, y = Sodium, fill = Instructor)) +
  geom_boxplot(alpha = 0.5) +
  theme(legend.position = "bottom")

# Plot means with confidence intervals
library(rcompanion)
Mean_conf <- groupwiseMean(Sodium ~ Instructor, 
                           data = MyData4,
                           conf = 0.95, digits = 3)

ggplot(Mean_conf, aes(x = Instructor, y = Mean)) +
  geom_errorbar(aes(ymin = Trad.lower, ymax = Trad.upper), 
                width = 0.2, size = 1) +
  geom_point(shape = 15, size = 4) +
  theme_bw() +
  ylab("Mean Sodium (mg)")
```

### Step 3: Check Assumptions

```r
# Run ANOVA first to get model
model_aov <- aov(Sodium ~ Instructor, data = MyData4)

# Check normality of residuals
shapiro.test(residuals(model_aov))
# p > 0.05 → ปกติ ✅

# Check homogeneity of variance
leveneTest(Sodium ~ Instructor, data = MyData4)
# p > 0.05 → variance เท่ากัน ✅
```

### Step 4: Run ANOVA

```r
# One-way ANOVA
result_anova <- aov(Sodium ~ Instructor, data = MyData4)

summary(result_anova)
# Output:
#             Df   Sum Sq  Mean Sq F value Pr(>F)
# Instructor   2   50000   25000   2.45    0.087
# Residuals   21  214000   10190
#
# → p = 0.087 > 0.05 → Fail to reject H0
# → ไม่มีหลักฐานว่าครูต่างคนให้ผลต่างกัน
```

### Step 5: Post-hoc Test (ถ้า p < 0.05)

```r
# Tukey HSD (Honestly Significant Difference)
TukeyHSD(result_anova)
# Output:
#   diff        lwr      upr     p adj
# B-A  -150  -400.5   100.5   0.32
# C-A  -250  -550.5    50.5   0.12
# C-B  -100  -430.5   230.5   0.75
#
# → ไม่มีคู่ใดต่างกันอย่างมีนัยสำคัญ
```

### Step 6: Check Residuals

```r
# Plot diagnostics
plot(result_anova)
# ดู 4 graphs:
# 1. Residuals vs Fitted → ต้องเป็นเส้นตรง
# 2. Q-Q plot → ต้องเป็นเส้นตรง (normality)
# 3. Scale-Location → homogeneity
# 4. Residuals vs Leverage → ไม่มี outliers
```

---

## 4. TWO-WAY ANOVA

**คำถาม:** ครู AND วิตามิน มีผลต่อการกินโซเดียมหรือไม่?

### Code Example

```r
MyData5 <- read_xlsx("parametric_data.xlsx", sheet = "Sheet5")

# Convert to factors
MyData5$Instructor <- factor(MyData5$Instructor,
                             levels = unique(MyData5$Instructor))
MyData5$Supplement <- factor(MyData5$Supplement,
                             levels = unique(MyData5$Supplement))

# Two-way ANOVA (without interaction)
model_2way <- aov(Sodium ~ Instructor + Supplement, data = MyData5)
summary(model_2way)

# Two-way ANOVA (with interaction)
model_2way_int <- aov(Sodium ~ Instructor * Supplement, data = MyData5)
summary(model_2way_int)

# Post-hoc tests
TukeyHSD(model_2way)
```

---

---

# PART B: NONPARAMETRIC TESTS 🎯

## 1. SIGN TEST (One-sample, Ordinal)

**คำถาม:** คะแนน Likert เฉลี่ยต่างจาก 3 (neutral) หรือไม่?

### Code Example

```r
LikertData1 <- read_xlsx("nonparametric_data.xlsx", sheet = "Sheet1")

# Convert to factor
LikertData1$Likert_f <- factor(LikertData1$Likert, 
                               levels = 1:5, 
                               ordered = TRUE)

# Summary
summary(LikertData1)

# Cross tabulation
XT <- xtabs(~ Speaker + Likert_f, data = LikertData1)
prop.table(XT, margin = 1)

# SIGN TEST
# H0: median = 3
# H1: median > 3
library(BSDA)
SIGN.test(LikertData1$Likert,
          m = 3,
          alternative = "greater",
          conf.level = 0.95)

# Output:
#   Below Median = 8, Above Median = 32
#   p-value = 0.000093
#   → p < 0.05 → Reject H0 ✅
#   → นักเรียน เห็นด้วยมากกว่า neutral
```

---

## 2. MANN-WHITNEY U TEST (Two-sample, Unpaired)

**คำถาม:** Likert scores ต่างกันระหว่าง 2 ลำโพง?

### Code Example

```r
# สมมติมีข้อมูล Likert ของ Pooh vs Piglet
LikertData2 <- read_xlsx("nonparametric_data.xlsx", sheet = "Sheet2")

# Summarize by group
LikertData2 %>%
  group_by(Speaker) %>%
  summarise(
    n = n(),
    median = median(Likert),
    mean = mean(Likert),
    sd = sd(Likert)
  )

# Box plot
ggplot(LikertData2, aes(x = Speaker, y = Likert, fill = Speaker)) +
  geom_boxplot(alpha = 0.5) +
  geom_jitter(width = 0.2) +
  theme_bw()

# Mann-Whitney U Test
result_mw <- wilcox.test(Likert ~ Speaker, data = LikertData2)
result_mw

# Output:
#   W = 245, p-value = 0.023
#   → p < 0.05 → มีความแตกต่าง ✅
```

---

## 3. WILCOXON SIGNED-RANK TEST (Paired)

**คำถาม:** คะแนน Before → After มีการเปลี่ยนแปลงหรือไม่?

### Code Example

```r
LikertData3 <- read_xlsx("nonparametric_data.xlsx", sheet = "Sheet3")

# Sort by Time and Student
LikertData3 <- LikertData3[order(LikertData3$Time, LikertData3$Student), ]

# Extract paired data
Time_1 <- LikertData3$Likert[LikertData3$Time == 1]
Time_2 <- LikertData3$Likert[LikertData3$Time == 2]

# Calculate differences
Difference <- Time_2 - Time_1

# Wilcoxon Signed-Rank Test
# H0: median of differences = 0
# H1: median of differences ≠ 0
result_wilcox <- wilcox.test(Time_1, Time_2, paired = TRUE)
result_wilcox

# Output:
#   V = 120, p-value = 0.0032
#   → p < 0.05 → มีการเปลี่ยนแปลง ✅
```

---

## 4. KRUSKAL-WALLIS TEST (3+ groups)

**คำถาม:** Likert scores ต่างกันระหว่าง 3 ลำโพง (Pooh, Piglet, Tigger)?

### Code Example

```r
LikertData4 <- read_xlsx("nonparametric_data.xlsx", sheet = "Sheet4")

# Factor
LikertData4$Speaker <- factor(LikertData4$Speaker)
LikertData4$Likert_f <- factor(LikertData4$Likert, ordered = TRUE)

# Summarize
LikertData4 %>%
  group_by(Speaker) %>%
  summarise(
    n = n(),
    median = median(Likert),
    mean = mean(Likert),
    sd = sd(Likert)
  )

# Kruskal-Wallis Test
result_kw <- kruskal.test(Likert ~ Speaker, data = LikertData4)
result_kw

# Output:
#   H = 8.45, p-value = 0.015
#   → p < 0.05 → มีความแตกต่าง ✅

# Post-hoc: Dunn Test
library(FSA)
dunnTest(Likert ~ Speaker, data = LikertData4,
         method = "bonferroni")  # p-value adjustment

# Output:
#   Pooh-Piglet: p = 0.023 ✅ ต่างกัน
#   Pooh-Tigger: p = 0.18 ไม่ต่างกัน
#   Piglet-Tigger: p = 0.89 ไม่ต่างกัน
```

---

---

# PART C: NOMINAL (CATEGORICAL) TESTS 🎲

## 1. CHI-SQUARE GOODNESS-OF-FIT TEST

**คำถาม:** สีดอกไม้สัดส่วน 3:2:1 หรือไม่?

### Code Example

```r
# Data
tulip <- c(76, 48, 26)  # red, yellow, white
names(tulip) <- c("Red", "Yellow", "White")

# Expected proportions (3:2:1)
total <- sum(tulip)
expected_props <- c(3/6, 2/6, 1/6)
expected_counts <- expected_props * total

# Chi-square test
result_chi <- chisq.test(tulip, p = expected_props)
result_chi

# Output:
#   χ² = 11.03, df = 2, p-value = 0.004
#   → p < 0.05 → สีไม่เป็นสัดส่วน 3:2:1 ✅

# Visualize
barplot(tulip, main = "Observed Tulip Colors")
barplot(expected_counts, main = "Expected Tulip Colors")
```

---

## 2. CHI-SQUARE TEST OF ASSOCIATION (2 Variables)

**คำถาม:** เพศ AND งานบ้าน มีความเกี่ยวข้องกัน หรือไม่?

### Code Example

```r
housetasks <- read_xlsx("nominal_data.xlsx", sheet = "Sheet2")

# Convert to contingency table
housetasks_tbl <- as.table(as.matrix(housetasks[, -1]))
rownames(housetasks_tbl) <- housetasks$Task

housetasks_tbl
#              Father  Mother  Son  Daughter
# Laundry          2       8    4         2
# Kitchen          2       7    5         3
# Dishes           4       3    6         4
# Bedroom          5       2    1         2

# Chi-square test of association
# H0: งานบ้านไม่เกี่ยวข้องกับเพศ
# H1: งานบ้านเกี่ยวข้องกับเพศ
result_chi2 <- chisq.test(housetasks_tbl)
result_chi2

# Output:
#   χ² = 9.23, df = 9, p-value = 0.42
#   → p > 0.05 → Fail to reject H0
#   → ไม่มีหลักฐานความสัมพันธ์

# Post-hoc: Pairwise comparisons
library(rcompanion)
pairwiseNominalIndependence(housetasks_tbl,
                            chisq = TRUE,
                            method = "fdr")

# Mosaic plot
library(ggmosaic)
housetasks_long <- pivot_longer(housetasks, 
                                 names_to = "Who",
                                 values_to = "Freq",
                                 cols = -Task)
housetasks_case <- uncount(housetasks_long, Freq)

ggplot(housetasks_case) +
  geom_mosaic(aes(x = product(Task), fill = Who)) +
  theme_bw()
```

---

## 3. McNeMAR TEST (Paired Categorical)

**คำถาม:** ก่อน → หลังชมวิดีโอ เปลี่ยนใจหรือไม่?

### Code Example

```r
# Create data
set.seed(345)
before <- rbinom(100, size = 1, prob = 0.2)
after <- rbinom(100, size = 1, prob = 0.8)

before_f <- factor(before, levels = c(0, 1), 
                   labels = c("Negative", "Positive"))
after_f <- factor(after, levels = c(0, 1), 
                  labels = c("Negative", "Positive"))

law_support <- data.frame(Before = before_f, After = after_f)

# Contingency table
law_support_tbl <- xtabs(~ Before + After, law_support)
law_support_tbl
#         After
# Before   Negative Positive
#   Negative       16       64
#   Positive        4       16

# McNemar Test
# H0: ความคิดเห็นไม่เปลี่ยนแปลง
# H1: ความคิดเห็นเปลี่ยนแปลง
result_mcnemar <- mcnemar.test(law_support_tbl)
result_mcnemar

# Output:
#   McNemar's chi-squared = 48, p-value = 1.63e-12
#   → p < 0.05 → Reject H0 ✅
#   → ความคิดเห็นเปลี่ยนแปลงมาก

# Interpretation:
# 64 คน เปลี่ยนจาก Negative → Positive (เห็นด้วยเพิ่ม)
# 4 คน เปลี่ยนจาก Positive → Negative
# → วิดีโอมีประสิทธิ์ในการเปลี่ยนมุมมอง
```

---

---

# 📋 QUICK REFERENCE TABLE

| **สถานการณ์** | **Code** | **Function** |
|---|---|---|
| **Parametric - 1 sample** | `t.test(x, mu = value)` | One-sample t |
| **Parametric - 2 samples** | `t.test(y ~ x, data = df)` | Two-sample t |
| **Parametric - 3+ samples** | `aov(y ~ x, data = df)` | One-way ANOVA |
| **Parametric - 2 factors** | `aov(y ~ x1 * x2, data = df)` | Two-way ANOVA |
| **Nonparametric - 1 sample** | `SIGN.test(x, m = value)` | Sign test |
| **Nonparametric - 2 samples** | `wilcox.test(y ~ x, data = df)` | Mann-Whitney U |
| **Nonparametric - paired** | `wilcox.test(x1, x2, paired = TRUE)` | Wilcoxon |
| **Nonparametric - 3+ samples** | `kruskal.test(y ~ x, data = df)` | Kruskal-Wallis |
| **Nominal - 1 variable** | `chisq.test(counts, p = probs)` | Goodness-of-fit |
| **Nominal - 2 variables** | `chisq.test(table)` | Chi-square |
| **Nominal - paired** | `mcnemar.test(table)` | McNemar |

---

# 🚀 Full Workflow Example

```r
# ============================================
# COMPLETE ANALYSIS WORKFLOW
# ============================================

# 1. Load data
MyData <- read_xlsx("data.xlsx")

# 2. Explore
str(MyData)
summary(MyData)
head(MyData)

# 3. Visualize
ggplot(MyData, aes(x = Variable)) + geom_histogram()

# 4. Check assumptions
shapiro.test(MyData$Variable)  # Normality
leveneTest(Variable ~ Group, data = MyData)  # Homogeneity

# 5. Choose test
if (assumptions_met) {
  result <- t.test(Variable ~ Group, data = MyData)
} else {
  result <- wilcox.test(Variable ~ Group, data = MyData)
}

# 6. Check results
result

# 7. Post-hoc (if needed)
TukeyHSD(aov_result)  # or dunnTest()

# 8. Report
# "Mean height was significantly different (t(45) = 2.3, p = 0.02)"
```

---

**ตอนนี้พร้อมทั้ง code และอธิบาย!** 💪
