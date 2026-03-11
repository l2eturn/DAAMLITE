# 🗺️ DECISION TREE: เลือกการทดสอบที่เหมาะสม

---

## 🎯 ขั้นตอนแรก: ชนิดข้อมูล (Data Type)

```
ข้อมูลของเรา คือ... ?

┌─────────────────────────────────────────────────────┐
│                                                     │
│  1️⃣  ตัวเลข ต่อเนื่อง (Continuous, Interval/Ratio)  │
│      เช่น: ส่วนสูง, น้ำหนัก, ความเข้มข้น              │
│      → ไปขั้น 2                                      │
│                                                     │
│  2️⃣  ประเภท (Categorical, Nominal)                  │
│      เช่น: เพศ, สี, ประเภท                          │
│      → ไปยัง NOMINAL TESTS                          │
│                                                     │
│  3️⃣  อันดับ (Ordinal, Ranking)                     │
│      เช่น: Likert (1-5), ความพึงพอใจ               │
│      → อาจเป็น Nonparametric หรือ Parametric       │
│      → ตรวจ Normality ก่อน                          │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

## 📊 ขั้นตอนที่ 2: ตรวจสอบ Normality (สำหรับ Continuous Data)

### 🧪 วิธีตรวจสอบ

```r
# 1. Visual Check
ggplot(data, aes(x = variable)) + 
  geom_histogram(bins = 10, fill = "lightblue")
# → ต้องเป็นรูประฆัง (bell curve)

# 2. Normal Q-Q Plot
ggqqplot(data$variable)
# → ต้องเป็นเส้นตรงหากไม่มี deviation

# 3. Statistical Test
shapiro.test(data$variable)
# → p > 0.05 = ปกติ ✅
# → p < 0.05 = ไม่ปกติ ❌
```

### 💡 การตีความ

```
Shapiro-Wilk Test Result:

┌─────────────────────────┐
│  p-value > 0.05         │
│  → ข้อมูลปกติ ✅         │
│  → ใช้ PARAMETRIC TESTS  │
└─────────────────────────┘

┌─────────────────────────┐
│  p-value < 0.05         │
│  → ข้อมูลไม่ปกติ ❌     │
│  → ใช้ NONPARAMETRIC     │
└─────────────────────────┘
```

---

## 🔀 DECISION FLOWCHART

### ✅ ถ้า DATA PARAMETRIC (Normal + Continuous)

```
                    PARAMETRIC DATA
                         │
                    ┌────┴────┐
                    │          │
              ▼ 1 GROUP    ▼ 2+ GROUPS
                    │          │
            ONE-SAMPLE     ┌────┴─────┐
            T-TEST         │           │
                      ▼ 2 GROUPS  ▼ 3+ GROUPS
                      │           │
                  T-TEST      ANOVA
                  (unpaired)   (One-way)
                      │           │
                      │      ┌────┴────────┐
                      │      │             │
                      │   ▼ With 1 Factor │ 2+ Factors
                      │      │             │
                      │   One-way     Two-way ANOVA
                      │   ANOVA       (or higher)
                      │
                   Post-hoc?
                  (if p < 0.05)
                      │
                   Tukey HSD
                    t-tests
```

### ❌ ถ้า DATA NONPARAMETRIC (Not Normal, Ordinal, Outliers)

```
              NONPARAMETRIC DATA
                     │
              ┌──────┴──────┐
              │             │
         ▼ 1 GROUP    ▼ 2+ GROUPS
             │             │
        ┌────┴────┐    ┌────┴──────┐
        │          │    │           │
    SIGN TEST  WILCOXON │ 2 GROUPS  3+ GROUPS
    (Ordinal)  (Paired)  │          │
                     │    │          │
                MANN-WHITNEY  KRUSKAL-WALLIS
                (Unpaired)
                     │
                  Post-hoc?
                (if p < 0.05)
                     │
                  DUNN TEST
```

### 🎲 ถ้า DATA CATEGORICAL (Nominal)

```
           CATEGORICAL DATA
                  │
          ┌───────┴─────┐
          │              │
    ▼ 1 VARIABLE   ▼ 2+ VARIABLES
          │              │
    ┌────┴────┐      ┌────┴────┐
    │          │      │         │
GOODNESS  SPECIFIC  PAIRED   UNPAIRED
OF-FIT    PROBS   (Before/  (Association)
TEST               After)      │
  │                   │        │
Chi-Square     McNeMAR TEST   Chi-Square
G-test                        TEST
```

---

# 📚 DETAILED DECISION TREES

## TREE 1: ข้อมูล Parametric - ทั้ง 1, 2, 3+ groups

```
═══════════════════════════════════════════════════════
              PARAMETRIC TESTS TREE
═══════════════════════════════════════════════════════

START → "กี่ groups?"

    ▼ 1 GROUP
    └─→ "Compare to default value?"
        ├─ YES → ONE-SAMPLE T-TEST
        │        H0: μ = μ₀
        │        Code: t.test(x, mu = value)
        │
        └─ NO → Can't test (need reference)

    ▼ 2 GROUPS
    └─→ "Paired or Independent?"
        ├─ PAIRED
        │  └─→ PAIRED T-TEST
        │      H0: μ₁ = μ₂ (within pairs)
        │      Code: t.test(x1, x2, paired = TRUE)
        │
        └─ INDEPENDENT
           └─→ TWO-SAMPLE T-TEST
               H0: μ₁ = μ₂
               Code: t.test(y ~ x, data = df)
               Note: Welch's t (default) ≠ need homogeneity

    ▼ 3+ GROUPS (ANOVA)
    └─→ "How many factors?"
        ├─ 1 FACTOR → ONE-WAY ANOVA
        │             H0: μ₁ = μ₂ = μ₃ = ...
        │             Code: aov(y ~ x)
        │             Post-hoc: TukeyHSD()
        │
        └─ 2 FACTORS → TWO-WAY ANOVA
                       H0: Main effects + Interaction
                       Code: aov(y ~ x1 * x2)
                       Post-hoc: TukeyHSD()
```

---

## TREE 2: ข้อมูล Nonparametric - ทั้ง 1, 2, 3+ groups

```
═══════════════════════════════════════════════════════
            NONPARAMETRIC TESTS TREE
═══════════════════════════════════════════════════════

START → "กี่ groups?"

    ▼ 1 GROUP
    └─→ SIGN TEST
        H0: Median = μ₀
        Code: SIGN.test(x, m = value)
        ✅ ทำได้กับ Ordinal data

    ▼ 2 GROUPS
    └─→ "Paired or Independent?"
        ├─ PAIRED
        │  └─→ WILCOXON SIGNED-RANK TEST
        │      H0: Median of differences = 0
        │      Code: wilcox.test(x1, x2, paired = TRUE)
        │      ✅ ทำได้กับ Ordinal + Interval
        │
        └─ INDEPENDENT
           └─→ MANN-WHITNEY U TEST
               H0: Distributions are identical
                   (or stochastic equality)
               Code: wilcox.test(y ~ x)
               ✅ ทำได้กับ Ordinal + Interval
               Note: Don't assume testing medians

    ▼ 3+ GROUPS (One-way)
    └─→ "Paired or Independent?"
        ├─ PAIRED/REPEATED
        │  └─→ FRIEDMAN TEST
        │      H0: No difference across repeated measures
        │      Code: friedman.test(y ~ x | block)
        │
        └─ INDEPENDENT
           └─→ KRUSKAL-WALLIS TEST
               H0: Distributions are identical
               Code: kruskal.test(y ~ x)
               Post-hoc: dunnTest() or pairwise.wilcox.test()
               ✅ ทำได้กับ Ordinal + Interval
```

---

## TREE 3: ข้อมูล Categorical (Nominal)

```
═══════════════════════════════════════════════════════
               NOMINAL TESTS TREE
═══════════════════════════════════════════════════════

START → "ทดสอบอะไร?"

    ▼ 1 VARIABLE
    (Compare to expected proportions)
    └─→ GOODNESS-OF-FIT TEST
        
        ├─ EQUAL PROPORTIONS?
        │  └─→ H0: p₁ = p₂ = p₃ = ...
        │      Code: chisq.test(c(76, 48, 26))
        │
        └─ SPECIFIC PROPORTIONS?
           └─→ H0: p₁ = 0.5, p₂ = 0.33, p₃ = 0.17
               Code: chisq.test(counts, p = c(0.5, 0.33, 0.17))

    ▼ 2+ VARIABLES
    (Test association)
    └─→ "Paired data (Before-After)?"
        ├─ YES (PAIRED)
        │  └─→ McNeMAR TEST
        │      H0: Contingency table is symmetric
        │      Code: mcnemar.test(table)
        │      ✅ 2x2 table
        │
        │  └─→ McNeMAR-BOWKER TEST
        │      ✅ Larger tables
        │
        └─ NO (INDEPENDENT)
           └─→ CHI-SQUARE TEST OF ASSOCIATION
               H0: Variables are independent
               Code: chisq.test(table)
               Post-hoc: pairwiseNominalIndependence()
               
               ⚠️  Watch out:
               - Low cell counts?
                 → Use Fisher's exact (2x2)
                 → Use Monte Carlo
```

---

# 🚦 QUICK DECISION GUIDE

## ☑️ "ตัดสินใจ 30 วินาที"

```
Q1: ข้อมูลของฉัน คือ... ?
├─ Categorical (เพศ, สี, ประเภท) → NOMINAL TESTS
├─ Ordinal (Likert 1-5, rankings) → Check Q2
└─ Continuous (ส่วนสูง, น้ำหนัก) → Check Q2

Q2: ข้อมูล Normal Distribution? (Shapiro-Wilk p > 0.05)
├─ YES ✅ → PARAMETRIC TESTS
└─ NO ❌ → NONPARAMETRIC TESTS

Q3: กี่ groups?
├─ 1 group → One-sample test
├─ 2 groups → Two-sample test
│  └─ Paired? → Paired test
│  └─ Independent? → Independent test
└─ 3+ groups → Multi-group test
   └─ 1 factor → One-way
   └─ 2+ factors → Multi-way
```

---

# 📋 QUICK LOOKUP TABLE

```
┌──────────┬──────────┬──────────┬──────────┐
│ SITUATION│ Parametric│Nonparam │ Nominal │
├──────────┼──────────┼──────────┼──────────┤
│1 group   │ 1-sample │ Sign test│ G-o-F  │
│          │ t-test   │          │ test   │
├──────────┼──────────┼──────────┼──────────┤
│2 groups  │ 2-sample │Mann-Whit│ Chi-sq │
│ unpaired │ t-test   │ U test   │ test   │
├──────────┼──────────┼──────────┼──────────┤
│2 groups  │ Paired   │Wilcoxon │ McNemar│
│ paired   │ t-test   │S-R test │ test   │
├──────────┼──────────┼──────────┼──────────┤
│3+ groups │ One-way  │Kruskal- │ Chi-sq │
│ unpaired │ ANOVA    │Wallis   │ test   │
├──────────┼──────────┼──────────┼──────────┤
│3+ groups │ Two-way  │Aligned  │ Chi-sq │
│ 2+ factors│ ANOVA   │ ranks   │ test   │
└──────────┴──────────┴──────────┴──────────┘
```

---

# ⚠️ COMMON MISTAKES & FIXES

## ❌ ผิด: ใช้ Parametric กับข้อมูลไม่ปกติ

```r
# ❌ WRONG
MyData <- read_xlsx("data.xlsx")
t.test(MyData$Variable ~ MyData$Group)  # ไม่เช็ก Normality

# ✅ RIGHT
shapiro.test(MyData$Variable)  # ตรวจก่อน
if (p > 0.05) {
  t.test(MyData$Variable ~ MyData$Group)  # ปกติ → ใช้ t-test
} else {
  wilcox.test(MyData$Variable ~ MyData$Group)  # ไม่ปกติ → ใช้ U-test
}
```

---

## ❌ ผิด: ใช้ Mean กับ Categorical Data

```r
# ❌ WRONG
mean(Gender)  # ไม่มีความหมาย!

# ✅ RIGHT
table(Gender)  # นับจำนวน
prop.table(table(Gender))  # สัดส่วน
```

---

## ❌ ผิด: ลืม Post-hoc Test สำหรับ 3+ groups

```r
# ❌ WRONG
aov_result <- aov(y ~ x)
summary(aov_result)
# p < 0.05 → "There are differences"
# แต่ไม่รู้ว่า group ไหนต่างกัน

# ✅ RIGHT
aov_result <- aov(y ~ x)
summary(aov_result)
if (p < 0.05) {
  TukeyHSD(aov_result)  # ดูว่า group ไหนต่างกัน
}
```

---

## ❌ ผิด: ใช้ Chi-square กับ Low Cell Counts

```r
# ❌ WRONG (cell count = 2)
chisq.test(matrix(c(22, 2, 10, 5), nrow = 2))
# Warning: Chi-squared approximation may be incorrect

# ✅ RIGHT
fisher.test(matrix(c(22, 2, 10, 5), nrow = 2))  # Exact test
# or
chisq.test(..., simulate.p.value = TRUE)  # Monte Carlo
```

---

# 🎓 STEP-BY-STEP WORKFLOW

## ✅ ขั้นตอนที่ 1: สำรวจข้อมูล

```r
# Load
data <- read_xlsx("file.xlsx")

# Check structure
str(data)
head(data)
summary(data)

# Check for missing values
sum(is.na(data))

# Check sample size
nrow(data)
```

---

## ✅ ขั้นตอนที่ 2: ระบุตัวแปร

```
ตัวแปรอิสระ (X):
- Categorical? → Factor
- Continuous? → Numeric

ตัวแปรตาม (Y):
- Categorical? → Nominal test
- Ordinal? → Check normality first
- Continuous? → Check normality
```

---

## ✅ ขั้นตอนที่ 3: Visualize

```r
# Categorical Y
ggplot(data, aes(x = X)) + geom_bar()

# Continuous Y
ggplot(data, aes(x = X, y = Y)) + geom_boxplot()
ggplot(data, aes(x = Y)) + geom_histogram()
```

---

## ✅ ขั้นตอนที่ 4: Check Assumptions

```r
# Normality (Continuous Y)
shapiro.test(data$Y)

# Homogeneity (Multiple groups)
leveneTest(Y ~ X, data = data)

# Paired data?
table(data$ID, data$Time)  # Should be 1 per combo
```

---

## ✅ ขั้นตอนที่ 5: Choose & Run Test

```r
# Based on:
# - Y data type (categorical/continuous)
# - Y distribution (normal/not normal)
# - Sample design (paired/independent)
# - Group count (1/2/3+)

result <- appropriate_test(...)
```

---

## ✅ ขั้นตอนที่ 6: Interpret Results

```r
# Look at:
# - Test statistic (t, χ², H, U, etc.)
# - p-value (< 0.05 = significant)
# - Effect size (how big is the difference?)
# - Confidence interval

result
```

---

## ✅ ขั้นตอนที่ 7: Post-hoc (if needed)

```r
# If p < 0.05 and 3+ groups:
# - Parametric → TukeyHSD()
# - Nonparametric → dunnTest()
# - Nominal → pairwiseNominalIndependence()
```

---

## ✅ ขั้นตอนที่ 8: Report

```
"Mean height was significantly higher in males (M = 175, SD = 5)
compared to females (M = 165, SD = 4), t(98) = 2.45, p = 0.016."
```

---

# 🛠️ DECISION TREE VISUAL SUMMARY

```
                    START
                      │
                ┌─────┴─────┐
                │             │
           WHAT DATA?      Categorical?
                │         Yes│  │No
        ┌───────┴────────┐   │  │
        │                │   │  │
    Continuous?      Ordinal?│ │
        │                │   │  │
       Yes NO          Yes NO │  │
        │  │            │  │  │  │
        │  │            │  │  └──┴────────┐
        │  │            │  │              │
        │  │         Check Normality  Check Normality
        │  │            │  │              │
        │  │        P>0.05? NO           CATEGORICAL
        │  │          Yes│ │              │
        │  │            │ │          NOMINAL TESTS
        │  │            │ │
     PARAM NONPARAM   PARAM NONPARAM
        │  │            │ │
        ▼  ▼            ▼ ▼
     [Use Para-    [Use Para-
      metric       metric or
      tests]       Nonparam]
                   [based on
                    data]
```

---

# 📞 QUICK DECISION PHONE TREE

```
Press 1: Data is CATEGORICAL
↓
Choose test:
  1. Goodness-of-fit
  2. Association (Chi-square)
  3. Paired (McNemar)

Press 2: Data is CONTINUOUS
↓
Question: "Is it normally distributed?"
(Check: shapiro.test() or histogram)
↓
Press 1: YES, normally distributed
↓
Question: "How many groups?"
  1. One group → One-sample t
  2. Two groups → Two-sample t
  3. Three+ → ANOVA

Press 2: NO, not normally distributed
↓
Question: "How many groups?"
  1. One group → Sign test
  2. Two groups → Mann-Whitney U
  3. Three+ → Kruskal-Wallis

Press 3: Data is ORDINAL (Likert, rankings)
↓
See "CONTINUOUS" → "NO" pathway
(Treat as Nonparametric)
```

---

# 🎯 FINAL TIPS

## ✨ "Golden Rules" ที่ต้องจำ

1. **Always visualize first**
   - กราฟบอกความจริงได้ดีกว่า summary statistics

2. **Check assumptions before testing**
   - Normality, homogeneity, independence, etc.

3. **When in doubt, use Nonparametric**
   - ปลอดภัยกว่า (แม้จะมี power ต่ำกว่า)

4. **Report both p-value and effect size**
   - p-value บอก "มีจริง" หรือไม่
   - Effect size บอก "สำคัญไหม"

5. **Post-hoc test for 3+ groups always**
   - ต้องดูว่า group ไหนต่างกัน

6. **Be careful with sample size**
   - n < 5 per group → ทำ test ได้ยาก
   - n > 100 → ถึงผลต่างเล็กน้อยก็ significant

7. **Don't p-hack**
   - ห้องห้ามลองหลาย test จนกว่า p < 0.05
   - ต้อง pre-specify hypothesis ก่อน

---

**ตอนนี้คุณพร้อมแล้ว!** 🚀
