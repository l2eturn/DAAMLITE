# 📊 สอนพื้นฐาน: การทดสอบทางสถิติ (Parametric, Nonparametric, Nominal)

## 🎯 บทนำ: ทำไมต้องเรียนนี้?

ก่อนเลือกการทดสอบทางสถิติ ต้องเข้าใจว่าข้อมูลของเรา **ชนิดไหน** ข้อมูลแต่ละชนิดต้องใช้การทดสอบแบบต่างกัน

---

## 🔍 ขั้นตอนแรก: ระบุ **ชนิดของข้อมูล (Data Type)**

### 1️⃣ **ข้อมูลประเภท Parametric (Interval/Ratio)**

```
ข้อมูลที่วัดได้ด้วยตัวเลขต่อเนื่องและมีความหมาย
```

**ตัวอย่าง:**
- 🐟 ความยาวปลา (cm): 10.5, 12.3, 11.8, ...
- 📏 ส่วนสูงเด็ก (cm): 140, 145, 150, ...
- ⚖️ น้ำหนักพืช (kg): 2.3, 3.5, 2.8, ...
- 💧 ความเข้มข้นมลพิษในน้ำ (mg/L): 0.5, 0.7, 1.2, ...

**คุณสมบัติ:**
- ✅ ต่อเนื่อง (ไม่ใช่เพียงจำนวนเต็ม)
- ✅ มีค่า 0 ที่มีความหมาย (น้ำหนัก = 0 ≠ ไม่มีของ)
- ✅ ความห่างระหว่างตัวเลขสม่ำเสมอ (1-2 = 2-3)

---

### 2️⃣ **ข้อมูลประเภท Nonparametric (Ordinal + Interval/Ratio)**

```
ข้อมูลที่ไม่ปกติ หรือที่เป็นการจัดอันดับ
```

**ตัวอย่าง:**
- ⭐ คะแนน Likert: 1 = ไม่เห็นด้วย → 5 = เห็นด้วยมากที่สุด
- 📊 ความพึงพอใจ: ต่ำ, ปานกลาง, สูง, สูงมาก
- 🏆 อันดับคะแนนสอบ: อันดับ 1, 2, 3, ...
- 📈 ข้อมูลที่ไม่เป็นการแจกแจงปกติ (normal distribution)

**เมื่อไหร่ใช้ Nonparametric:**
- ข้อมูล interval/ratio แต่ **ไม่ปกติ** (ตรวจด้วย Shapiro-Wilk test)
- ข้อมูล **ordinal** (จัดอันดับ)
- มี **outliers** (ค่าผิดปกติ)

---

### 3️⃣ **ข้อมูลประเภท Nominal (Categorical)**

```
ข้อมูลที่เป็นหมวดหมู่/ประเภท (ไม่มีลำดับ)
```

**ตัวอย่าง:**
- 🎨 สีดอกไม้: แดง, เหลือง, ฟ้า
- 🚹🚺 เพศ: ชาย, หญิง
- 👨‍👩‍👧‍👦 สถานภาพครอบครัว: โสด, แต่งงาน, หย่า
- 🛫 สายการบิน: Thai Airways, Bangkok Air, Nok Air

**คุณสมบัติ:**
- ❌ **ไม่ต่อเนื่อง** (ไม่สามารถจัดอันดับได้อย่างมีความหมาย)
- ❌ ใช้ **นับจำนวน (count)** เท่านั้น
- ❌ ไม่สามารถคำนวณ mean หรือ SD ได้

---

## 📋 ตารางเปรียบเทียบ 3 ประเภท

| เกณฑ์ | **Parametric** | **Nonparametric** | **Nominal** |
|------|---|---|---|
| **ชนิดข้อมูล** | Interval/Ratio ต่อเนื่อง | Ordinal + Interval/Ratio ที่ไม่ปกติ | Categorical (ประเภท) |
| **การแจกแจง** | ✅ ต้องปกติ (Normal) | ❌ ไม่ต้อง | ❌ ไม่ต้อง |
| **วิธีวิเคราะห์** | ใช้ค่า (mean, SD) | ใช้อันดับ (ranks) | ใช้นับจำนวน (counts) |
| **ตัวอย่าง** | ส่วนสูง, น้ำหนัก | Likert scores, อันดับ | เพศ, สี, ประเภท |
| **ความ "พลัง"** | 💪 สูงกว่า | 💪💪 ต่ำกว่า Parametric | 💪💪 ต่ำสุด |
| **ความอ่อนไหว** | 😰 ไม่ทนต่อ outliers | 😊 ทนต่อ outliers | - |

---

---

# 🧪 ส่วนที่ 2: การทดสอบแต่ละประเภท

## A. PARAMETRIC TESTS 📊

**ข้อกำหนดเบื้องต้น (Assumptions):**

1. ✅ **Random Sampling**: ข้อมูลสุ่มจากประชากร
2. ✅ **Normal Distribution**: ข้อมูลแจกแจงแบบปกติ
   - ตรวจด้วย: **Histogram** หรือ **Normal Q-Q Plot**
   - ทดสอบด้วย: **Shapiro-Wilk test** (p > 0.05 = ปกติ)
3. ✅ **Homogeneity of Variance**: ความแปรปรวนของแต่ละกลุ่มเท่ากัน
4. ✅ **No Outliers**: ไม่มีค่าผิดปกติ

---

### 🧪 การทดสอบแต่ละประเภท:

#### **1. One-sample t-test** (ทดสอบ 1 กลุ่ม)

**คำถาม:** "คะแนนของนักเรียนแตกต่างจากค่า default (เช่น 75) หรือไม่?"

**สูตร:**
```
t = (x̄ - μ₀) / (SD / √n)
```

**ตัวอย่าง:**
- นักเรียนกินโซเดียม: ค่าเฉลี่ย = 1200 mg
- ทดสอบว่า ≠ 1500 mg (ค่าแนะนำ)
- t-test ได้ p-value

```R
# Code
t.test(MyData$Sodium, 
       mu = 1500,               # ค่า default
       alternative = "two.sided", # ≠
       conf.level = 0.95)
```

---

#### **2. Two-sample t-test** (เปรียบเทียบ 2 กลุ่ม)

**คำถาม:** "กลุ่ม A และ B มีค่าเฉลี่ยต่างกันหรือไม่?"

**ตัวอย่าง:**
- 👨 ชาย: ส่วนสูง ค่าเฉลี่ย = 175 cm
- 👩 หญิง: ส่วนสูง ค่าเฉลี่ย = 165 cm
- t-test ได้ p-value → เห็นด้วย/ต่างกัน?

```R
# Code
t.test(height ~ gender, data = MyData)
```

---

#### **3. One-way ANOVA** (เปรียบเทียบ 3+ กลุ่ม)

**คำถาม:** "ครูแบบต่างๆ (3 คน) ทำให้นักเรียนกินโซเดียมแตกต่างกันหรือไม่?"

**ตัวอย่าง:**
- ครู A: ค่าเฉลี่ย = 1200
- ครู B: ค่าเฉลี่ย = 1150
- ครู C: ค่าเฉลี่ย = 1180
- ANOVA → เห็นด้วยหรือไม่?

```R
# Code
result <- aov(Sodium ~ Instructor, data = MyData)
summary(result)

# Post-hoc (ดูว่ากลุ่มไหนต่างกัน)
TukeyHSD(result)
```

---

#### **4. Two-way ANOVA** (เปรียบเทียบด้วย 2 ตัวแปร)

**คำถาม:** "ครู AND วิตามิน(supplement) ส่งผลต่อการกินโซเดียม?"

**ตัวอย่าง:**
- ครู × วิตามิน = 3 × 4 = 12 กลุ่ม
- ANOVA → ดูผล **ครู**, **วิตามิน**, และ **interaction**

```R
# Code
result <- aov(Sodium ~ Instructor + Supplement, data = MyData)
summary(result)
```

---

## B. NONPARAMETRIC TESTS 🎯

**เมื่อใช้:**
- ข้อมูล **Ordinal** (Likert, rankings)
- ข้อมูล **Interval/Ratio** แต่ **ไม่ปกติ**
- มี **outliers** หรือ **skewed**

---

### 🧪 การทดสอบแต่ละประเภท:

#### **1. Sign Test** (ทดสอบ 1 กลุ่ม Ordinal)

**คำถาม:** "คะแนน Likert แตกต่างจากค่า default (เช่น 3 = neutral) หรือไม่?"

```R
SIGN.test(MyData$Likert, 
          m = 3,                 # default = 3
          alternative = "greater") # > 3
```

---

#### **2. Mood's Median Test** (ทดสอบ 2+ กลุ่ม)

**คำถาม:** "เปรียบเทียบ median ของ 2+ กลุ่ม"

ใช้เมื่อ:
- ไม่มีการสันนิษฐานเกี่ยวกับ distribution
- ข้อมูล ordinal หรือ ordinal ที่ไม่ปกติ

```R
# Code: Mood's Median Test
```

---

#### **3. Mann-Whitney U Test** (เทียบ 2 กลุ่ม)

**คำถาม:** "2 กลุ่มมี distribution ต่างกันหรือไม่?"

**ตัวอย่าง:**
- กลุ่ม A: Likert = [1, 2, 3, 4, 5]
- กลุ่ม B: Likert = [2, 3, 4, 5, 5]
- U-test → เห็นด้วยหรือไม่?

```R
# Code
wilcox.test(Likert ~ Group, data = MyData)
```

---

#### **4. Wilcoxon Signed-Rank Test** (เทียบ 2 กลุ่มแบบ Paired)

**คำถาม:** "Before → After มีการเปลี่ยนแปลงหรือไม่?"

**ตัวอย่าง:**
- Before training: Likert = [1, 2, 3, 2, 4]
- After training:  Likert = [3, 4, 5, 4, 5]
- ทดสอบ Paired differences

```R
# Code
wilcox.test(Before, After, paired = TRUE)
```

---

#### **5. Kruskal-Wallis Test** (เทียบ 3+ กลุ่ม)

**คำถาม:** "Likert scores ต่างกันระหว่าง 3 ครู?"

```R
# Code
kruskal.test(Likert ~ Speaker, data = MyData)

# Post-hoc: Dunn test
dunnTest(Likert ~ Speaker, data = MyData)
```

---

## C. NOMINAL (CATEGORICAL) TESTS 🎲

**เมื่อใช้:**
- ข้อมูล **Categorical** (สี, เพศ, ประเภท)
- ใช้ **นับจำนวน (counts)** เท่านั้น
- ไม่สามารถใช้ mean/SD

---

### 🧪 การทดสอบแต่ละประเภท:

#### **1. Goodness-of-Fit Test (Chi-square, G-test)**

**คำถาม:** "สีดอกไม้สัดส่วน: สีแดง (3), สีเหลือง (2), สีฟ้า (1)?"

**ข้อมูล:**
- สีแดง: 76 ดอก
- สีเหลือง: 48 ดอก
- สีฟ้า: 26 ดอก
- ทดสอบว่า ตรงสัดส่วน 3:2:1 หรือไม่?

```R
# Chi-square test
tulip <- c(76, 48, 26)
chisq.test(tulip, p = c(3/6, 2/6, 1/6))

# G-test
GTest(tulip, p = c(3/6, 2/6, 1/6))
```

---

#### **2. Chi-square Test of Association** (ทดสอบความสัมพันธ์)

**คำถาม:** "เพศ AND งานบ้าน มีความเกี่ยวข้องกันหรือไม่?"

**ตัวอย่าง Contingency Table:**

|       | งานซักผ้า | งานล้างจาน | งานทำอาหาร |
|-------|:--------:|:--------:|:--------:|
| ชาย   |    2     |     8    |    4    |
| หญิง  |    7     |     3    |   11    |

```R
# Code
housetasks_tbl <- xtabs(~ Gender + Task, data = MyData)
chisq.test(housetasks_tbl)

# Pairwise comparisons
pairwiseNominalIndependence(housetasks_tbl, chisq = TRUE)
```

---

#### **3. McNemar Test** (Paired Nominal Data)

**คำถาม:** "ก่อนชมวิดีโอ → หลังชมวิดีโอ เปลี่ยนใจหรือไม่?"

**ตัวอย่าง:**

| Before \ After | เห็นด้วย | ไม่เห็นด้วย |
|---|:---:|:---:|
| **เห็นด้วย** | 2 | 0 |
| **ไม่เห็นด้วย** | 21 | 7 |

```R
# Code
mcnemar.test(law_support_tbl)
```

---

---

# 🛠️ ส่วนที่ 3: ขั้นตอนการทดสอบ Step-by-Step

## **ขั้นตอน 1️⃣: ระบุชนิดข้อมูล**

```
1. ข้อมูลเป็น Categorical (เพศ, สี) → NOMINAL
2. ข้อมูล Ordinal (Likert 1-5) → NONPARAMETRIC
3. ข้อมูล Interval/Ratio ต่อเนื่อง (ความยาว, น้ำหนัก) → เช็ค Normality ก่อน
```

---

## **ขั้นตอน 2️⃣: ตรวจสอบ Normality (ถ้าเป็น Interval/Ratio)**

```R
# 📊 Histogram
ggplot(MyData, aes(x = Sodium)) + 
  geom_histogram(bins = 6, fill = "lightblue")

# 📈 Normal Q-Q Plot
ggqqplot(MyData$Sodium)

# 🧪 Shapiro-Wilk Test
shapiro.test(MyData$Sodium)
# p > 0.05 → ปกติ ✅
# p < 0.05 → ไม่ปกติ ❌
```

---

## **ขั้นตอน 3️⃣: เลือกการทดสอบ**

```
ถ้าปกติ ✅         → Parametric (t-test, ANOVA)
ถ้าไม่ปกติ ❌      → Nonparametric (Mann-Whitney, Kruskal-Wallis)
ถ้า Categorical  → Nominal (Chi-square, McNemar)
```

---

## **ขั้นตอน 4️⃣: รัน Test และ Interpret**

```
p-value < 0.05 → มีความแตกต่าง ✅ (Reject H0)
p-value > 0.05 → ไม่มีความแตกต่าง ❌ (Fail to reject H0)
```

---

---

# 📚 ตัวอย่างจากไฟล์ของคุณ

## ตัวอย่าง 1: Parametric (One-sample t-test)

**สถานการณ์:** นักเรียน Brendon Small กินโซเดียม เปรียบเทียบ 1500 mg

```R
t.test(MyData$Sodium, 
       mu = 1500,
       alternative = "greater",  # > 1500?
       conf.level = 0.9)

# ผล: t = -0.82, p-value = 0.78
# → ไม่มีหลักฐานว่านักเรียนกินน้อยกว่า 1500 mg
```

---

## ตัวอย่าง 2: Nonparametric (Kruskal-Wallis)

**สถานการณ์:** คะแนน Likert แตกต่างกัน 3 ลำโพง (Pooh, Piglet, Tigger)?

```R
kruskal.test(Likert ~ Speaker, data = MyData)

# Post-hoc: Dunn test
dunnTest(Likert ~ Speaker, data = MyData)

# ดูว่าคู่ไหนต่างกัน
```

---

## ตัวอย่าง 3: Nominal (Chi-square)

**สถานการณ์:** สีดอกไม้สัดส่วน 3:2:1?

```R
tulip <- c(76, 48, 26)  # red, yellow, white

chisq.test(tulip, p = c(0.5, 0.333, 0.167))

# ผล: p-value = 0.0008
# → สีไม่เป็นสัดส่วน 3:2:1 ✅
```

---

---

# 💡 เคล็ดลับ & มีดที่พบบ่อย

## ❌ ผิดพลาดทั่วไป:

1. **ใช้ Parametric กับ Categorical Data** → ✗
   - ❌ แบบผิด: `t.test(gender ~ group)`
   - ✅ ถูก: `chisq.test(contingency_table)`

2. **ใช้ Mean กับ Nominal Data**
   - ❌ "ความเพศเฉลี่ย = 1.5" ← ไม่สมควร!
   - ✅ "46% ชาย, 54% หญิง"

3. **ไม่ตรวจ Assumptions** ก่อนใช้ Parametric
   - ✅ ต้องตรวจ Normality ก่อน

4. **Count data มี Low cell counts** ← ทำให้ Chi-square ไม่แม่นยำ
   - ✅ ใช้ Fisher's exact test หรือ Monte Carlo

---

## ✅ Best Practices:

1. **เสมอตรวจสอบข้อมูล** ก่อน:
   ```R
   str(MyData)
   summary(MyData)
   head(MyData)
   ```

2. **วาด Plot** เพื่อ visualize:
   ```R
   # Parametric
   ggplot(MyData, aes(x = Variable)) + geom_histogram()
   
   # Nominal
   ggplot(MyData, aes(x = Category)) + geom_bar()
   ```

3. **ตรวจ Assumptions**:
   ```R
   # Normality
   shapiro.test(MyData$Variable)
   
   # Homogeneity
   leveneTest(Variable ~ Group, data = MyData)
   ```

4. **ทำ Post-hoc** ถ้าต้องเปรียบเทียบเป็นจペア:
   ```R
   TukeyHSD(anova_result)  # Parametric
   dunnTest(...)           # Nonparametric
   ```

---

---

# 📊 Summary Table

| **สถานการณ์** | **Parametric** | **Nonparametric** | **Nominal** |
|---|---|---|---|
| **1 กลุ่ม** | One-sample t-test | Sign test | Goodness-of-fit |
| **2 กลุ่ม** | Two-sample t-test | Mann-Whitney U | Chi-square |
| **2 กลุ่ม (Paired)** | Paired t-test | Wilcoxon | McNemar |
| **3+ กลุ่ม** | One-way ANOVA | Kruskal-Wallis | Chi-square |
| **2 ตัวแปร** | Two-way ANOVA | Aligned ranks | Chi-square |

---

# 🎓 บทสรุป

1. **Parametric** = ใช้ค่าตัวเลข, ต้องเช็ก Normality
2. **Nonparametric** = ใช้อันดับ, ทนต่อ outliers
3. **Nominal** = ใช้นับจำนวน, Categorical only

**Key principle:** ระบุชนิดข้อมูลก่อน → ตรวจ Assumptions → เลือก Test ที่เหมาะสม!

---

**นี่คือพื้นฐาน 80% ของสิ่งที่คุณต้องรู้** 🚀

ถ้ามีส่วนไหนต้องการอธิบายเพิ่มเติม ลองบอกมาเลย!
