📌 Objective

This notebook implements the TOPSIS (Technique for Order Preference by Similarity to Ideal Solution) algorithm to rank different pre-trained NLP models based on multiple performance criteria.

The models evaluated:

BERT

RoBERTa

Sentence-BERT

DistilBERT

📂 1️⃣ Libraries Used
import numpy as np
import pandas as pd

NumPy → Mathematical operations

Pandas → Data handling

⚠ Note: matplotlib.pyplot as plt is used later but not imported in the notebook. You should add:

import matplotlib.pyplot as plt
📊 2️⃣ Dataset (Embedded Data)

The dataset is created inside the notebook:

Model	CosineSimilarity	Accuracy	InferenceTime	ModelSize
BERT	0.87	0.92	120	420
RoBERTa	0.89	0.94	150	500
Sentence-BERT	0.91	0.93	90	350
DistilBERT	0.85	0.90	60	250
Criteria Explanation

CosineSimilarity → Higher is better (+)

Accuracy → Higher is better (+)

InferenceTime → Lower is better (-)

ModelSize → Lower is better (-)

⚙ 3️⃣ TOPSIS Parameters
weights = np.array([0.4, 0.3, 0.2, 0.1])
impacts = ['+', '+', '-', '-']
🔹 Weights

CosineSimilarity → 0.4 (Most important)

Accuracy → 0.3

InferenceTime → 0.2

ModelSize → 0.1

🔹 Impacts

+ → Benefit criteria (maximize)

- → Cost criteria (minimize)

🧮 4️⃣ TOPSIS Steps Implemented
✅ Step 1: Normalization
norm_matrix = decision_matrix / np.sqrt((decision_matrix ** 2).sum(axis=0))

Purpose:

Removes unit differences

Converts values into comparable scale

Formula used:

𝑅
𝑖
𝑗
=
𝑥
𝑖
𝑗
∑
𝑥
𝑖
𝑗
2
R
ij
	​

=
∑x
ij
2
	​

	​

x
ij
	​

	​

✅ Step 2: Weighted Normalization
weighted_matrix = norm_matrix * weights

Each normalized column is multiplied by its weight.

✅ Step 3: Ideal Best & Ideal Worst
if impacts[i] == '+':
    ideal_best[i] = weighted_matrix[:, i].max()
    ideal_worst[i] = weighted_matrix[:, i].min()
else:
    ideal_best[i] = weighted_matrix[:, i].min()
    ideal_worst[i] = weighted_matrix[:, i].max()

Ideal Best → Best possible value

Ideal Worst → Worst possible value

For:

Benefit criteria → max is best

Cost criteria → min is best

✅ Step 4: Distance Calculation
distance_best = np.sqrt(((weighted_matrix - ideal_best) ** 2).sum(axis=1))
distance_worst = np.sqrt(((weighted_matrix - ideal_worst) ** 2).sum(axis=1))

Calculates Euclidean distance from:

Ideal Best

Ideal Worst

✅ Step 5: TOPSIS Score
topsis_score = distance_worst / (distance_best + distance_worst)

Formula:

𝑆
𝑐
𝑜
𝑟
𝑒
=
𝐷
−
𝐷
+
+
𝐷
−
Score=
D
+
+D
−
D
−
	​


Where:

𝐷
+
D
+
 = Distance from ideal best

𝐷
−
D
−
 = Distance from ideal worst

Higher score = Better model

✅ Step 6: Ranking
result["Rank"] = result["TOPSIS Score"].rank(ascending=False)

Higher score → Rank 1

Results sorted by rank

💾 5️⃣ Output Files Generated
📄 1. CSV File
topsis_result.csv

Contains:

Model

TOPSIS Score

Rank

📊 2. Bar Chart Image
topsis_ranking.png

Shows:

Model names (X-axis)

TOPSIS Score (Y-axis)

📈 Final Output Displayed

The notebook prints:

Confirmation message

Ranked result table

Generated file names

🧠 What This Notebook Achieves

✔ Multi-criteria decision making
✔ Model comparison using mathematical ranking
✔ Automatic CSV export
✔ Visualization of ranking
✔ Fully self-contained (no external dataset required)

⚠ Improvements Suggested

Add missing import:

import matplotlib.pyplot as plt

Add input validation for:

Weight sum check

Impact length check

Make it dynamic (accept CSV input instead of hardcoding data)

🎯 Conclusion

This notebook successfully applies the TOPSIS algorithm to rank pre-trained NLP models based on performance metrics and system constraints.

It is clean, modular, and follows the correct TOPSIS mathematical steps.
