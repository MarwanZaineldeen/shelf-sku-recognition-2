# Top-20 Hard Negative SKU Pairs & Resolution Strategy Report

Empirical analysis of 31,656 DINOv3 768-D reference embeddings across 67 active SKU categories.

| Rank | Cosine Similarity | Class A (ID & Display Name) | Class B (ID & Display Name) | Primary Cause of Confusion |
| :---: | :---: | :--- | :--- | :--- |
| **#01** | **99.63%** | **[Class 25]** Lipton Green Tea Mint - 50 Tea Bags | **[Class 36]** Lipton Green Tea Mint Saver box - 50 Tea Bags | Packaging Promo Banner (Saver/Offer) |
| **#02** | **99.57%** | **[Class 28]** Lipton Yellow Label Tea - 400 g | **[Class 32]** Lipton Yellow Label Tea - 800 g | Net Weight / Pack Count Digit Difference |
| **#03** | **99.51%** | **[Class 0]** Lipton Green Tea Lemon - 50 Tea Bags | **[Class 37]** Lipton Green Tea Lemon Saver box - 50 Tea Bags | Packaging Promo Banner (Saver/Offer) |
| **#04** | **99.47%** | **[Class 24]** Lipton Green Tea Pure - 50 Tea Bags | **[Class 35]** Lipton Green Tea Pure - 50 Tea Bags | Duplicate Class Labeling |
| **#05** | **99.44%** | **[Class 0]** Lipton Green Tea Lemon - 50 Tea Bags | **[Class 61]** Lipton Green Tea Lemon - 100 Tea Bags | Net Weight / Pack Count Digit Difference |
| **#06** | **99.43%** | **[Class 3]** Lipton Yellow Label Tea Special Offer - 100 Tea Bags | **[Class 44]** Lipton Yellow Label Tea - 100 Tea Bags | Packaging Promo Banner (Saver/Offer) |
| **#07** | **99.37%** | **[Class 4]** Lipton Yellow Label Tea - 200 Tea Bags | **[Class 26]** Lipton Yellow Label Tea - 200 Tea Bags | Duplicate Class Labeling |
| **#08** | **99.36%** | **[Class 32]** Lipton Yellow Label Tea - 800 g | **[Class 38]** Lipton Yellow Label Tea Special Offer - 800 g | Packaging Promo Banner (Saver/Offer) |
| **#09** | **99.32%** | **[Class 38]** Lipton Yellow Label Tea Special Offer - 800 g | **[Class 63]** Lipton Yellow Label Tea Super Saver - 400 g | Packaging Promo Banner (Saver/Offer) |
| **#10** | **99.23%** | **[Class 29]** Lipton Green Tea Pure - 100 Tea Bags | **[Class 41]** Lipton Green Tea Mint Value Pack - 100 Tea Bags | Net Weight / Pack Count Digit Difference |
| **#11** | **99.12%** | **[Class 28]** Lipton Yellow Label Tea - 400 g | **[Class 38]** Lipton Yellow Label Tea Special Offer - 800 g | Packaging Promo Banner (Saver/Offer) |
| **#12** | **99.09%** | **[Class 37]** Lipton Green Tea Lemon Saver box - 50 Tea Bags | **[Class 61]** Lipton Green Tea Lemon - 100 Tea Bags | Packaging Promo Banner (Saver/Offer) |
| **#13** | **98.99%** | **[Class 35]** Lipton Green Tea Pure - 50 Tea Bags | **[Class 36]** Lipton Green Tea Mint Saver box - 50 Tea Bags | Packaging Promo Banner (Saver/Offer) |
| **#14** | **98.98%** | **[Class 28]** Lipton Yellow Label Tea - 400 g | **[Class 63]** Lipton Yellow Label Tea Super Saver - 400 g | Packaging Promo Banner (Saver/Offer) |
| **#15** | **98.93%** | **[Class 10]** Brooke Bond Red Label Strong - 800 g | **[Class 31]** Brooke Bond Red Label Strong - 400 g | Net Weight / Pack Count Digit Difference |
| **#16** | **98.87%** | **[Class 29]** Lipton Green Tea Pure - 100 Tea Bags | **[Class 35]** Lipton Green Tea Pure - 50 Tea Bags | Net Weight / Pack Count Digit Difference |
| **#17** | **98.86%** | **[Class 36]** Lipton Green Tea Mint Saver box - 50 Tea Bags | **[Class 41]** Lipton Green Tea Mint Value Pack - 100 Tea Bags | Packaging Promo Banner (Saver/Offer) |
| **#18** | **98.86%** | **[Class 8]** Brooke Bond Red Label - 100 Tea Bags | **[Class 48]** Brooke Bond Red Label Strong Special Offer - 100 Tea Bags | Packaging Promo Banner (Saver/Offer) |
| **#19** | **98.73%** | **[Class 24]** Lipton Green Tea Pure - 50 Tea Bags | **[Class 25]** Lipton Green Tea Mint - 50 Tea Bags | Net Weight / Pack Count Digit Difference |
| **#20** | **98.73%** | **[Class 25]** Lipton Green Tea Mint - 50 Tea Bags | **[Class 41]** Lipton Green Tea Mint Value Pack - 100 Tea Bags | Net Weight / Pack Count Digit Difference |