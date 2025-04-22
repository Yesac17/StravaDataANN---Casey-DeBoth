# Strava 'Relative Effort' Predictor
This repo shows how I cleaned up my Strava run data and trained a small neural network to predict “Relative Effort” from seven summary features (distance, pace, heart rate, etc.).

---

## Project Overview

1. **Data Prep**  
   - Load `activities.csv` from Strava (provides all activities).  
   - Filter only “Run” entries from the last 4 years.  
   - Pick 7 key features + the built‑in `Relative Effort` score.  
   - Split 80/20 into `TrainClean.csv` and `ValidClean.csv`.

2. **Feature Scaling**  
   - Compute per‑feature standard deviation on the training set.  
   - Divide both train & valid features by those stds.

3. **Model**  
   - Using three hidden layers with ReLU activations.  
   - Trained with MSE loss.

4. **Training and Results**  
   - Print train/validation MSE every epoch (I ran for 200 epochs).  
   - Without standardizing, I got MSE ≈ 634.
   - After standardizing, I got MSE = 54.65 for train and for validate I got MSE = 53.06
   - The conclusion is that the ANN gets excellent results.


