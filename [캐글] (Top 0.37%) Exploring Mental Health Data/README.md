# Kaggle Playground Series S4E11: Reaching Top 0.37%

## 🏆 Competition Overview
- **Competition**: Playground Series S4E11 - Mental Health Data
- **Challenge**: Binary classification to predict Depression
- **Final Ranking**: Top 0.37% on the leaderboard
- **Evaluation Metric**: Accuracy

## 📊 Dataset Description
In this competition, we had to build a model to predict depression based on survey data.

Key features included:
- Demographic information: Age, Gender, Profession, Education, etc.
- Lifestyle habits: Sleep Duration, Dietary Habits, etc.
- Academic/Work-related: Academic Pressure, Work Pressure, Job Satisfaction, etc.
- Social/Economic factors: Financial Stress, Social Support, etc.

## 🔍 Approach

### 1️⃣ Exploratory Data Analysis (EDA)
- Identifying missing data patterns
  - Missing value patterns related to occupation type (student/working professional)
  - E.g., students had missing Work Pressure data, working professionals had missing Academic Pressure data
- Analyzing relationships between variables and depression
  - Spearman correlation analysis to identify key factors
  - Visualization of relationships between sleep duration, academic/work pressure and depression
- Analysis and visualization of depression rates by age group and profession

### 2️⃣ Feature Engineering
- **Age Binning**: Created 'Age_Bind' variable by binning age into 5-year intervals
  ```python
  def binding_age(df, cols):
      bins = [i for i in range(0, 105, 5)]
      labels = [f'{i}-{i+4}' for i in range(0, 100, 5)]
      df['Age_Bind'] = pd.cut(df[cols], bins=bins, labels=labels)
      return df
  ```
- **Age Group Statistics**: Calculated depression rates and calibration scores for each age group
  - Created calibration scores by multiplying depression rate with the proportion of each age group
- **Missing Value Treatment**: 
  - Handled missing values based on occupation (student/professional)
  - Filled student-related metrics ('Academic Pressure', 'Study Satisfaction') and professional-related metrics ('Work Pressure', 'Job Satisfaction') with respective group means
- **Category Cleanup**: 
  - Consolidated degree categories (e.g., 'B.Com', 'BCom', 'B.Comm' → 'B.Com')
  - Handled low-frequency categories by grouping them as 'Other' (Name, City, Profession, Degree)
  - Treated invalid categories in Sleep Duration, Dietary Habits as 'Noise'

### 3️⃣ Modeling Strategy
- **CatBoost-based Model**: 
  - Selected CatBoost for its effective handling of categorical variables
  - Used GPU acceleration to reduce training time
  ```python
  # CatBoost core parameters
  Params = {
      'loss_function': 'Logloss',
      'eval_metric': 'AUC',
      'learning_rate': 0.08114394459649094,
      'iterations': 1000,
      'depth': 6,
      'random_strength': 0,
      'l2_leaf_reg': 0.7047064221215757,
      'task_type': 'GPU'
  }
  ```
- **Cross-validation**: Implemented 10-fold cross-validation for model stability
- **Hyperparameter Optimization**: 
  - Used Optuna for hyperparameter tuning
  - Narrowed search range around previously optimal parameters for efficient exploration

### 4️⃣ Ensemble and Final Approach
- **AutoGluon Implementation**:
  - Used AutoGluon library to automatically train and ensemble multiple models
  - Enhanced performance through model stacking (combination of LightGBM, CatBoost, XGBoost, etc.)
- **Hill Climbing Ensemble**: 
  - Applied Hill Climbing technique starting from a base model and progressively improving
  - Searched for optimal ensemble combinations by adjusting weights
  ```python
  # Hill Climbing example code
  def hill_climbing(base_weights, predictions, target, steps=100):
      best_weights = base_weights.copy()
      best_score = compute_score(predictions, best_weights, target)
      
      for _ in range(steps):
          new_weights = perturb_weights(best_weights)
          new_score = compute_score(predictions, new_weights, target)
          
          if new_score > best_score:
              best_weights = new_weights
              best_score = new_score
              
      return best_weights, best_score
  ```
- **Various Experimental Approaches**:
  - Testing with/without original dataset integration
  - Trying different feature engineering combinations

## 📈 Progress in Performance Improvement
1. **Base Model** (Vanilla CatBoost): Leaderboard score ~0.93
2. **Feature Engineering Improvements**: 
   - Age binning + Occupation-specific missing value handling: Score ~0.94
   - Low-frequency category handling: Score ~0.942
3. **Hyperparameter Tuning**: Score ~0.944
4. **Ensemble Strategies**:
   - AutoGluon baseline: Score ~0.943
   - CatBoost + AutoGluon: Score ~0.944
   - Hill Climbing ensemble: Final score 0.94381 (Top 0.37%)

## 🔬 Key Findings from Experiments
Below are the results from various experimental approaches:

```
[1] Original data integration + Degree category cleanup + clean columns + remove_noise + Catboost (default params)
- Overall Train Accuracy: 0.9450541
- Overall Valid Accuracy: 0.9406308
- Gap Between Train-Valid : 0.0044233

[2] (No original) + Degree category cleanup + clean columns + remove_noise + Catboost (default params)
- Overall Train Accuracy: 0.9447374
- Overall Valid Accuracy: 0.9400355
- Gap Between Train-Valid : 0.0047019

[3] Original data integration + Degree category cleanup + (age binning + age group stats) + clean columns + remove_noise + Catboost (default params)
- Overall Train Accuracy: 0.9455993
- Overall Valid Accuracy: 0.9404632
- Gap Between Train-Valid : 0.0051361

[4] Original data integration + Degree category cleanup + age binning + age group stats + (Student job cleanup) + clean columns + remove_noise + Catboost (default params)
- Overall Train Accuracy: 0.9453643
- Overall Valid Accuracy: 0.9404911
- Gap Between Train-Valid : 0.0048732

[5] Original data integration + Degree category cleanup + age binning + age group stats + (Null handling) + clean columns + remove_noise + Catboost (default params)
- Overall Train Accuracy: 0.9460073
- Overall Valid Accuracy: 0.9405958
- Gap Between Train-Valid : 0.0054115
```

## 💡 Key Lessons Learned
- **Importance of Domain Understanding**: Differentiating between students and working professionals was crucial
- **Missing Value Strategy**: Group-specific handling of missing values proved more effective than simple mean imputation
- **Categorical Variable Handling**: CatBoost showed excellent performance with datasets containing many categorical variables
- **Importance of Cross-validation**: 10-fold cross-validation prevented overfitting and ensured model stability
- **Ensemble Methodology**: Consistent performance improvements were achievable through diverse model ensembling

## 📁 Code Structure
- `playground-s04e11-finding-best-fe.ipynb`: Exploration of optimal feature engineering
- `playground-s04e11-autogluon.ipynb`: Automated modeling using AutoGluon
- `playground-s04e11-autogluon-hillclimbing.ipynb`: Ensemble using Hill Climbing technique
- `playground-s04e11-final-testing.ipynb`: Final model testing
- `playground-s04e11-custom-ensemble.ipynb`: Custom ensemble approach

## 🚀 Future Improvement Possibilities
- Apply NLP techniques to text-based features (names, cities, etc.)
- Try more diverse model ensembles (Stacking, Blending)
- Introduce neural network-based approaches (Neural Networks for Tabular data)
- Utilize additional external datasets
- Develop differentiated feature engineering through feature importance analysis 