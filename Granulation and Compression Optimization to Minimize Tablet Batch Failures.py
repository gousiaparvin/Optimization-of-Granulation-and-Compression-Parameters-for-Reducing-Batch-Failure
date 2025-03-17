#!/usr/bin/env python
# coding: utf-8

# In[2]:


pip install numpy pandas matplotlib seaborn scikit-learn


# In[5]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression




# In[6]:


# Set random seed for reproducibility
np.random.seed(42)

# Define the number of samples
num_samples = 200

# Generate synthetic formulation data
granulation_method = np.random.choice([0, 1], num_samples)  # 0 = Dry, 1 = Wet
compression_force = np.random.uniform(5, 50, num_samples)  # Compression force in kN
binder_ratio = np.random.uniform(1, 10, num_samples)  # Binder ratio in %

# Batch Failure Rate Calculation (Logarithmic Influence)
batch_failure_rate = 30 - 5 * granulation_method - 0.3 * np.log(compression_force) - 2 * np.log(binder_ratio) + np.random.normal(0, 1, num_samples)
batch_failure_rate = np.clip(batch_failure_rate, 0, None)  # Ensure non-negative values

# Create DataFrame
df_granulation = pd.DataFrame({
    "Granulation Method (Wet=1, Dry=0)": granulation_method,
    "Compression Force (kN)": compression_force,
    "Binder Ratio (%)": binder_ratio,
    "Batch Failure Rate (%)": batch_failure_rate
})

# Display first few rows
print(df_granulation.head())


# In[7]:


# Log transformation for compression force and binder ratio
df_granulation["log(Compression Force)"] = np.log(df_granulation["Compression Force (kN)"])
df_granulation["log(Binder Ratio)"] = np.log(df_granulation["Binder Ratio (%)"])

# Define Features and Target
X_granulation = df_granulation[["Granulation Method (Wet=1, Dry=0)", "log(Compression Force)", "log(Binder Ratio)"]]
y_granulation = df_granulation["Batch Failure Rate (%)"]


# In[8]:


# Train a linear regression model
reg_model_granulation = LinearRegression()
reg_model_granulation.fit(X_granulation, y_granulation)

# Predict batch failure rates
batch_failure_pred = reg_model_granulation.predict(X_granulation)

print("Model trained successfully!")


# In[10]:


plt.figure(figsize=(8, 5))
sns.scatterplot(x=batch_failure_rate, y=batch_failure_pred, color="blue", label="Predicted Batch Failures")
plt.plot([0, max(batch_failure_rate)], [0, max(batch_failure_rate)], color="red", linestyle="--", label="Perfect Prediction Line")
plt.xlabel("Actual Batch Failure Rate (%)")
plt.ylabel("Predicted Batch Failure Rate (%)")
plt.title("Granulation & Compression Optimization - Batch Failure Reduction (200 Samples)")
plt.legend()
plt.show()



# In[ ]:




