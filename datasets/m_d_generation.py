import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

n = 500

# Generate columns
data = {
    "age": np.random.randint(18, 65, n),
    "gender": np.random.choice(["Male", "Female", "Other"], n, p=[0.48, 0.48, 0.04]),
    "work_stress": np.random.randint(1, 11, n),  # 1-10 scale
    "sleep_hours": np.random.normal(6.5, 1.5, n).clip(3, 10),  # realistic range
    "exercise_freq": np.random.choice(["None", "1-2 days", "3-5 days", "Daily"], n, p=[0.25, 0.3, 0.3, 0.15]),
    "social_activity": np.random.randint(0, 6, n),  # 0-5 outings/week
    "alcohol_consumption": np.random.choice(["None", "Occasional", "Regular"], n, p=[0.4, 0.45, 0.15]),
    "screen_time": np.random.normal(5, 2, n).clip(1, 12),  # daily hours
    "job_satisfaction": np.random.randint(1, 11, n),  # 1-10 scale
    "income": np.random.randint(20000, 200000, n),  # annual income
    "family_support": np.random.randint(1, 11, n),  # 1-10 scale
}

df = pd.DataFrame(data)

# Introduce realistic relationship for target (mental illness)
# People with high stress, low sleep, low exercise, or low family support more likely to have issues
prob = (
    0.1 * (df["work_stress"] / 10)
    + 0.1 * (10 - df["sleep_hours"]) / 10
    + 0.05 * (10 - df["family_support"]) / 10
    + 0.05 * (10 - df["job_satisfaction"]) / 10
    + 0.05 * (df["screen_time"] / 10)
)

df["mental_illness"] = np.random.binomial(1, prob.clip(0, 0.9))

# Save to CSV
df.to_csv("mental_health_survey.csv", index=False)

print("✅ mental_health_survey.csv created successfully with 500 records.")
print(df.head())
