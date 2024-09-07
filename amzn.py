import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import random
from random import randint


file_path = "portfolio_data.csv" 
df = pd.read_csv(file_path)
data = np.array(df)

DAYS_TRACKING = 10
dates = pd.to_datetime(df['Date'])
values = df['AMZN']

# Step 3: Plot the values
plt.figure(figsize=(10, 6))
plt.plot(dates, values, marker='o', linestyle='-')

# Customizing the plot
plt.title('Values over Time')
plt.xlabel('Date')
plt.ylabel('Value')
plt.grid(True)
plt.xticks(rotation=45)  # Rotate date labels for better readability

# Show the plot
plt.tight_layout()
plt.show()