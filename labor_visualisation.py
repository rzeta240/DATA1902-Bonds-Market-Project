import pandas as pd
import matplotlib.pyplot as plt


from Scripts.dataset_reader import get_labor_productivity
df = get_labor_productivity()


print(df.head())   # first 5 rows
print(df.info())   # column types, null counts
print(df.columns)  # all column names

x = df["Date"].tolist()
productivity_nonfarm = df["Nonfarm business sector Labor productivity"].tolist()
productivity_business = df["Business sector Labor productivity"].tolist()
productivity_nonfinancial = df["Nonfinancial corporate sector Labor productivity"].tolist()
productivity_manufacturing = df["Manufacturing sector Labor productivity"].tolist()
productivity_durable = df["Durable manufacturing sector Labor productivity"].tolist()
productivity_nondurable = df["Nondurable manufacturing sector Labor productivity"].tolist()

plt.plot(x, productivity_nonfarm, color='red', linewidth=1.5, linestyle='-.', label='Nonfarm', alpha=0.7)
plt.plot(x, productivity_business, color='green', linewidth=2.5, label='Business', alpha=0.7)
plt.plot(x, productivity_nonfinancial, color='blue', linewidth=1.5, linestyle=':', label='Nonfinancial', alpha=0.7)
plt.plot(x, productivity_manufacturing, color='purple', linewidth=1.5, linestyle='-.', label='Manufacturing', alpha=0.7)
plt.plot(x, productivity_durable, color='orange', linewidth=2.5, label='Durable', alpha=0.7)
plt.plot(x, productivity_nondurable, color='brown', linewidth=1.5, linestyle=':', label='Nondurable', alpha=0.7)

plt.xlabel("Date")
plt.ylabel("Labor Productivity (% Change from Previous Quarter)")
plt.title("Labor Productivity Across Sectors Over Time")
plt.xticks(rotation=45)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('plot.png')
plt.show()
