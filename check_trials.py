import pandas as pd
from pathlib import Path

# Load CSV
df = pd.read_csv("trials.csv")

print("✅ trials.csv loaded!")
print(df)

# Check files exist
print("\n🔍 Checking image files...")
missing = []
for col in ["real_url", "ai_url"]:
    for path in df[col]:
        if not Path(path).exists():
            missing.append(path)

if missing:
    print("⚠️ Missing files:")
    for m in missing:
        print(" -", m)
else:
    print("✅ All image files found!")
