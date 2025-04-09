import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt


df = pd.read_csv("health_activity_data.csv")
df.drop("ID", axis=1)

df[["BP_first", "BP_second"]] = df["Blood_Pressure"].str.split("/", expand=True).astype(int)
df.drop("Blood_Pressure", axis=1, inplace=True)

encoder = LabelEncoder()

for col in df.select_dtypes(include=['object']).columns:
    df[col] = encoder.fit_transform(df[col])

x = df.drop("Gender", axis=1)
y = df["Gender"]

model = RandomForestClassifier(n_estimators=100, random_state=999)
model.fit(x, y)

example = [x.iloc[11].values]
print("Prediction (0 = Female, 1 = Male): ", model.predict(example))

importances = model.feature_importances_
features = x.columns
plt.barh(features, importances)
plt.xlabel("Importance")
plt.title("Importance of attributes in random forest")
plt.show()
