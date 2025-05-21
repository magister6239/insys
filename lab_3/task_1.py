import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

df = pd.read_csv("health_activity_data.csv")
df.drop("ID", axis=1, inplace=True)
df[["BP_first","BP_second"]] = (
    df["Blood_Pressure"].str.split("/", expand=True).astype(int)
)
df.drop("Blood_Pressure", axis=1, inplace=True)

for col in df.select_dtypes(include="object"):
    df[col] = LabelEncoder().fit_transform(df[col])

X = df.drop("Smoker", axis=1)
y = df["Smoker"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

models = {
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=999),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=999)
}

for name, mdl in models.items():
    mdl.fit(X_train, y_train)
    y_pred = mdl.predict(X_test)
    print(f"{name} accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(classification_report(y_test, y_pred))

tree = models["Random Forest"]
importances = tree.feature_importances_
plt.barh(X.columns, importances)
plt.xlabel("Importance")
plt.show()
