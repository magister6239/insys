import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

df = pd.read_csv("health_activity_data.csv")
df.drop("ID", axis=1, inplace=True)
df[["BP_first","BP_second"]] = (
    df["Blood_Pressure"].str.split("/", expand=True).astype(int)
)
df.drop("Blood_Pressure", axis=1, inplace=True)

for col in df.select_dtypes(include="object"):
    df[col] = LabelEncoder().fit_transform(df[col])

x = df.drop("Smoker", axis=1)
y = df["Smoker"]
x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size = 0.3,
    random_state = 999,
    stratify = y
)


svm = SVC(kernel="rbf", gamma=0.001, C=100, random_state=999)
svm.fit(x_train, y_train)

y_predicted = svm.predict(x_test)

res = accuracy_score(y_test, y_predicted)

print(res)
