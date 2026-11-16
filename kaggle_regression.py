"""
Linear Regression on a real Kaggle dataset: Housing Prices (yasserh).
https://www.kaggle.com/datasets/yasserh/housing-prices-dataset

545 houses, columns:
  price, area, bedrooms, bathrooms, stories, parking           (numbers)
  mainroad, guestroom, basement, hotwaterheating,
  airconditioning, prefarea                                    (yes / no)
  furnishingstatus  (furnished / semi-furnished / unfurnished)

Getting the data (either option works):
  A) Automatic:  pip install kagglehub   (script downloads it on first run)
  B) Manual:     download Housing.csv from the Kaggle page and save it as
                 data/Housing.csv in this repo

Run:  python kaggle_regression.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

LOCAL_CSV = Path("data/Housing.csv")
KAGGLE_DATASET = "yasserh/housing-prices-dataset"


# ---------------------------------------------------------------
# STEP 1. Load the data (local file first, otherwise Kaggle)
# ---------------------------------------------------------------
def load_data() -> pd.DataFrame:
    if LOCAL_CSV.exists():
        print(f"Loading {LOCAL_CSV}")
        return pd.read_csv(LOCAL_CSV)
    try:
        import kagglehub
    except ImportError:
        raise SystemExit(
            "Data not found. Either `pip install kagglehub`, or download Housing.csv "
            f"from https://www.kaggle.com/datasets/{KAGGLE_DATASET} into data/")
    path = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    print(f"Downloaded from Kaggle to {path}")
    df = pd.read_csv(path / "Housing.csv")
    LOCAL_CSV.parent.mkdir(exist_ok=True)
    df.to_csv(LOCAL_CSV, index=False)          # cache a copy for next time
    return df


df = load_data()

# ---------------------------------------------------------------
# STEP 2. Explore (always look before you model)
# ---------------------------------------------------------------
print("\n=== Shape ===", df.shape)
print("\n=== First rows ===\n", df.head())
print("\n=== Missing values ===\n", df.isna().sum()[df.isna().sum() > 0]
      if df.isna().any().any() else "none")
print("\n=== Price summary ===\n", df["price"].describe().round(0))

numeric = ["area", "bedrooms", "bathrooms", "stories", "parking"]
yes_no = ["mainroad", "guestroom", "basement", "hotwaterheating",
          "airconditioning", "prefarea"]
categorical = ["furnishingstatus"]

print("\n=== Correlation of numeric features with price ===")
print(df[numeric + ["price"]].corr()["price"].drop("price")
      .sort_values(ascending=False).round(2))

# yes/no -> 1/0 (simple and readable)
for col in yes_no:
    df[col] = df[col].str.strip().str.lower().map({"yes": 1, "no": 0})

# ---------------------------------------------------------------
# STEP 3. Simple regression: price ~ area
# ---------------------------------------------------------------
simple = LinearRegression().fit(df[["area"]], df["price"])
print("\n=== Simple regression (price ~ area) ===")
print(f"Each extra sq ft adds ~{simple.coef_[0]:,.0f} to price")
print(f"R^2 = {simple.score(df[['area']], df['price']):.3f}")

plt.figure(figsize=(7, 4.5))
plt.scatter(df["area"], df["price"], s=10, alpha=0.5, label="houses")
xs = pd.DataFrame({"area": np.linspace(df["area"].min(), df["area"].max(), 100)})
plt.plot(xs["area"], simple.predict(xs), color="red", label="fitted line")
plt.xlabel("Area (sq ft)"); plt.ylabel("Price"); plt.legend()
plt.title("Kaggle housing: price vs area")
plt.tight_layout(); plt.savefig("kaggle_simple.png", dpi=120); plt.close()

# ---------------------------------------------------------------
# STEP 4. Multiple regression with all features
# ---------------------------------------------------------------
X = df.drop(columns="price")
y = df["price"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

def make_model(log_target: bool = False):
    prep = ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("bin", "passthrough", yes_no),
        ("cat", OneHotEncoder(drop="first"), categorical),
    ])
    pipe = Pipeline([("prep", prep), ("lr", LinearRegression())])
    if log_target:   # model log(price): handles the right-skewed prices
        return TransformedTargetRegressor(regressor=pipe, func=np.log, inverse_func=np.exp)
    return pipe

def evaluate(name, model):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    cv = cross_val_score(model, X, y, scoring="r2",
                         cv=KFold(5, shuffle=True, random_state=42))  # data is sorted by price, so shuffle
    print(f"\n=== {name} — test set ===")
    print(f"MAE : {mean_absolute_error(y_test, pred):>12,.0f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, pred)):>12,.0f}")
    print(f"R^2 : {r2_score(y_test, pred):>12.3f}")
    print(f"5-fold CV R^2: {cv.mean():.3f} ± {cv.std():.3f}")
    return pred

model = make_model()
pred = evaluate("Multiple regression", model)

# ---------------------------------------------------------------
# STEP 5. Which features matter most?
# (numeric features are scaled, so their coefficients = effect of +1 std-dev;
#  yes/no features = effect of having it vs not)
# ---------------------------------------------------------------
names = model.named_steps["prep"].get_feature_names_out()
coefs = pd.Series(model.named_steps["lr"].coef_, index=names).sort_values()
print("\n=== Coefficients (price change) ===")
print(coefs.round(0))

plt.figure(figsize=(7, 5))
coefs.plot.barh(color=["#c0504d" if c < 0 else "#4f81bd" for c in coefs])
plt.title("Effect of each feature on price"); plt.xlabel("Price change")
plt.tight_layout(); plt.savefig("kaggle_coefficients.png", dpi=120); plt.close()

# ---------------------------------------------------------------
# STEP 6. Residuals: real data is rarely as tidy as synthetic data
# ---------------------------------------------------------------
plt.figure(figsize=(7, 4.5))
plt.scatter(pred, y_test - pred, s=12, alpha=0.6)
plt.axhline(0, color="red")
plt.xlabel("Predicted price"); plt.ylabel("Residual (actual - predicted)")
plt.title("Residuals — look for funnels or curves")
plt.tight_layout(); plt.savefig("kaggle_residuals.png", dpi=120); plt.close()

# ---------------------------------------------------------------
# STEP 7. Try an improvement: predict log(price)
# ---------------------------------------------------------------
evaluate("Multiple regression on log(price)", make_model(log_target=True))

# ---------------------------------------------------------------
# STEP 8. Predict a new house
# ---------------------------------------------------------------
new_house = pd.DataFrame([{
    "area": 6000, "bedrooms": 3, "bathrooms": 2, "stories": 2, "parking": 1,
    "mainroad": 1, "guestroom": 0, "basement": 1, "hotwaterheating": 0,
    "airconditioning": 1, "prefarea": 1, "furnishingstatus": "semi-furnished",
}])
print(f"\nPredicted price for the example house: {model.predict(new_house)[0]:,.0f}")
