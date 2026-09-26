from xgboost import XGBClassifier

model = XGBClassifier()

model.load_model("model.json")

print("Model loaded successfully!")
print(type(model))