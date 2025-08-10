import pickle

# Update with your file path
with open('startup_profit_prediction_lr_model.pkl', 'rb') as f:
    model = pickle.load(f)

print(type(model))         # Check what kind of object this is
print(model)               # Print a summary (if it’s an ML model)
