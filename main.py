import yaml
from src.data.loadData import load_data
from src.features.preprocess import preprocess
from src.models.train import train_model

with open("config/config.yaml") as f:
    config = yaml.safe_load(f)

df = load_data(config['data'])
X, y, preprocessor = preprocess(df)
train_model(X, y)
