"""Train the provisional scaled Logistic Regression scorer."""
from src.models.specs import make_estimator
from src.models.training import fit_and_export,load_training_data


def train(data_path=None,random_seed=42,output_dir='models/member2/logistic_regression'):
    frame,features=load_training_data(data_path) if data_path else load_training_data()
    return fit_and_export(make_estimator('logistic_regression',seed=random_seed),'logistic_regression',frame,features,random_seed=random_seed,output_dir=output_dir)


if __name__=='__main__':
    train()
