"""Train boosting without a random row-level validation split."""
from src.models.specs import make_estimator
from src.models.training import fit_and_export,load_training_data


def train(data_path=None,random_seed=42,output_dir='models/member2/hist_gradient_boosting'):
    frame,features=load_training_data(data_path) if data_path else load_training_data()
    return fit_and_export(make_estimator('hist_gradient_boosting',seed=random_seed),'hist_gradient_boosting',frame,features,random_seed=random_seed,output_dir=output_dir)


if __name__=='__main__':
    train()
