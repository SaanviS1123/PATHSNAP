"""Train the bounded Random Forest scorer."""
from src.models.specs import make_estimator
from src.models.training import fit_and_export,load_training_data


def train(data_path=None,random_seed=42,output_dir='models/member2/random_forest'):
    frame,features=load_training_data(data_path) if data_path else load_training_data()
    return fit_and_export(make_estimator('random_forest',seed=random_seed),'random_forest',frame,features,random_seed=random_seed,output_dir=output_dir)


if __name__=='__main__':
    train()
