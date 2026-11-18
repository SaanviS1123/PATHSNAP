"""Show a real API prediction summary in the host terminal."""
import argparse
import json
import os
from integration import wait_prediction
parser=argparse.ArgumentParser()
parser.add_argument('trip_id')
parser.add_argument('model',nargs='?',default='best',choices=['best','knn','mlp','logistic_regression','random_forest','hist_gradient_boosting'])
args=parser.parse_args()
result=wait_prediction(os.getenv('API_URL','http://127.0.0.1:8000').rstrip('/'),args.trip_id,args.model)
print(json.dumps({key:result[key] for key in ['trip_id','model','status','n_observations','timing_ms','metrics']},indent=2))
