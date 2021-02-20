"""
By : Paritosh
Date 20/Feb/21
Description: 
"""
import os
import json
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard, ModelCheckpoint

###### change for featuer
def load_data(args):
    """
    load dataset from csv
    """
    processed_data= pd.read_csv(f"{args.train}/process.csv")
    X_feature = processed_data[['sex','age','thalach']]
    y_feature =  processed_data['trestbps']
    X_train, X_test, y_train, y_test = train_test_split(X_feature, y_feature, test_size=args.test_size, random_state=42)
    return X_train, X_test, y_train, y_test
    


def training_main(args):
    """
    """
    print("Training Start...")

    default_jobname_dict = {
        'job_name': f"model-epoch-{args.epochs}-batch-{args.batch_size}-train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    }
    job_name = json.loads(
        os.environ.get('SM_TRAINING_ENV', json.dumps(default_jobname_dict))
    )["job_name"]
    os.makedirs(args.tf_logs_path, exist_ok=True)
    os.makedirs(args.model_output, exist_ok=True)
    logs_dir = "{}/{}".format(args.tf_logs_path, job_name)
    print("Writing TensorBoard logs to {}".format(logs_dir))
    # Initializing TensorFlow summary writer
    tf_writer = tf.summary.create_file_writer(logs_dir)
    tf_writer.set_as_default()
    
    call_back = [
        ModelCheckpoint(monitor='val_loss',
                        filepath=f'{args.checkpoint_path}/chkpoint_best_weights_{args.version}.hdf5',
                        save_best_only=True,
                        save_weights_only=False),
        # stop prog when got min loss
        EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=200),
        # tensor board callback.
        TensorBoard(log_dir=logs_dir)
    ]
    
    X_train, X_test, y_train, y_test = load_data(args) #load dataset
    print(f" Total Train: {len(X_train)} \n Total Test: {len(X_test)} ")
    model=get_model(args) 
    
    history = model.fit(X_train, y_train, 
                        validation_data=(X_test, y_test), 
                        epochs=args.epochs, 
                        batch_size=args.batch_size, 
                        verbose=2, 
                        callbacks=[call_back])
    
    model.save(args.model_output + f'/{args.version}')
    
    print("Training finished....")
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--input_size', type=int, default=3,
                        help='input feature size')

    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Initial learning rate.')

    parser.add_argument('--epochs', type=int, default=500,
                        help='The number of steps to use for training.')

    parser.add_argument('--batch_size', type=int, default=18,
                        help='Batch size for training.')
    
    parser.add_argument('--test_size', type=float, default=0.15,
                        help='text split')

    parser.add_argument('--optimizer', type=str.lower, default='adam',
                        help='Optimizer to be used during training.[adam,rmsprop, sgd,adadelta]')
    
    parser.add_argument('--version', type=str, default=1,
                        help='Default Pose')
    
    parser.add_argument('--model_dir', type=str)

    args = parser.parse_args()
    
    print("---------------------------------------")
    print(args)
    print("---------------------------------------")
    
    training_main(args)


#     python training.py --train "./" --model_output --tf_logs_path "logs"