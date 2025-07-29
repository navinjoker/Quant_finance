from keras_tuner.tuners import RandomSearch
from tensorflow.keras.callbacks import EarlyStopping
from pipeline.model import build_model

def train_model(X_train, y_train, input_shape, save_path, tuner_logdir):
    tuner = RandomSearch(
        lambda hp: build_model(hp, input_shape),
        objective='val_loss',
        max_trials=50,
        executions_per_trial=1,
        directory=tuner_logdir,
        project_name='hybrid_tuning'
    )
    tuner.search(X_train, y_train, validation_split=0.2, epochs=10,
                 callbacks=[EarlyStopping(patience=3)])
    best_model = tuner.get_best_models(num_models=1)[0]
    best_model.save(save_path)