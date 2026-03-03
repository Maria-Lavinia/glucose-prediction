import keras
from matplotlib import units
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import keras_tuner as kt

from model_handling import create_sequences



def build_hyperparameter_tuning( hp, input_shape):
     
    model = keras.models.Sequential()
    
    hp_units = hp.Int('units', min_value=32, max_value=256, step=32)
    hp_dropout_rate = hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1)
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    number_lstm_layers = hp.Int('num_lstm_layers', min_value=1, max_value=2, step=1)
    for i in range(number_lstm_layers):
        model.add(keras.layers.LSTM(
            units=hp_units, 
            return_sequences=(i < number_lstm_layers - 1),  
            input_shape=input_shape if i == 0 else None
        ))

    # model.add(keras.layers.LSTM(
    #     units=hp_units, 
    #     input_shape=input_shape,
    # ))
    
    model.add(keras.layers.Dropout(hp_dropout_rate))
    
    number_dense_layers = hp.Int('num_dense_layers', min_value=1, max_value=2, step=1)
    for _ in range(number_dense_layers):
        model.add(keras.layers.Dense(units=hp_units//2, activation='relu'))  
    
    # model.add(keras.layers.Dense(32, activation='relu')) 
    
    model.add(keras.layers.Dense(1))  # Output layer for regression
    
    model.summary()
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
        loss='mean_squared_error',
        metrics=['mean_absolute_error', keras.metrics.RootMeanSquaredError(name='rmse')]
    )

    return model


def setup_hyperparameter_tuning(X_train, y_train, X_val, y_val, input_shape, epochs):

    tuner = kt.BayesianOptimization(
        hypermodel=lambda hp: build_hyperparameter_tuning(hp, input_shape),
        objective='val_loss',
        max_trials=20,
        directory='hyperparameter_tuning',
        project_name='glucose_prediction_lstm'
    )
    
    tuner.search(X_train, y_train, epochs, validation_data=(X_val, y_val), callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)])
    
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f"Best hyperparameters: Units={best_hps.get('units')}, Dropout Rate={best_hps.get('dropout_rate')}, Learning Rate={best_hps.get('learning_rate')}")
    
    return best_hps
    

def run_hyperparameter_search(df) :
    patients_ids = df['patient_id'].unique()
    tuning_ids = patients_ids[:4] 
    
    all_X = []
    all_Y= []
    
    for p_id in tuning_ids: 
        p_df = df[df['patient_id'] == p_id].copy()
        
        features = [col for col in p_df.columns if col not in ['patient_id', 'timestamp', 'glucose']]
        target = 'glucose'
        
        scaler_X = StandardScaler()
        X_train_scaled = scaler_X.fit_transform(p_df[features])

        scaler_y = StandardScaler()
        y_train_scaled = scaler_y.fit_transform(p_df[[target]])
        
        train_data_combined = np.hstack([X_train_scaled, y_train_scaled])
        
        X_train, y_train = create_sequences(train_data_combined, y_train_scaled, time_steps=36)
        
        all_X.append(X_train)
        all_Y.append(y_train)
      
    X_pool = np.concatenate(all_X, axis=0)
    Y_pool = np.concatenate(all_Y, axis=0)
    
    X_train, X_val, y_train, y_val = train_test_split(X_pool, Y_pool, test_size=0.2, random_state=42)
    
    setup_hyperparameter_tuning(X_train, y_train,X_val, y_val, input_shape=(36, X_train.shape[2]), epochs=10)
    
    
    