import os
import keras
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import shap

def read_csv_for_modeling(folder_path):
    files = os.listdir(folder_path)
    
    csv_files = [os.path.join(folder_path, f) for f in files if f.endswith('.csv')]
    df_list = [pd.read_csv(f) for f in csv_files]
    combined_df = pd.concat(df_list, ignore_index=True)
    print(combined_df.columns.tolist())
    
    return combined_df


def create_sequences(data_array, target_array, time_steps=36, horizon=6):
    """
    Turns a 2D array into a 3D array of sequences.
    time_steps=12 assumes 1 hour of history if data is every 5 mins.
    """
    X, y = [], []
    
    for i in range(len(data_array) - time_steps - horizon + 1):
        
        x_window = data_array[i : i + time_steps]
       
        y_target = target_array[i + time_steps + horizon - 1] 
        
        if np.isnan(x_window).any() or np.isnan(y_target):
            continue
        
        X.append(x_window)
        y.append(y_target)
        
    return np.array(X), np.array(y)

def train_patient_model(df, model_data_folder):

    os.makedirs(model_data_folder, exist_ok=True)
    
    patient_ids = df['patient_id'].unique()
    print("Unique patient IDs:", patient_ids)
       
    results = []
    for test_patient in patient_ids:
        print(f"Processing patient_id: {test_patient}")
        train_df = df[df['patient_id'] != test_patient].copy()
        test_df = df[df['patient_id'] == test_patient].copy()
        
        # print("Train patients", train_df['patient_id'].unique())
        # print("Test patient", test_df['patient_id'].unique())
        
        TIME_STEPS = 36
        features = [col for col in train_df.columns if col not in ['patient_id', 'timestamp', 'glucose']]
        target = 'glucose'
    
        X_train_list = []
        Y_train_list = []  
        
        
        scaler_X = StandardScaler()
        X_train_scaled = scaler_X.fit_transform(train_df[features])

        scaler_y = StandardScaler()
        y_train_scaled = scaler_y.fit_transform(train_df[[target]])
        
        train_data_combined = train_df.copy()
        train_data_combined[features] = X_train_scaled
        train_data_combined[target] = y_train_scaled
        
        for p_id in train_data_combined['patient_id'].unique():
            p_data = train_data_combined[train_data_combined['patient_id'] == p_id]
            
            
            p_features = p_data[features].to_numpy()
            p_target = p_data[[target]].to_numpy()
              
            
            p_data_combined = np.hstack([p_features, p_target])
            x_p, y_p = create_sequences(p_data_combined, p_target, time_steps=TIME_STEPS)
            
            X_train_list.append(x_p)
            Y_train_list.append(y_p)
            
            
        X_test_scaled = scaler_X.transform(test_df[features])
        y_test_scaled = scaler_y.transform(test_df[[target]])
        test_data_combined = np.hstack([X_test_scaled, y_test_scaled])
        
        X_train, y_train = np.concatenate(X_train_list, axis=0), np.concatenate(Y_train_list, axis=0)
        X_test, y_test = create_sequences(test_data_combined, y_test_scaled, time_steps=TIME_STEPS)
        
        print("Missing values in features:", train_df.isnull().sum().sum())
        print("Missing values in target:", train_df[target].isnull().sum())
        
        print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
        
        print("====================================")
        print("Building model...")
        print("====================================")
        
        
        
        model = build_lstm_model(input_shape=(TIME_STEPS, len(features)+1))
        
        early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=5, 
        restore_best_weights=True
        )   
        
        model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=1, callbacks=[early_stop])
        
        y_pred_scaled = model.predict(X_test)
        y_pred = scaler_y.inverse_transform(y_pred_scaled)
        y_true = scaler_y.inverse_transform(y_test)
        
        plot_results(y_true, y_pred, model_data_folder, test_patient)

        if test_patient == patient_ids[0]:

            print("Running explainability analysis...")

            # Optional: reduce size to make it faster
            X_explain = X_test[:200]
            y_explain = y_test[:200]

            # 1️⃣ Fixed permutation feature importance
            feature_names = features + ["past_glucose"]
            permutation_feature_importance(model, X_explain, y_explain, feature_names)

            # 2️⃣ Time-step importance
            timestep_importance(
                model,
                X_explain,
                y_explain
            )

            # 3️⃣ Feature × Time heatmap (MAIN ONE)
            feature_time_importance(
                model,
                X_explain,
                y_explain,
                features + ["glucose"]
            )

        mae = mean_absolute_error(y_true, y_pred)
        results.append(mae)
        print(f"Final MAE for {test_patient}: {mae:.2f} mg/dL")
            
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        print(f"Final RMSE for {test_patient}: {rmse:.2f} mg/dL")
                
    print("=========")
    print("MODEL REUSLTS")
    print("=========")
    
    print(results)

    return results

def build_lstm_model(input_shape, units=32, dropout_rate=0.1, learning_rate=0.001):
    """
    Builds and compiles a Sequential LSTM model.
    
    Parameters:
    - input_shape: (time_steps, num_features) -> e.g., (12, 11)
    - units: Number of 'neurons' in the LSTM layer.
    - dropout_rate: Fraction of neurons to 'turn off' to prevent overfitting.
    - learning_rate: How fast the model updates its weights.
    """
    
    model = keras.models.Sequential()

    model.add(keras.layers.LSTM(
        units=units, 
        input_shape=input_shape,
        return_sequences=True
    ))
    
    model.add(keras.layers.LSTM(
        units=units, 
        return_sequences=False
    ))
    
    model.add(keras.layers.Dropout(dropout_rate))
    
    model.add(keras.layers.Dense(units//2, activation='relu')) 
    
    model.add(keras.layers.Dense(units//2, activation='relu')) 
    
    model.add(keras.layers.Dense(1))  # Output layer for regression
    
    model.summary()
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mean_squared_error',
        metrics=['mean_absolute_error', keras.metrics.RootMeanSquaredError(name='rmse')]
    )

    return model


def plot_results(expected_value, predicted_value, model_data_folder, test_patient):
    plt.figure(figsize=(12,5))
    
    plt.plot(expected_value[:300], label="True")
    plt.plot(predicted_value[:300], label="Predicted")
    
    plt.title("True vs Predicted Glucose")
    plt.xlabel("Time Stpes")
    plt.ylabel("Glucose (mg/dL)")
    plt.legend()
    
    plt.savefig(model_data_folder + f"/patient_{test_patient}_1.png")
    
    plt.figure(figsize=(6,6))
    
    plt.scatter(expected_value, predicted_value, alpha=0.3)
    
    plt.plot([expected_value.min(), expected_value.max()], [expected_value.min(), expected_value.max()], color="red", linestyle="--")  
    
    plt.xlabel("True Glucose (mg/dL)")
    plt.ylabel("Predicted Glucose (mg/dL)")
    plt.title("True vs Predicted Glucose Scatter Plot")

    plt.savefig(model_data_folder + f"/patient_{test_patient}_2.png")
    
    plt.figure(figsize=(6,6))
    errors = expected_value - predicted_value
    plt.hist(errors, bins=30, edgecolor='black')
    plt.title("Distribution of Prediction Errors")
    plt.xlabel("Prediction Error (mg/dL)")
    plt.ylabel("Frequency")

    plt.savefig(model_data_folder + f"/patient_{test_patient}_3.png")
    plt.close()



def permutation_feature_importance(model, X_test, y_test, features):

    import numpy as np
    from sklearn.metrics import mean_absolute_error

    baseline_pred = model.predict(X_test)
    baseline_mae = mean_absolute_error(y_test, baseline_pred)

    importances = []

    for f in range(X_test.shape[2]):

        X_permuted = X_test.copy()

        # shuffle entire sequences, not timesteps
        perm_idx = np.random.permutation(X_permuted.shape[0])
        X_permuted[:, :, f] = X_permuted[perm_idx, :, f]

        perm_pred = model.predict(X_permuted)
        perm_mae = mean_absolute_error(y_test, perm_pred)

        importance = perm_mae - baseline_mae
        importances.append(importance)

    importances = np.array(importances)

    plt.figure(figsize=(8,6))
    plt.barh(features, importances)
    plt.xlabel("Increase in MAE after permutation")
    plt.title("Permutation Feature Importance")
    plt.tight_layout()

    plt.show()


def timestep_importance(model, X_test, y_test):
    baseline_pred = model.predict(X_test)
    baseline_mae = mean_absolute_error(y_test, baseline_pred)

    time_steps = X_test.shape[1]
    importances = []

    for t in range(time_steps):
        X_permuted = X_test.copy()

        # shuffle ONLY timestep t
        perm_idx = np.random.permutation(X_permuted.shape[0])
        X_permuted[:, t, :] = X_permuted[perm_idx, t, :]

        perm_pred = model.predict(X_permuted)
        perm_mae = mean_absolute_error(y_test, perm_pred)

        importances.append(perm_mae - baseline_mae)

    plt.plot(importances)
    plt.title("Timestep Importance")
    plt.xlabel("Time step (past)")
    plt.ylabel("MAE increase")
    plt.show()


def feature_time_importance(model, X_test, y_test, feature_names):
    baseline_pred = model.predict(X_test)
    baseline_mae = mean_absolute_error(y_test, baseline_pred)

    T = X_test.shape[1]
    F = X_test.shape[2]

    importance_matrix = np.zeros((T, F))

    for t in range(T):
        for f in range(F):
            X_perm = X_test.copy()

            perm_idx = np.random.permutation(X_perm.shape[0])
            X_perm[:, t, f] = X_perm[perm_idx, t, f]

            perm_pred = model.predict(X_perm)
            perm_mae = mean_absolute_error(y_test, perm_pred)

            importance_matrix[t, f] = perm_mae - baseline_mae

    plt.figure(figsize=(12,6))
    plt.imshow(importance_matrix, aspect='auto')
    plt.colorbar(label="MAE Increase")
    plt.xlabel("Features")
    plt.ylabel("Time Steps")
    plt.xticks(range(F), feature_names, rotation=45)
    plt.title("Feature-Time Importance Heatmap")
    plt.tight_layout()
    plt.show()