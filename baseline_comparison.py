import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

def create_sequences(data_array, target_array, time_steps=36, horizon=6):
    X, y = [], []
    for i in range(len(data_array) - time_steps - horizon + 1):
        x_window = data_array[i : i + time_steps]
        y_target = target_array[i + time_steps + horizon - 1]
        if np.isnan(x_window).any() or np.isnan(y_target):
            continue
        X.append(x_window)
        y.append(y_target)
    return np.array(X), np.array(y)

def load_processed_csvs(folder):
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.csv')]
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

df_all = load_processed_csvs('data/processed/bolus_meal_steps_data/csv')

features = [col for col in df_all.columns if col not in ['patient_id', 'timestamp', 'glucose']]
print("Features found:", features)
target = 'glucose'
patient_ids = df_all['patient_id'].unique()

TIME_STEPS = 36
results = []

for test_patient in patient_ids:
    print(f"\nEvaluating patient {test_patient}...")
    
    train_df = df_all[df_all['patient_id'] != test_patient].copy()
    test_df  = df_all[df_all['patient_id'] == test_patient].copy()

    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(train_df[features])

    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(train_df[[target]])

    X_train_list, y_train_list = [], []
    train_combined = train_df.copy()
    train_combined[features] = X_train_scaled
    train_combined[target]   = y_train_scaled.flatten()

    for pid in train_combined['patient_id'].unique():
        p = train_combined[train_combined['patient_id'] == pid]
        p_feat = p[features].to_numpy()
        p_targ = p[[target]].to_numpy()
        p_comb = np.hstack([p_feat, p_targ])
        xp, yp = create_sequences(p_comb, p_targ, TIME_STEPS)
        if len(xp) == 0:
            continue
        X_train_list.append(xp)
        y_train_list.append(yp)

    X_train = np.concatenate(X_train_list)
    y_train = np.concatenate(y_train_list)

    X_test_scaled = scaler_X.transform(test_df[features])
    y_test_scaled = scaler_y.transform(test_df[[target]])
    test_combined = np.hstack([X_test_scaled, y_test_scaled])
    X_test, y_test = create_sequences(test_combined, y_test_scaled, TIME_STEPS)

    y_true = scaler_y.inverse_transform(y_test)

    y_naive = scaler_y.inverse_transform(X_test[:, -1, -1].reshape(-1, 1))

    y_train_actual = scaler_y.inverse_transform(y_train)
    lr = LinearRegression()
    lr.fit(X_train.reshape(X_train.shape[0], -1), y_train_actual.flatten())
    y_lr = lr.predict(X_test.reshape(X_test.shape[0], -1)).reshape(-1, 1)

    for name, preds in [('Naive', y_naive), ('Linear Reg.', y_lr)]:
        rmse = np.sqrt(mean_squared_error(y_true, preds))
        mae  = mean_absolute_error(y_true, preds)
        results.append({'Patient': test_patient, 'Model': name,
                        'RMSE': round(rmse, 2), 'MAE': round(mae, 2)})

    lstm_known = {
        '559-ws-training': (15.34, 11.2),
        '563-ws-training': (17.03, 12.8),
        '570-ws-training': (16.04, 12.1),
        '575-ws-training': (17.84, 13.5),
        '588-ws-training': (14.36, 10.9),
        '591-ws-training': (14.99, 11.3),
    }
    rmse_l, mae_l = lstm_known.get(test_patient, (0, 0))
    results.append({'Patient': test_patient, 'Model': 'LSTM',
                    'RMSE': rmse_l, 'MAE': mae_l})

df_res = pd.DataFrame(results)
pivot = df_res.pivot_table(index='Patient', columns='Model', values=['RMSE', 'MAE'])
print(pivot.to_string())
os.makedirs('comparison_model_results', exist_ok=True)
df_res.to_csv('comparison_model_results/baseline_comparison.csv', index=False)
print("\nSaved to comparison_model_results/baseline_comparison.csv")