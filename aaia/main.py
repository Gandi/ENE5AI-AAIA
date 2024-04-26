from aaia.helpers import (
    load_dataframe,
    save_dataframe,
    load_model)
from aaia.time_series import (
    TimeSeriesModel,
    train as train_ts,
    predict,
    plot)
from aaia.environment import ResourceAllocationEnv
from aaia.agent import (
    Policy,
    PPOAgent,
    train as train_agent)
import os
import numpy as np
import torch


# THROUGHOUT THE SCRIPT, MODELS, PLOTS, AND INTERMEDIATE DATAFRAMES ARE
# SAVED/LOADED. IF YOU WISH TO DISABLE THIS, REMOVE THE SAVE/LOAD FUNCTION
# CALLS AND PATHS.

# INSERT DIRECTORY PATHS
input_dir = ''
output_dir = ''

df_train_path = os.path.join(output_dir, 'df_train.csv')
df_test_path = os.path.join(output_dir, 'df_test.csv')
df_predicted_path = os.path.join(output_dir, 'df_predicted.csv')
df_allocated_path = os.path.join(output_dir, 'df_allocated.csv')

model_ts_path = os.path.join(output_dir, 'model_ts.pth')
model_policy_path = os.path.join(output_dir, 'model_policy.pth')

plot_ts_cpu_path = os.path.join(output_dir, 'ts_cpu.png')
plot_ts_ram_path = os.path.join(output_dir, 'ts_ram.png')
plot_agent_path = os.path.join(output_dir, 'agent_training_reward.png')

used_column_cpu = 'CPU Used (%)'
used_column_ram = 'RAM Used (GB)'
requested_column_cpu = 'CPU Requested (%)'
requested_column_ram = 'RAM Requested (GB)'
predicted_column_cpu = 'CPU Predicted (%)'
predicted_column_ram = 'RAM Predicted (GB)'
allocated_column_cpu = 'CPU Allocated (%)'
allocated_column_ram = 'RAM Allocated (GB)'

target_columns = [used_column_cpu, used_column_ram]
predicted_columns = [predicted_column_cpu, predicted_column_ram]

# Data preprocessing / loading
df_train = load_dataframe(df_train_path)
df_test = load_dataframe(df_test_path)


# Param init
max_cpu = df_test[requested_column_cpu].max()
max_ram = df_test[requested_column_ram].max()
max_resources = [max_cpu, max_ram]

#   Time series
lookback = 5
n_epochs = 40
batch_size = 8

#   Agent
episodes = 1000
min_time_steps = 10
max_time_steps = 100
coef_margin_cpu = 0.1
coef_margin_ram = 0.2


# Time series training
model_ts = train_ts(df_train=df_train,
                    df_test=df_test,
                    target_columns=target_columns,
                    lookback=lookback,
                    n_epochs=n_epochs,
                    batch_size=batch_size,
                    output_path=model_ts_path)

plot(df_train=df_train,
     df_test=df_test,
     model=model_ts,
     target_columns=target_columns,
     lookback=lookback,
     y_max=max_resources,
     title='Prediction',
     x_label='Time step',
     output_paths=[plot_ts_cpu_path, plot_ts_ram_path])


# Time series prediction
model_ts = load_model(ModelClass=TimeSeriesModel,
                      filepath=model_ts_path,
                      model_params=target_columns)


def predict_ts(model,
               df,
               target_columns,
               predicted_columns,
               lookback,
               fill_values):
    predictions = predict(model=model,
                          df=df,
                          target_columns=target_columns,
                          lookback=lookback)
    for i in range(len(target_columns)):
        shift = np.ones(lookback) * fill_values[i]
        prediction = np.concatenate([shift, predictions[:, i]])
        df[predicted_columns[i]] = np.ceil(prediction).astype(int)
    return df


df_train_agent = df_train.copy(deep=True)  # Agent training data
df_train_agent = predict_ts(model=model_ts,
                            df=df_train_agent,
                            target_columns=target_columns,
                            predicted_columns=predicted_columns,
                            lookback=lookback,
                            fill_values=max_resources)

df = df_test.copy(deep=True)  # Test data
df = predict_ts(model=model_ts,
                df=df,
                target_columns=target_columns,
                predicted_columns=predicted_columns,
                lookback=lookback,
                fill_values=max_resources)
save_dataframe(df, df_predicted_path)


# Agent training
used_cpu_data = df_train_agent[used_column_cpu].to_list()
predicted_cpu_data = df_train_agent[predicted_column_cpu].to_list()
used_ram_data = df_train_agent[used_column_ram].to_list()
predicted_ram_data = df_train_agent[predicted_column_ram].to_list()

env = ResourceAllocationEnv(requested_cpu=max_cpu,
                            requested_ram=max_ram,
                            coef_margin_cpu=coef_margin_cpu,
                            coef_margin_ram=coef_margin_ram)

model_agent = train_agent(env=env,
                          episodes=episodes,
                          min_time_steps=min_time_steps,
                          max_time_steps=max_time_steps,
                          used_cpu_data=used_cpu_data,
                          predicted_cpu_data=predicted_cpu_data,
                          used_ram_data=used_ram_data,
                          predicted_ram_data=predicted_ram_data,
                          model_policy_path=model_policy_path,
                          plot_path=plot_agent_path)

# Agent execution
model_policy = load_model(ModelClass=Policy,
                          filepath=model_policy_path,
                          model_params=env)
model_agent = PPOAgent(env)
model_agent.policy = model_policy

df[allocated_column_cpu] = np.nan
df[allocated_column_ram] = np.nan
df.iloc[0, df.columns.get_loc(allocated_column_cpu)] = max_cpu
df.iloc[0, df.columns.get_loc(allocated_column_ram)] = max_ram


def df_to_state_tensor(df, i_row):
    df_row = df.iloc[i_row]
    state_vec = np.array([df_row[requested_column_cpu],
                          df_row[requested_column_ram],
                          df_row[used_column_cpu],
                          df_row[used_column_ram],
                          df_row[allocated_column_cpu],
                          df_row[allocated_column_ram],
                          df_row[predicted_column_cpu],
                          df_row[predicted_column_ram]])
    return torch.FloatTensor(state_vec)


total_reward = 0
env.reset()

for i in range(1, df.shape[0]):
    state = df_to_state_tensor(df, i - 1)

    action, _ = model_agent.select_action(state,
                                          coef_margin_cpu=coef_margin_cpu,
                                          coef_margin_ram=coef_margin_ram)

    df.iloc[i, df.columns.get_loc(allocated_column_cpu)] =\
        action['CPU']['Allocation']\
        if action['CPU']['Reallocating']\
        else df.iloc[i - 1][allocated_column_cpu]
    df.iloc[i, df.columns.get_loc(allocated_column_ram)] =\
        action['RAM']['Allocation']\
        if action['RAM']['Reallocating']\
        else df.iloc[i - 1][allocated_column_ram]

    _, reward, _ = env.step(
        action=action,
        next_used_cpu=df.iloc[i][used_column_cpu],
        next_used_ram=df.iloc[i][used_column_ram],
        next_predicted_cpu=df.iloc[i][predicted_column_cpu],
        next_predicted_ram=df.iloc[i][predicted_column_ram])
    total_reward += reward

df[allocated_column_cpu] = df[allocated_column_cpu].astype(int)
df[allocated_column_ram] = df[allocated_column_ram].astype(int)

save_dataframe(df, df_allocated_path)

print('====== Test ======')
print(f'Time steps: {df.shape[0] - 1}, Total reward: {total_reward}')
