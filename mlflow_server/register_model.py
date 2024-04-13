import os
import mlflow
import pandas as pd
import psycopg2 as psycopg

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from category_encoders import CatBoostEncoder
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from catboost import CatBoostRegressor

from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error



os.environ["DB_DESTINATION_HOST"] = os.getenv("DB_DESTINATION_HOST")
os.environ["DB_DESTINATION_PORT"] = os.getenv("DB_DESTINATION_PORT")
os.environ["DB_DESTINATION_NAME"] = os.getenv("DB_DESTINATION_NAME")
os.environ["DB_DESTINATION_USER"] = os.getenv("DB_DESTINATION_USER")
os.environ["DB_DESTINATION_PASSWORD"] = os.getenv("DB_DESTINATION_PASSWORD")

os.environ["MLFLOW_S3_ENDPOINT_URL"] = "https://storage.yandexcloud.net" #endpoint бакета от YandexCloud
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID") # получаем id ключа бакета, к которому подключён MLFlow, из .env
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY") # получаем ключ бакета, к которому подключён MLFlow, из .env

# определяем глобальные переменные
# поднимаем MLflow локально
TRACKING_SERVER_HOST = "127.0.0.1"
TRACKING_SERVER_PORT = 5000


registry_uri = f"http://{TRACKING_SERVER_HOST}:{TRACKING_SERVER_PORT}"
tracking_uri = f"http://{TRACKING_SERVER_HOST}:{TRACKING_SERVER_PORT}"

mlflow.set_tracking_uri(tracking_uri)

# название тестового эксперимента и запуска (run) внутри него
EXPERIMENT_NAME = "estate_project"
RUN_NAME = "baseline"
REGISTRY_MODEL_NAME = "estate_model"


connection = {"sslmode": "require", "target_session_attrs": "read-write"}
postgres_credentials = {
    "host": os.environ["DB_DESTINATION_HOST"], 
    "port": os.environ["DB_DESTINATION_PORT"],
    "dbname": os.environ["DB_DESTINATION_NAME"],
    "user": os.environ["DB_DESTINATION_USER"],
    "password": os.environ["DB_DESTINATION_PASSWORD"],
}
assert all([var_value != "" for var_value in list(postgres_credentials.values())])

connection.update(postgres_credentials)

TABLE_NAME = "clean_real_estate"

with psycopg.connect(**connection) as conn:
    with conn.cursor() as cur:
        cur.execute(f"SELECT * FROM {TABLE_NAME}")
        data = cur.fetchall()
        columns = [col[0] for col in cur.description]

df = pd.DataFrame(data, columns=columns)

target_col = 'price'
features = df.drop(columns=['flat_id','building_id', 'price'])
num_features = features.select_dtypes(include=['int64', 'float64']).drop(columns=['building_type_int'])
cat_features = features.select_dtypes(include=['object'])
cat_features = pd.concat([cat_features, df['building_type_int']], axis=1)

potential_bin_features = cat_features.nunique() == 2
bin_features = cat_features[potential_bin_features[potential_bin_features].index]
cat_features = cat_features[potential_bin_features[~potential_bin_features].index]

num_features_cols = num_features.columns.tolist()
bin_features_cols = bin_features.columns.tolist()
cat_features_cols = cat_features.columns.tolist()

preprocessor = ColumnTransformer(
    [
        
        ('binary', OneHotEncoder(drop='if_binary'), bin_features_cols),
        ('numeric', StandardScaler(), num_features_cols),
        ('categorial', CatBoostEncoder(), cat_features_cols),
    ],
    verbose_feature_names_out=False,
    remainder='drop'
)

model = CatBoostRegressor() 

pipeline = Pipeline(
    [
        ('preprocessor', preprocessor),
        ('model', model)
    ]
)

X_train, X_test, y_train, y_test = train_test_split(df, df[target_col], test_size=0.3, random_state=3)

pipeline.fit(X_train, y_train)
prediction = pipeline.predict(X_test)

metrics = {}
mae = mean_absolute_error(y_test, prediction).round(2)
mse = mean_squared_error(y_test, prediction).round(2)
mape = mean_absolute_percentage_error(y_test, prediction).round(2)
metrics["mae"] = mae
metrics["mse"] = mse
metrics["mape"] = mape

pip_requirements = "requirements.txt"
metadata =  {'model_type': 'estate_regression'}


experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
with mlflow.start_run(run_name=RUN_NAME, experiment_id=experiment_id) as run:
    # получаем уникальный идентификатор запуска эксперимента
    run_id = run.info.run_id 
    
    model_info = mlflow.sklearn.log_model( 
			sk_model=model,
            pip_requirements=pip_requirements,
            metadata=metadata,
            await_registration_for=60,
            artifact_path="models",
            registered_model_name=REGISTRY_MODEL_NAME)

    mlflow.log_metrics(metrics) 
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

run = mlflow.get_run(run_id) 
assert (run.info.status =='FINISHED')