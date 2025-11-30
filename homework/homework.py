# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Lue debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Descompone la matriz de entrada usando componentes principales.
#   El pca usa todas las componentes.
# - Escala la matriz de entrada al intervalo [0, 1].
# - Selecciona las K columnas mas relevantes de la matrix de entrada.
# - Ajusta una red neuronal tipo MLP.
#
#
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#

# Importar librerias
import pandas as pd
import gzip
import pickle
import json
from sklearn.pipeline import Pipeline # type: ignore
from sklearn.compose import ColumnTransformer # type: ignore
from sklearn.preprocessing import OneHotEncoder # type: ignore
from sklearn.feature_selection import SelectKBest, f_classif # type: ignore
from sklearn.decomposition import PCA # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore
from sklearn.neural_network import MLPClassifier # type: ignore
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix # type: ignore
from sklearn.model_selection import GridSearchCV # type: ignore
import os
from glob import glob 

# Cargar los datos
def load_data():

    df_test = pd.read_csv(
        "./files/input/test_default_of_credit_card_clients.csv",
        index_col=False,
    )

    df_train = pd.read_csv(
        "./files/input/train_default_of_credit_card_clients.csv",
        index_col = False,
    )

    return df_train, df_test

# Limpiar todos los datos
def clean_data(df):
    df_copy = df.copy()
    df_copy = df_copy.rename(columns={'default payment next month' : "default"})
    df_copy = df_copy.drop(columns=["ID"])
    df_copy = df_copy.loc[df["MARRIAGE"] != 0]
    df_copy = df_copy.loc[df["EDUCATION"] != 0]
    df_copy["EDUCATION"] = df_copy["EDUCATION"].apply(lambda x: 4 if x >= 4 else x)
    df_copy = df_copy.dropna()
    return df_copy

# Dividir los datasets
def split_data(df):
    #X , Y
    return df.drop(columns=["default"]), df["default"]
    
    
# Definir pipeline
def create_pipeline(x_train):
    categorical = ["SEX", "EDUCATION", "MARRIAGE"]
    numerical = [col for col in x_train.columns if col not in categorical]

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(), categorical),
            ('scaler', StandardScaler(), numerical),
        ]
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ('feature_selection', SelectKBest(score_func=f_classif)),  
        ('pca', PCA()),
        ('classifier', MLPClassifier(max_iter=15000, random_state=21))
    ])

    return pipeline

# Optimizar hiperparametros
def estimator(pipeline):
    param_grid = {
        "pca__n_components": [None],
        "feature_selection__k": [20],
        "classifier__hidden_layer_sizes": [(50, 30, 40, 60)],
        "classifier__alpha": [0.26],
        'classifier__learning_rate_init': [0.001],
    }

    grid_search = GridSearchCV(
        estimator=pipeline,           
        param_grid=param_grid,        
        cv=10,                        
        scoring='balanced_accuracy',
        n_jobs=-1,
        refit=True  
    )

    return grid_search
    
# Guardar nuevos modelos
def output(output_directory):
    if os.path.exists(output_directory):
        for file in glob(f"{output_directory}/*"):
            os.remove(file)
        os.rmdir(output_directory)
    os.makedirs(output_directory)

def save(path, estimator):
    output("files/models/")  

    with gzip.open(path, "wb") as f: 
        pickle.dump(estimator, f) 
        

# Funcion para calcular metricas
def metrics(dataset_type, y_true, y_pred):
    """Calculate metrics"""
    return {
        "type": "metrics",
        "dataset": dataset_type,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
    }
    
# Funcion para calcular confusion
def confusion(dataset_type, y_true, y_pred):
    """Confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    return {
        "type": "cm_matrix",
        "dataset": dataset_type,
        "true_0": {"predicted_0": int(cm[0][0]), "predicted_1": int(cm[0][1])},
        "true_1": {"predicted_0": int(cm[1][0]), "predicted_1": int(cm[1][1])},
    }

# Funcion principal
def _run_jobs():
    data_train, data_test = load_data()
    data_train = clean_data(data_train)
    data_test = clean_data(data_test)
    x_train, y_train = split_data(data_train)
    x_test, y_test = split_data(data_test)
    
    pipeline_model = create_pipeline(x_train)  # Se cambia pipeline() por create_pipeline()
    estimator_model = estimator(pipeline_model)
    estimator_model.fit(x_train, y_train)

    save(
        os.path.join("files/models/", "model.pkl.gz"),
        estimator_model,
    )

    y_test_pred = estimator_model.predict(x_test)
    test_precision_metrics = metrics("test", y_test, y_test_pred)
    y_train_pred = estimator_model.predict(x_train)
    train_precision_metrics = metrics("train", y_train, y_train_pred)

    test_confusion_metrics = confusion("test", y_test, y_test_pred)
    train_confusion_metrics = confusion("train", y_train, y_train_pred)

    os.makedirs("files/output/", exist_ok=True)

    with open("files/output/metrics.json", "w", encoding="utf-8") as file:
        file.write(json.dumps(train_precision_metrics) + "\n")
        file.write(json.dumps(test_precision_metrics) + "\n")
        file.write(json.dumps(train_confusion_metrics) + "\n")
        file.write(json.dumps(test_confusion_metrics) + "\n")

if __name__ == "__main__":
    _run_jobs()