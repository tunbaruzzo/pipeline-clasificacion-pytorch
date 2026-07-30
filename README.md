# Pipeline de clasificación en PyTorch

## Descripción

Este proyecto fue desarrollado como primera pre-entrega del proyecto final del curso de Deep Learning y NLP de CODERHOUSE.

El objetivo principal fue construir un pipeline base de entrenamiento, validación y evaluación utilizando PyTorch. Para esta etapa se priorizó la organización del proceso, la reproducibilidad y la correcta implementación del ciclo de entrenamiento, antes que la complejidad del modelo.

## Dataset

Se utilizó el dataset Iris, disponible en Scikit-learn.

El dataset contiene 150 observaciones correspondientes a tres especies de flores:

- setosa;
- versicolor;
- virginica.

Cada observación incluye cuatro variables numéricas:

- largo del sépalo;
- ancho del sépalo;
- largo del pétalo;
- ancho del pétalo.

Las tres clases se encuentran balanceadas, con 50 observaciones por especie, y no se encontraron valores faltantes.

## División de los datos

Los datos fueron divididos de manera estratificada en:

- entrenamiento: 105 registros;
- validación: 22 registros;
- prueba: 23 registros.

Las variables fueron estandarizadas mediante `StandardScaler`.

El escalador fue ajustado únicamente con los datos de entrenamiento para evitar data leakage. Luego, la misma transformación fue aplicada a los conjuntos de validación y prueba.

## Arquitectura del modelo

Se implementó una red neuronal multicapa utilizando `nn.Module` y `nn.Sequential`.

La arquitectura utilizada fue:

- capa de entrada con 4 variables;
- primera capa oculta de 16 neuronas;
- función de activación ReLU;
- segunda capa oculta de 8 neuronas;
- función de activación ReLU;
- capa de salida con 3 valores, uno por cada especie.

El modelo posee 243 parámetros entrenables.

## Configuración del experimento

- Python: 3.12.13
- PyTorch: 2.11.0+cpu
- Dispositivo utilizado: CPU
- Semilla: 42
- Batch size: 16
- Learning rate: 0.001
- Cantidad de épocas: 50
- Función de pérdida: CrossEntropyLoss
- Optimizador: Adam

## Entrenamiento y validación

El ciclo de entrenamiento fue implementado de manera explícita e incluye:

1. reinicio de gradientes mediante `optimizer.zero_grad()`;
2. forward pass;
3. cálculo de la pérdida;
4. backward pass mediante `loss.backward()`;
5. actualización de pesos mediante `optimizer.step()`.

Después de cada época, el modelo fue evaluado sobre el conjunto de validación sin actualizar sus parámetros, utilizando `model.eval()` y `torch.no_grad()`.

Se registraron la pérdida y el accuracy de entrenamiento y validación por cada época.

## Resultados

Al finalizar las 50 épocas se obtuvieron los siguientes resultados:

| Conjunto | Loss | Accuracy |
|---|---:|---:|
| Entrenamiento | 0.2218 | 92.38 % |
| Validación | 0.2900 | 90.91 % |
| Prueba | 0.3059 | 86.96 % |

La mejor época según la pérdida de validación fue la época 50.

## Interpretación

Durante las 50 épocas se pudo observar que el modelo fue aprendiendo de manera progresiva. La pérdida de entrenamiento y validación fue disminuyendo, mientras que el accuracy fue aumentando.

Los resultados de entrenamiento y validación fueron cercanos y la pérdida de validación continuó disminuyendo hasta la última época. Por este motivo, no se observaron señales importantes de overfitting.

En el conjunto de prueba, el modelo clasificó correctamente 20 de las 23 observaciones.

Todas las flores de la especie setosa fueron clasificadas correctamente. Los tres errores se produjeron entre las especies versicolor y virginica.

En general, el modelo logró aprender a diferenciar las tres especies y obtuvo un buen desempeño sobre datos no vistos.

## Estructura del repositorio

```text
pipeline-clasificacion-pytorch/
│
├── data/
│   └── README.md
│
├── notebooks/
│   ├── README.md
│   └── pre_entrega_pipeline_pytorch.ipynb
│
├── outputs/
│   ├── metricas_entrenamiento.csv
│   └── modelo_iris.pth
│
├── .gitignore
├── README.md
└── requirements.txt
