# Api Rest para predicción del modelo

La siguiente aplicación está creada en fastapi y se encarga de cargar un modelo, usar el método post para predecir.

```
/
|--.github
|	|-- workflows
|		|-- test.yml
|--src
|	|-- __init__.py
|	|-- config.py
|	|-- main.py
|	|-- predict.py
|	|-- schema.py
|--tests
|	|-- __init__.py
|	|-- test_code.py
|	|-- test_endpoint.py
|-- Dockerfile
|-- .env
|-- doubleit_model.pt
|-- entrypoint.sh
|-- readme.md
|-- requirements.txt
```


## **GitHub Workflow**


En esta sección se detalla el flujo de trabajo de integración continua (CI) para la aplicación, gestionado a través de GitHub Actions. El proceso comienza con la ejecución de todas las pruebas unitarias, asegurando la integridad del código. Una vez superadas las pruebas, se procede a la creación de la imagen Docker, la cual se almacena en el repositorio.

Es esencial comprender que el flujo de trabajo se describe aquí con fines didácticos. En un entorno de producción, se recomienda crear la imagen solo al realizar un release, garantizando que la imagen se genere cuando esté completamente validada. 


## 2. Sección src (Fuente)

En esta sección se presenta el código de la aplicación desarrollado en FastAPI. Se divide en cuatro componentes principales:

### 2.1 Configuración del Proyecto

En este componente, se maneja la configuración del proyecto, incluyendo la lectura de variables desde el archivo `.env`. En el caso de utilizar una base de datos, se establecen aquí las credenciales como el host y la contraseña.

### 2.2 Main.py

El archivo `main.py` alberga la API principal de la aplicación. Contiene un único método, `post`, que permite la transferencia de valores y realiza la lógica asociada a la carga y uso del modelo. Se implementa un flujo de trabajo riguroso para procesar los datos enviados a la API.

### 2.3 Patrón de Diseño Strategy

La lógica de la aplicación del modelo está en predict, que sigue un patrón de diseño Strategy para garantizar la ejecución consistente de pasos específicos en el flujo de trabajo de los datos enviados a la API. Este enfoque asegura que se ejecuten los mismos pasos para un input dado, produciendo el mismo resultado. Este diseño modular permite la fácil expansión y mantenimiento del sistema.

### 2.4 Uso del Patrón en Ciclo For

El patrón de diseño Strategy se puede combinar de manera efectiva en un ciclo `for`. Si la aplicación cuenta con múltiples modelos con flujos similares, se puede utilizar un ciclo `for` para realizar predicciones de manera eficiente, aplicando el mismo proceso a cada modelo.

Esta sección destaca la flexibilidad y escalabilidad del diseño, permitiendo la expansión y optimización del sistema a medida que se incorporan nuevos modelos o se realizan mejoras en el flujo de trabajo.


### 2.5 Esquema (Schema)

El esquema de la aplicación se encuentra en el módulo `src.schema` y define la estructura de entrada, salida y respuesta de la API. A continuación se presenta el detalle de cada una de estas partes.

#### 1. Esquema de Entrada

El esquema de entrada (`InferenceInput`) especifica la estructura de los datos que la API espera recibir. En este caso, se espera una lista de valores de punto flotante representada por el campo `data`. El uso de `pydantic` y el campo `Field` permite definir reglas y ejemplos para el esquema.

#### 2. Esquema de Salida

El esquema de salida (`InferenceOutput`) define la estructura de los datos que la API devolverá como respuesta. En este caso, se espera una lista de valores de punto flotante representada por el campo `result`. Al igual que en el esquema de entrada, se utiliza `pydantic` y `Field` para establecer reglas y ejemplos.

#### 3. Esquema de Respuesta

El esquema de respuesta (`InferenceResponse`) especifica cómo la API responde a las solicitudes. Incluye un campo booleano `error` para indicar si se produjo un error y un campo opcional `results` que contiene la salida de la inferencia. Se proporcionan ejemplos para clarificar la estructura de la respuesta.

Estos esquemas proporcionan una guía clara sobre la estructura de los datos que la API espera y devuelve, facilitando la interacción con la aplicación y mejorando la consistencia en las comunicaciones.


## 3. Pruebas (Tests)

En esta sección se describen las configuraciones de prueba para la aplicación, abarcando la verificación de la API y las pruebas asociadas al patrón Strategy.

### 3.1 Pruebas de la API

Las pruebas de la API se centran en dos configuraciones principales:

#### 3.1.1 Verificación de Datos en la API

Se ha implementado una configuración de prueba para garantizar que los datos enviados a la API cumplan con el esquema definido. Esto asegura que la entrada sea consistente y siga la estructura especificada en el esquema `InferenceInput`.

#### 3.1.2 Uso de Token `X-API-KEY`

Otra configuración de prueba se enfoca en verificar que las solicitudes a la API estén protegidas mediante un token `X-API-KEY`. Esto asegura que solo las solicitudes autorizadas, que incluyen el token correcto, sean aceptadas por la API.

### 3.2 Pruebas del Patrón Strategy y Flujo de Trabajo

Las pruebas del patrón Strategy y el flujo de trabajo se han diseñado para asegurar que las clases herederas del patrón Strategy cumplan con los patrones establecidos. Estas pruebas verifican que el flujo de trabajo se ejecute correctamente y que la aplicación siga el diseño modular deseado.

Es importante señalar que realizar pruebas unitarias directas a los modelos de machine learning puede ser desafiante debido a la naturaleza compleja y no determinista de los modelos. Sin embargo, las pruebas de estructura pueden realizarse para garantizar que la interfaz entre la lógica de la aplicación y los modelos sea coherente con la estructura del software.
