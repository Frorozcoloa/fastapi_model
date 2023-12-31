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

Es importante señalar que realizar pruebas unitarias directas a los modelos de machine learning puede ser desafiante debido a la naturaleza compleja y no determinista de los modelos. Sin embargo, las pruebas de estructura pueden realizarse para garantizar que la interfaz entre la lógica de la aplicación y los modelos sea coherente con la estructura del software. Aunque estás se interpretan más como pruebas de integración

Es importante entender que en la investigación se describe el uso de pruebas morfologicas, pero que aun no han sido probadas.

## 4 Configuración de Entorno (.env) y Estrategias para CI/CD Continuo

La configuración de entorno es crucial para la ejecución exitosa de la aplicación, y actualmente, el archivo `.env` contiene dos variables importantes: `API_KEY` y `MODEL_PATH`. Para garantizar un flujo de CI/CD continuo y eficiente, se pueden explorar otras estrategias y herramientas.

### 4.1 Configuración de Entorno

En el archivo `.env`, se encuentran las siguientes variables:

* `API_KEY`: Clave de API necesaria para autenticar las solicitudes a la API.
* `MODEL_PATH`: Ruta del modelo utilizado por la aplicación.

### 4.2 Estrategias para CI/CD Continuo

#### 4.2.1 Uso de Jenkins u Otras Herramientas CI/CD

Considerando la necesidad de un flujo de CI/CD continuo, se puede integrar Jenkins u otras herramientas CI/CD en el proceso. Estas herramientas permiten automatizar la construcción, prueba y despliegue del software de manera eficiente.

* **Jenkins** : Configurar un pipeline en Jenkins que, al detectar cambios en el repositorio, realice automáticamente la construcción, las pruebas y el despliegue si se cumplen los criterios establecidos.

## 5. Entrypoint y Configuración de Gunicorn para Desplegar FastAPI

El entrypoint es el punto de inicio de ejecución de una aplicación y, en este caso, se utiliza Gunicorn (Green Unicorn) para desplegar FastAPI. El código de entrada (entrypoint) generalmente se encuentra en un archivo que se ejecuta para iniciar el servidor web. A continuación, se presenta un ejemplo de código de entrada para Gunicorn configurado para desplegar una aplicación FastAPI.

## 6. requirements.txt

Para definir los requisitos de la aplicación, se utiliza un archivo llamado `requirements.txt`. Este archivo enumera todas las dependencias y versiones específicas de las bibliotecas que la aplicación necesita para ejecutarse correctamente

## 7. Monitoreo del modelo

1. **Infraestructura Segura:**
   * Establecer un servidor público dentro de la subnet.
   * Restringir el acceso al servidor solo a través de la VPN de la empresa para garantizar un entorno seguro y controlado.
2. **Despliegue de MLflow:**
   * Implementar un servidor de MLflow en el servidor público.
   * Garantizar que el código siempre seleccione el mejor modelo disponible en el repositorio de MLflow.
3. **Tracking del Flujo de Modelos:**
   * Utilizar MLflow para registrar y seguir el flujo de los modelos, incluyendo versiones, métricas de rendimiento y parámetros utilizados.
4. **Detección de Data Drift:**
   * Configurar una alarma en la base de datos para detectar incertidumbre alta en el modelo, sirviendo como señal temprana de posible data drift.
5. **Integración de Evently AI:**
   * Implementar la librería Evently AI para análisis periódicos del modelo con datos nuevos y cálculo de errores.
6. **Dashboard de Monitoreo (Grafana):**
   * Crear un dashboard en Grafana para visualizar informes generados por Evently AI y la evolución del modelo.
7. **Alarmas y Notificaciones:**
   * Establecer alarmas en el sistema que notifiquen al equipo de data scientists ante signos de data drift o incertidumbre inusual.
