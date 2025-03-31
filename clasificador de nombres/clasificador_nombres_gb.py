import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import cross_val_score

def extraer_caracteristicas(nombre):
    """
    Extrae características simples de un nombre.
    En este caso, usamos la última letra y la longitud del nombre.
    """
    return {
        'ultima_letra': nombre[-1].lower(),
        'longitud': len(nombre)
    }

def preparar_datos(nombres):
    """
    Prepara los datos para el entrenamiento del clasificador.
    Crea un DataFrame con características extraídas y etiquetas de género.
    """
    datos_caracteristicas = [(extraer_caracteristicas(n), genero) for n, genero in nombres]
    df_caracteristicas = pd.DataFrame.from_dict(datos_caracteristicas)
    df_caracteristicas.columns = ['caracteristicas', 'genero']

    # Separa características y etiquetas
    X_dict = df_caracteristicas['caracteristicas'].tolist()
    y = df_caracteristicas['genero']

    # Vectoriza las características de diccionario a numéricas
    vectorizer = DictVectorizer(sparse=False)
    X = vectorizer.fit_transform(X_dict)

    return X, y

def entrenar_clasificador_gb(X_train, y_train):
    """
    Entrena un clasificador Gradient Boosting.
    """
    clasificador_gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
    clasificador_gb.fit(X_train, y_train)
    return clasificador_gb

def clasificar_nombre(nombre, clasificador_gb):
    """
    Clasifica un nombre usando el clasificador entrenado.
    """
    caracteristicas_nombre = extraer_caracteristicas(nombre)
    # Necesitamos convertir las características a un formato que el clasificador pueda entender
    # En este ejemplo simple, podemos simplemente pasar el diccionario de características
    prediccion = clasificador_gb.predict([caracteristicas_nombre])
    return prediccion[0]

if __name__ == '__main__':
    # Datos de ejemplo (nombres y géneros)
    nombres_ejemplo = [
        ('Juan', 'masculino'),
        ('Maria', 'femenino'),
        ('Carlos', 'masculino'),
        ('Laura', 'femenino'),
        ('Pedro', 'masculino'),
        ('Sofia', 'femenino'),
        ('Jorge', 'masculino'),
        ('Valentina', 'femenino'),
        ('Luis', 'masculino'),
        ('Isabela', 'femenino')
    ]

    X, y = preparar_datos(nombres_ejemplo)

    # Divide los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entrena el clasificador Gradient Boosting
    clasificador_gb_entrenado = entrenar_clasificador_gb(X_train, y_train)

    # Evalúa el clasificador en el conjunto de prueba
    y_pred = clasificador_gb_entrenado.predict(X_test)
    precision = accuracy_score(y_test, y_pred)
    print(f"Precisión del clasificador en el conjunto de prueba: {precision:.2f}")

    # Validación cruzada
    scores_cv = cross_val_score(clasificador_gb_entrenado, X, y, cv=5)
    print("Puntajes de validación cruzada:", scores_cv)
    print(f"Precisión media de validación cruzada: {scores_cv.mean():.2f}")

    # Ejemplo de clasificación de nombres nuevos
    nombre_prueba_1 = 'Ricardo'
    nombre_prueba_2 = 'Daniela'
    genero_predicho_1 = clasificar_nombre(nombre_prueba_1, clasificador_gb_entrenado)
    genero_predicho_2 = clasificar_nombre(nombre_prueba_2, clasificador_gb_entrenado)

    print(f"El nombre '{nombre_prueba_1}' se clasifica como: {genero_predicho_1}")
    print(f"El nombre '{nombre_prueba_2}' se clasifica como: {genero_predicho_2}")
