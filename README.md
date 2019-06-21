# Clasificador de Preguntas

El objetivo de este proyecto es implementar un algoritmo no supervisado capaz de agrupar preguntas, permitiendo así, dadas un gran número de preguntas, detectar cuales son similares entre ellas. De esta forma, podríamos saber cuales son las preguntas más recurrentes, los temas de interés más relevantes... de un grupo de usuarios. Cree este proyecto por que pensé que sera muy útil para los streaming que realizó en la plataforma youtube donde mis seguidores pueden preguntar sus dudas, en ocasiones en número de preguntas es tan alto que es imposible responder a todas, sin embargo, muchas de las preguntas se repiten constantemente, por lo que esta aplicación podrá ser extremadamente útil. 


README OBSOLETO, SE ACTUALIZARÁ A LA VERSIÓN 2.0 CUANDO TENGA TIEMPO. MIENTRAS TANTO AQUÍ ESTÁ EXPLICADO TODO LO QUE HE HECHO Y LOS RESULTADOS OBTENIDOS (https://github.com/ikergarcia1996/QuestionClustering/releases/download/2.0/Measuring_Question_Similarity_GitHub.pdf)


# ¿Qué se ha hecho hasta ahora?

Ahora mismo hay dos funciones principales que permiten extraer información de un grupo de preguntas:
* Clustering: Aplica el algoritmo kmeans sobre las preguntas transformadas a una representación vectorial.
* k_nearest: Dada una pregunta, responde con las k preguntas más cercanas usando como métrica la similitud coseno.

A parte de esto se han implementado funciones para obtener todas las preguntas de un chat de youtube, y un gestor de preguntas que normaliza las preguntas (tokenización, eliminación de palabras poco relevantes, eliminación de símbolos y emoticonos...), calcula las palabras que más se repiten y transforma las preguntas a representaciones vectoriales. Para esto último, se calcula la representación vectorial para una pregunta como la media de los vectores que representan a las palabras que la forman (Tras normalizar la pregunta). Las representaciones vectoriales de palabras se han obtenido aplicando el algoritmo FastText (https://fasttext.cc/) sobre un corpus generado a partir de texto extraído de webs de noticias tecnológicas españolas usando el crawler https://github.com/fhamborg/news-please


# Direcciones futuras

De momento, el algoritmo no proporciona resultados de la calidad esperada, por lo que son necesarias mejoras. Algunas ideas son:
* Modificar el corpus de entrenamiento de los word embeddings para que los word embeddings sean capaces de representar conocimiento del tipo "El 1700X es un procesador Ryzen, el 8700K es un procesador intel, la 2060 es una GPU de Nvidia..." puesto que parece que esa clase de conocimiento no se ha capturado. Una manera de hacer esto es modificando la ventana usada por los algoritmos de generación de word embeddings.

* Modificar el algoritmo de clustering para usar como medida la similitud coseno. Parece que la función k_nearest si obtiene los resultados esperados, es posible que el clustering de mejores resultados si en vez de usar la implementación por defecto del kmeans (sklearn) usaremos un algoritmo que use como distancia la similitud coseno. 

* Entrenar sentence embeddings en vez de word embeddings, quizá buscar la forma de aplicar encoders como BERT, ELMO... a este problema.

* Añadir información que conviertan el problema en una tarea supervisada o semi-supervisada, por ejemplo generar un dataset donde dadas dos frases se diga si son similares o no. El problema es que esto llevaría demasiado tiempo...

# Autores
```
Iker García Ferrero - ikergarcia1996
```

Toda ayuda es bienvenida, si quieres aportar al proyecto te animo a hacerlo, ya sea haciendo un pull request con código que hayas implementado como aportando ideas para mejorar el algoritmo!!!

# ¿Qué hay en este directorio?
* QuestionCluster.ipynb: Jupyter Notebook con los algoritmos para clasificar preguntas y descargar preguntas desde un chat de youtube.
* embedding.py: Código python3 que se encarga de cargar word embeddings y manipularlos (por ejemplo incluye funciones como word_to_vector que dada una palabra devuelve el vector que la representa)
* vocabulary.py: Gestor del vocabulario de un word embeddings, es una dependencia de embedding.py
* utils.py: Algunas funciones útiles para diferentes tareas, es una dependencia de embedding.py
* json_to_text.py: Transoforma el output (archivos json) de news-please (https://github.com/fhamborg/news-please) a un único archivo txt que contiene solo el texto de las noticias descargadas por news-please
* leidas.txt: Durante el directo de youtube donde se implementó la primera versión de este programa se descargaron una gran cantidad de preguntas realizadas por los espectadores. El programa descargaba el chat completo cada 0.2 segundos, por lo que las preguntas se repiten decenas de veces, una vez eliminadas las duplicadas hay 309 preguntas.
* test_questions.txt: 68 preguntas recopiladas a mano de este directo: https://youtu.be/aRKSRGDva84

# Word Embeddings

Para ejecutar el código necesitas word embeddings. He generado unos en español especializados en el dominio tecnológico. Para ello usando news-please (https://github.com/fhamborg/news-please) he descargado noticias de las principales webs de noticias tecnológicas en español y he aplicado dos algoritmos de generación de word embeddings. FastText (https://fasttext.cc) y GloVe (https://nlp.stanford.edu/projects/glove/). En ambos casos se han usado los parámetros por defecto del algoritmo y todas las letras del corpus se han transformado a minúsculas. 

* Embeddings generados mediante FastText (OneDrive): https://1drv.ms/u/s!AqTsNQJK2z6Lhdgu_8rat2jMbTa2bA
* Embeddings generados mediante Glove (OneDrive): https://1drv.ms/t/s!AqTsNQJK2z6LhdgvOS8_WjhgtE4TCg

# ¿Qué puedo hacer con este código? 

Lo que quieras :D Eres libre de hacer lo que te de la gana con él, usarlo en tus programas, hacer un nuevo programa basado en él, modificarlo como quieras, redistribuirlo... Lo único que te pido es que cites a todas las personas que hayan aportado su granito de arena al proyecto, se incluirán todas en la sección autores que se encuentra un poco más arriba. Cualquier programa derivado, copia, modificación... de este programa debe usar la licencia MIT al igual que todo el contenido de este github, para más información consulta el archivo LICENSE (https://github.com/ikergarcia1996/QuestionClustering/blob/master/LICENSE)

