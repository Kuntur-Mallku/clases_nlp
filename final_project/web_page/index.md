# Análisis Comparativo del Modelo Generativo Meta-Llama-3.1-8B: Implementación de RAG vs. Modelado Tradicional.

## Resumen

Los Large Language Models (LLM) han avanzado significativamente en tareas de procesamiento del lenguaje natural (NLP), como la traducción y la generación de texto
Una de las técnicas emergentes para mejorar el rendimiento de estos modelos es la Retrieval-Augmented Generation (RAG).
Esta técnica combina la generación de texto con la recuperación de información relevante, mejorando la precisión y adecuación contextual de las respuestas.

En este estudio se llevará a cabo un análisis comparativo del modelo generativo de texto Meta-Llama-3.1-8B,
implementando la técnica RAG y sin implementarla. El objetivo es evaluar cómo la incorporación de RAG puede ayudar al modelo LLAMA
a aprender un nuevo idioma que no fue parte de su entrenamiento inicial. A través de esta comparación, se pretende determinar si la técnica
RAG puede integrar el conocimiento de un nuevo lenguaje en el modelo, evitando que el modelo genere respuestas
incorrectas (o "alucine") en un idioma desconocido.

Se realizarán consultas específicas sobre el idioma Kichwa, y los resultados serán evaluados con la métrica BLEU y
por similitud semantica con embeddings para comparar la eficiencia
del modelo con y sin la implementación de RAG. Este estudio brindará una visión integral sobre el uso de RAG en modelos de lenguaje avanzados,
destacando su potencial como alternativa al preentrenamiento para incorporar nuevos conocimientos en modelos de procesamiento de lenguaje natural.

## Introduccion

Los modelos de lenguaje grande (LLM) han logrado un éxito notable, aunque todavía enfrentan limitaciones significativas,
especialmente en tareas específicas de dominio del conocimiento, produciendo notablemente 'alucinaciones'
cuando manejan consultas más allá de sus datos de entrenamiento o requieren información actualizada de los datos.
Para superar estos desafíos, el Retrieval Augmented Generation (RAG) mejora los LLMs recuperando fragmentos relevantes
de documentos de una base de conocimiento externa a través del cálculo de similitud semántica. Al consultar en un base de
conocimiento externo, los LLMs aumentan su efectividad y evitan el 'alucinaciones', de esta manera RAG reduce eficazmente
el problema de generar contenido incorrecto. El trabajo tiene como objetivo proporcionar una comprensión sobre los sistemas RAG
y sus aplicaciones practicas, así como una visión estructurada de los conceptos técnicos fundamentales y la influencia de esta
metodologías en el desarrollo de los LLMs.

## Dataset

Para este estudio se realizo una recopilacion de 130 documentos que tiene texto en Kichwa como diccionarios, libros de aprendizaje Kichwa,
cuentos en Kichwa, La constitucion del Ecuador en Kichwa y otros documentos relacionado al Kichwa. Estos documentos se los dividen en tres grupos de documentacion:

1. Completo: Son documentos que tienen todo su texto solo en el idioma Kichwa
2. Medio: Son documentos de traducion que tienen mitad del contenido en Kichwa y la otra mitad se ecnuentra en Español o Ingles.
3. Basico: son libros de aprendizaje del idioma Kichwa donde la mayoria del texto se enuentra en Español y pequeñas indicaciones se encuentran en Kichwa.
4. Textos en español: Son documentos que tiene todo su contendio en Español sin embargo su contenido es sobre la cultura y el idioma Kichwa.

Se introdujo pocos documentos en español para ver si el modelo es capaz de traducir su contenido al Kichwa.

## Retrieval Augmented Generation - RAG
La Generación Aumentada por Recuperación (RAG) es una técnica utilizada en modelos LLMs, esta técnica combina la generación de texto
con la recuperación de información relevante dentro de grandes bases de datos externas. El objetivo de RAG es mejorar la precisión de las respuestas
generadas por los modelos LLMs accediendo a información específica dentro de un conjunto de documentos, antes de generar una respuesta. Al integrar
datos externos, RAG reduce efectivamente las alucinaciones que podrían producir los modelos LLMs al no saber la respuesta y responder automaticamente.
El sistema de implementación de RAG se lo puede observar en la siguiente grafica.

Fig. 1: Diagrama del sistema de RAG.

En el grafico se puede ver el flujo de la consulta, como antes de responder la consulta accede a documentos de consulta (vectores almacenados)
para que el modelo LLM genere texto a partir de informacion recuperada en el sistema RAG. Esta implementación del sistema RAG consta de 3 partes:
- Indexing
- Retrieval
- Generate

### Indexing

En la primera parte, se descargan los datos que se convertirán en un almacenamiento de vectores, los cuales el modelo LLM consultará antes de generar una respuesta.
Para eso, es crucial dividir los textos en fragmentos, conocidos como chunks. Una vez que los datos están divididos en pequeños fragmentos,
se transforman en embeddings, que son representaciones vectoriales de esos fragmentos. Estas representaciones vectoriales conforman una biblioteca de consulta,
a la que el modelo accederá para generar respuestas. Al finalizar este proceso, obtenemos un almacenamiento de embeddings vectoriales que representa los pequeños
fragmentos de la base de datos de consulta.

### Retrieval

En esta fase, el sistema RAG crea un almacén indexado de embeddings para buscar los vectores más similares a una consulta.
Utilizamos la librería FAISS para crear índices de búsqueda basados en similitud semántica. Cuando el sistema recibe una consulta,
encuentra los vectores más cercanos a la consulta realizada, lo que permite al modelo LLM generar respuestas más precisas.
en esta parte se puede definir cuantos vectores más cercanos se quiere recuperar para generar la respuesta a la consulta.

### Generation

En esta etapa, el sistema RAG interactúa con el modelo LLM para realizar la consulta. Se utiliza un prompt, que puede ser manual o importado;
en este estudio, se creó un prompt manual. Con el prompt generado, se construye una cadena llamada "chain RAG", que asegura que la consulta
pase primero por el sistema RAG antes de ser procesada por el modelo LLM.


Existe algunos tipos de sistema RAG como Naive RAG, Advanced RAG, Modular RAG u otros, según el tipo de sistema de RAG estos tres pasos varían
un poco, se dividen en dos o se implementa un paso extra, sin embargo, estos tres pasos mencionado son lo que define a un sistema RAG. Para el estudio
realizado se implementó un RAG básico solo con estos tres pasos especificados.

## Metodologia

Al descargar la base de datos, se divide cada documento por chunck de 512 de longitud, Esto es debido a que ese es el tamana maximo
de secuencia del modelo embedding usado para la indexacion de fragmentos, por lo que tendremos una base de fragmentos con los 130 documentos en Kichwa,
para la implementación del sistema RAG.

```
def get_file_paths(root_path: str) -> List[str]:
    """Recursively get all file paths from the given root path."""
    return [str(file) for file in Path(root_path).rglob('*') if (file.is_file()) & ('pdfs' in str(file))]

def load_and_split_document(file_path: str, text_splitter: RecursiveCharacterTextSplitter) -> List:
    """Load and split a single document."""
    loader = UnstructuredLoader(file_path, post_processors=[clean_extra_whitespace, group_broken_paragraphs])
    return loader.load_and_split(text_splitter=text_splitter)
def generate_pdf_separators() -> List[str]:
    """
    Generate a list of separators tailored for PDF documents.
    """
    # Basic structural separators
    structural_seps = [
        "\n\n\n", 
        "\n\n",       # Saltos de línea dobles, típicos entre párrafos
        "\f",         # Saltos de página (form feed en PDF)
        "\n\d+\. ",   # Numeración seguida de un punto (por ejemplo, "1. ", "2. ")
        "\n• ",       # Viñetas de listas (como las que aparecen en muchos PDFs)
        "\n- ",       # Separador para guiones (también común en listas)
        "\n===+\n",   # Separadores de líneas largas (pueden aparecer como divisores)
        "\n---+\n",   # Variantes de líneas divisorias
        "\n___+\n",   # Otra variante de líneas divisorias
        "\n",         # Un simple salto de línea para casos en los que no haya una estructura clara
        " ",          # Separación por espacios si no hay mejor opción
        "",           # Cadena vacía para casos residuales
    ]

    # Heading patterns (adjust regex patterns as needed)
    heading_seps = [
        r"\n[\d\.]+\s+[A-Z]",  # Numbered headings: "1. Title", "1.1 Subtitle"
        r"\n[A-Z][a-z]+\s+\d+[\.\:]\s+",  # "Chapter 1: ", "Section 2."
        r"\n[A-Z][A-Z\s]+\n",  # ALL CAPS headings
    ]

    # Page number patterns
    page_num_seps = [
        r"\n\s*\-\s*\d+\s*\-\s*\n",  # Centered page numbers: "- 1 -"
        r"\n\s*\d+\s*\n",            # Simple page numbers: "1"
    ]

    # Combine all separators
    separators = structural_seps + heading_seps + page_num_seps

    return separators

path= os.getcwd()
files= get_file_paths(path)

PDF_SEPARATORS = generate_pdf_separators()
CHUNK_SIZE = 512

text_splitter = RecursiveCharacterTextSplitter(
        separators=PDF_SEPARATORS,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=int(CHUNK_SIZE/10),
        length_function=len,
        add_start_index=True,
        strip_whitespace=True,
        is_separator_regex=False,
)

docs = []
for file in files:
    doc_chunks = load_and_split_document(file, text_splitter)
    docs.extend(doc_chunks)
    print(f"Processed file: {file}")
```

Para la parte de vectorización se usa el modelo de embedding pre-entrenado  'multilingual-e5-large',
este es un modelo pre-enternado que captura la relacion semantica de palabras en diferentes idiomas, por lo que le convierte en un modelo de embedding adecuado para manejar
los documentos en idioma kichwa, español y pocos en ingles del dataset en estudio, en general este modelo transforma texto a embeddings.

```
MODEL_EMBEDDING = 'intfloat/multilingual-e5-large'
device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
embedding_model = HuggingFaceBgeEmbeddings(
                      model_name=MODEL_EMBEDDING,
                      model_kwargs={"device": device},
                      encode_kwargs={"normalize_embeddings": True}, # set True to compute cosine similarity
                  )
vectorstore= FAISS.from_documents(documents=docs, embedding=embedding_model, distance_strategy=DistanceStrategy.COSINE)
retriever = vectorstore.as_retriever(search_kwargs={"k": 100})
```

En esta parte podemos visualizar la distribución del tamaño de los chunks, los chunks ya están vectorizados, su distribución a traves de su tamaño
se los puede ver en la figura de abajo, La mayoría de los documentos en kichwa tienen una longitud de aproximadamente 50 caracteres,
esto significa que los fragmentos son muy pequeños. De esta manera el modelo podría tener poca interpretacion del idioma Kichwa,
la longuitud de los fragmentos puede ser debido a que la mayoria de documentos obtenidos son frases traducidas y no textos completos en Kichwa.

Figura 2

Después de generar los embeddings se debe crear la indexación de estos para el sistema de consultas, esto se lo realiza con la libreria FAISS especializado
de indexación para modelos LLMs, en esta parte también podemos visualizar algunas consultas que se realice con los embeddings
de los fragmentos generados para la base de consulta, esto se lo realiza con una reducción de dimensionalidad con el
modelo de UMAP (Uniform Manifold Approximation and Projection) de los embeddings creados, el resultado se puede observar en la figura de abajo,

Figura 3

Los queries generados se encuentran dentro del conjunto de embeddings de los documentos en Kichwa,
Es interesante notar que los queries en Kichwa e inglés están cercanos entre sí, amientras que el query en español aparece rodeado por los documentos en español.
El gráfico de dispersión (scatter plot) muestra cómo funciona el sistema RAG, recuperando los embeddings más relacionados según la similitud semántica.
En esta etapa, se puede ajustar el número de embeddings más cercanos a recuperar. Para este estudio, se ha establecido en 100,
debido a que los chunks que representan los embeddings son muy pequeños, como se observó en figuras anteriores.

Una vez generado la base de indexación de los embeddings, es momento de llamar al modelo pre-entrenado Meta-Llama-3.1-8B para generar texto en la parte de consultas.

```
reader_model_name = "meta-llama/Meta-Llama-3.1-8B"
tokenizer = AutoTokenizer.from_pretrained(reader_model_name)
model = AutoModelForCausalLM.from_pretrained(reader_model_name
                                             ,use_auth_token= hf_token
                                             ,device_map="auto")
text_generation_pipeline = pipeline(
    model=model,
    tokenizer=tokenizer,
    task="text-generation",
    temperature=0.2,
    do_sample=True,
    repetition_penalty=1.1,
    return_full_text=True,
    max_new_tokens=400,
    pad_token_id = tokenizer.eos_token_id
)
llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
```

Además, es correcto generar un prompt de consultas para que nuestro modelo genere texto encima del prompt y tener una salida de resultados ordenada.

```
prompt_template = """
<|context|>
Answer the question based on your knowledge. Use the following context to help:
{context}
<|question|>
{question}
<|answer|>
"""
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template,
)
```
De este modo, el modelo incluirá tanto la pregunta consultada como su respuesta en el prompt. Gracias al sistema RAG,
también se pueden añadir los documentos que respaldan la respuesta, que corresponden a los embeddings más cercanos a la consulta,
y se colocan en la variable 'context'. Después de llamar al modelo y generar el prompt, se crea la cadena RAG,
permitiendo que la salida del modelo ajuste el prompt y las respuestas queden listas. Así, se completan los tres pasos del sistema RAG.
```
llm_chain = prompt | llm | StrOutputParser()
rag_chain = {"context": retriever, "question": RunnablePassthrough()} | llm_chain
```

## Evaluacion
Para comparar el modelo Meta-Llama-3.1-8B sin RAG y con RAG, se usa la metric BLEU (Bilingual Evaluation Understudy) y similitud semantica con embeddings,
estas métricas evalúan la proximidad de las respuestas de los modelos a las respuestas correctas, principalmente estas métricas
fueron usadas para evaluar proximidad en traducción de palabra, sin embargo, podemos usarla para ver cuál es el modelo que
se acerca más a la respuesta correcta. Para esto se creó 10 preguntas relacionadas con los documentos de la base de datos.
Los resultados se los puede ver en la tabla.

## Discusiones

- Con RAG no fue necesariamente realizar un preentrenamieto al modelo LLM, simplemente con implementar
la técnica RAG mejoramos y actualizamos las respuestas del modelo, puede resultar beneficioso por
recursos computaciones porque es menos costoso crear un sistema RAG en comparación de pre-entrenar al modelo
con nuevos documentos.
- El rendimiento del modelo pude mejorar entre más documentos
de consulta tenga, esta biblioteca externa del modelo puede ser tan grande como los parámetros del
modelo pre-entrando, por lo tanto, almacenar e indexar se vuelta tan complejo como modelos de deep learning.
- Como existen varios modelos del sistema RAG, se debe analizar cuál es el sistema que conviene para el modelo,
puede ser un modelo que trabaje con consultas recursivas u otras modificaciones extras que pueden ayudar al
modelo en las consultas realizadas.

## Referencias

1. M. Malec. (2024). “LARGE LANGUAGE MODELS: CAPABILITIES, ADVANCEMENTS, AND LIMITATIONS [2024]”. hatchworksAI.
2. I. Ilin. (2024). “Advanced rag techniques: an illustrated overview”.
3. Zhao, W. X., Zhou, K., Li, J., Tang, T., Wang, X., Hou, Y., Min, Y., Zhang, B., Zhang, J., Dong, Z., Du, Y., Yang, C., Chen, Y., Chen, Z.,
Jiang, J., Ren, R., Li, Y., Tang, X., Liu, Z., Liu, P., Nie, J.-Y., Wen, J.-R. (2023) “A Survey of Large Language Models”.
4. Li, T., Chen, S., Yang, T., Jiang, H., Wang, C., Fu, Y., Xie, Z., Wang, J., Zheng, X., Zhang, Y. (2023). “ Siren’s Song in the AI Ocean: A Survey
on Hallucination in Large Language Models”.
5. Albert, Q., Sablayrolles, A., Mensch, A., Bamford, Chris., Singh, Devendra., De las Casas, D., Bressand, Florian., Lengyel, G., Lample,
G., Saulnier, L., Renard, L., Lachaux, M., Stock, P., Le T., Lavril, T., Wang, T., Lacroix, T., El Sayed, William. (2023), “Mistral 7B”.
6. Hoshi, Y., Miyashita, D., Ng, Y., Tatsuno, K., Morioka, Y., Torii, O., Deguchi, J. (2024). ”Retrieval-Augmented Generation for Large
Language Models: A Survey”.
7. Xiao, S., Liu, Z., Zhang, P., Muennighoff, N., (2023). ”C-Pack: Packaged Resources To Advance General Chinese Embedding”. Huggingface.
8. Roucher, A. ”Advanced RAG on Hugging Face documentation using LangChain”. Hugging Face.
9. Papineni, K., Roukos, S., Ward, T., Zhu, W. J. (2002). ”BLEU: a method for automatic evaluation of machine translation”.
