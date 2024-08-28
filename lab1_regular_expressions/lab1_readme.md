# Laboratorio 1 - Regular Expressions

En esta carpeta hay dos archivos, un archivo .py y otro archivo .ipynb, esto es debido a que la funcion se desarrollo en notebook sin embargo la interaccion con el chatbot es mejor desde el terminal espor eso que se creo el archivo `lab1_basic_chatbot.py`

el programa consta de una funcion `chatbot_response(user_input, question_count)` en el cual recibe como parametros la consulta del usuario y un intero como flag de las veces que esta usando el chat.

la parte en que se usa regular expresion es en la compilacion del saludo, weather, stock, seguir preguntando y despedida:

```
greeting_pattern = re.compile(r"^[^a-z]*(hi|[h']?ello|hey|good\s(morn[gin']{0,3}|afternoon|even[gin']{0,3}))", re.IGNORECASE)
weather_pattern = re.compile(r"\b(weather|forecast|temperature)\b", re.IGNORECASE)
stock_market_pattern = re.compile(r"\b(stock|market|stocks|shares)\b", re.IGNORECASE)
follow_up_pattern = re.compile(r"\b(yes|have|anything else)\b", re.IGNORECASE)
goodbye_pattern = re.compile(r"\b(goodbye|bye|see you)\b", re.IGNORECASE)
```

Se usan las expresiones regulares para guiar las conversaciones con el chatbot.