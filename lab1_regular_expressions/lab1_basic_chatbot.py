import re

greeting_pattern = re.compile(r"^[^a-z]*(hi|[h']?ello|hey|good\s(morn[gin']{0,3}|afternoon|even[gin']{0,3}))", re.IGNORECASE)
weather_pattern = re.compile(r"\b(weather|forecast|temperature)\b", re.IGNORECASE)
stock_market_pattern = re.compile(r"\b(stock|market|stocks|shares)\b", re.IGNORECASE)
follow_up_pattern = re.compile(r"\b(yes|have|anything else)\b", re.IGNORECASE)
goodbye_pattern = re.compile(r"\b(goodbye|bye|see you)\b", re.IGNORECASE)

def chatbot_response(user_input, question_count=0):
    answer = ''
    if question_count >= 2:
        answer= "You've already asked 2 questions. It's time to say goodbye!"
    else:
        if greeting_pattern.search(user_input):
            answer= "Hello! How can I assist you today? You can ask me about the weather or the stock market."
        elif weather_pattern.search(user_input):
            question_count += 1
            if 'weather' in weather_pattern.match(user_input).groups():
                answer= "The weather is sunny and warm today, but you don't forget to bring a sweater because the weather in Quito is crazy. Do you have any other questions?"
            elif 'forecast' in weather_pattern.match(user_input).groups():
                answer= "The forecast said it will rain this afternoon, you don't forget to bring a sweater or umbrella. Do you have any other questions?"
            elif 'temperature' in weather_pattern.match(user_input).groups():
                answer= "I don't know about the temperature outside because I am robot, but my favorite temperature is 25°C. Do you have any other questions?"
            else:
                answer= "I'm sorry, I don't understand about that. You can ask me about the weather or the stock market."
        
        elif stock_market_pattern.search(user_input):
            question_count += 1
            if ('stocks' in stock_market_pattern.match(user_input).groups() or 'shares' in stock_market_pattern.match(user_input).groups()):
                answer= "A stock or share is a small piece of a company that you can buy. If the company does well, your piece can be worth more. Do you have any other questions?"
            elif ('market' in stock_market_pattern.match(user_input).groups()):
                answer= "The stock market is where investors buy and sell shares of companies. It's a set of exchanges where companies issue shares and other securities for trading. Do you have any other questions?"
            elif ('diff' in user_input) and ('stocks' in stock_market_pattern.match(user_input).groups()) and ('shares' in stock_market_pattern.match(user_input).groups()):
                answer= "A stock and a share are essentially one and the same. They both represent a part of the capital of a joint stock company. Do you have any other questions?"
            elif 'stock' in stock_market_pattern.match(user_input).groups():
                answer= "The stock market is up by 5% today. but don't take my word for it; the information might be outdated. Do you have any other questions?"
            else:
                answer= "I'm sorry, I don't understand about that. You can ask me about the weather or the stock market."
        
        elif follow_up_pattern.search(user_input):
            if 'no' in follow_up_pattern.match(user_input).groups() or 'not' in follow_up_pattern.match(user_input).groups():
                answer= "OK. Goodbye! Have a great day!"
            else:
                answer= "Do you have any other questions?"
        
        elif goodbye_pattern.search(user_input):
            answer= "Goodbye! Have a great day!"
        
        else:
            answer= "I'm sorry, I don't know about that. You can ask me about the weather or the stock market."
    
    return question_count, answer


question_count = 0
while True:
    user_input = input("You: ")
    response = chatbot_response(user_input, question_count)
    print(f"Bot: {response[1]}")
    if "goodbye" in response[1].lower() or "great day" in response[1].lower():
        break
    if 'questions' in response[1]:
        question_count += 1