# chatbot  
This chatbot is designed to provide information and answer questions in daily conversation and some domain knowledge related to Michael Jordan. It utilizes various machine learning (ML) and natural language processing (NLP) techniques to handle user interactions, sentiment analysis, and text-based responses. The system is structured to facilitate both information retrieval from a knowledge base and casual conversation, also leveraging sentiment analysis to adapt responses based on user feedback.

## System Description:    
- Knowledge base in particular domain: ("knowledge_base.pickle")    
This chatbot can provide information and answer questions related to Michael Jordan. It utilizes various Natural Language Processing (NLP) techniques to understand user input, retrieve relevant information from a knowledge base created by a webcrawler, and generate appropriate responses.       
  
- Sentiment Analysis Model: ("sentiment_model.ipynb", "sentiment.keras")    
The model was trained by Convolutional Neural Network (CNN) and various Natural Language Processing (NLP) techniques to serve the purpose of sentiment analysis, classifying input messages into different emotion categories such as "happy", "sadness", or "neutral". Through this model, the chatbot can quickly understand the emotion expressed by user.   

- Conversation Model: ("chatbot.py")    
The model is based on a sequence-to-sequence (seq2seq) architecture using Long Short-Term Memory (LSTM) network, which is a special kind of Recurrent Neural Network (RNN) suitable for sequence prediction problems and prevent vanishing gradient problem. This architecture is particularly useful for conversation, where the model needs to generate sequences of text in response to input sequences.   

## Dialog tree or logic:
![image](https://https://github.com/JesseLee62/img-storage/blob/master/chatbot/dialog.jpg) 

## Sample dialog interactions:
![image](https://github.com/JesseLee62/img-storage/blob/master/chatbot/1.jpg)   
![image](https://github.com/JesseLee62/img-storage/blob/master/chatbot/2.jpg)   
![image](https://github.com/JesseLee62/img-storage/blob/master/chatbot/3.jpg)   
![image](https://github.com/JesseLee62/img-storage/blob/master/chatbot/4.jpg)   
![image](https://github.com/JesseLee62/img-storage/blob/master/chatbot/5.jpg)   
![image](https://github.com/JesseLee62/img-storage/blob/master/chatbot/6.jpg)   
![image](https://github.com/JesseLee62/img-storage/blob/master/chatbot/7.jpg)   
![image](https://github.com/JesseLee62/img-storage/blob/master/chatbot/8.jpg)   
![image](https://github.com/JesseLee62/img-storage/blob/master/chatbot/9.jpg)   

## References: 
https://www.kaggle.com/datasets/ishantjuyal/emotions-in-text?resource=download     
https://www.kaggle.com/datasets/kreeshrajani/3k-conversations-dataset-for-chatbot   
