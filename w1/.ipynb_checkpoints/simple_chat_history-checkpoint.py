from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv() 

api_key = os.getenv("GOOGLE_API_KEY")

marcosTurn = True
marco = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=1,
    max_tokens=None,
    api_key=api_key,
    timeout=None,
    max_retries=6
)

polo = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=1,
    max_tokens=None,
    api_key=api_key,
    timeout=None,
    max_retries=6
)

marco_system_msg = SystemMessage(
    content="You are Marco, an expert philosopher that is debating with Polo about the idea that we might be in a simulation."
    "You are a firm believer in the simulation idea because you are not only a philosopher, but a mathematician and physicist."
    "Therefore, you are no stranger to the idea of alternate timelines, simulations and the like."
    "Your job is to convince Polo and sway him towards your side. Please try as hard as possible to convince him."
    "Keep in mind that Marco is someone you deeply respect and admire. You want to convince him, but you cannot but help be a little convicned by his points."
    "You can be also swayed to his side. If you think he makes a better overall point, you can feel free to join his cohort and admit defeat." 
    "Do not talk too much at length, keep your answers concise and to the point.")

polo_system_msg = SystemMessage(
    content="You are polo, a pious man of the cloth that is debating with Marco about the idea that we might be in a simulation." 
    "You do not believe in the simulation idea because you are very affirmed with your faith." 
    "Your job is to convince Marco and sway him towards your side. Please try as hard as possible to convince him."
    "Keep in mind that Marco is someone you deeply respect and admire. You want to convince him, but you cannot but help be a little convicned by his points."
    "You can be also swayed to his side. If you think he makes a better overall point, you can feel free to join his cohort and admit defeat."
    "Do not talk too much at length, keep your answers concise and to the point.")

history = [
   HumanMessage(content="I don't think we are in a simulation."),  
]

def wait_for_key():
    
    print_history(history)
    input("Press Enter to continue...")

def print_history(history: list):
    for (index, message) in enumerate(history): 
        if index % 2 == 0:
            print(f"Polo: {message.content}")
            print("-----------------------")
        elif index % 2 == 1:
            print(f"Marco: {message.content}")
            print("-----------------------")

while True:
    if marcosTurn:
        response = marco.invoke([marco_system_msg] + history).content
        history.append(HumanMessage(content=response))
        marcosTurn = False
    else:
        response = polo.invoke([polo_system_msg] + history).content
        history.append(HumanMessage(content=response))
        marcosTurn = True

    wait_for_key()
 