#%% Temperatur-Vergleich
import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True))

MODEL_NAME = 'llama-3.3-70b-versatile'

# Kreative Aufgabe zum Testen
prompt = "Schreibe eine kurze, kreative Metapher über Künstliche Intelligenz."

print("=" * 60)
print("TEMPERATUR 0 (deterministisch)")
print("=" * 60)
model_temp0 = ChatGroq(model_name=MODEL_NAME, temperature=0, api_key=os.getenv('ki-chrizgl-learn'))
for i in range(3):
    res = model_temp0.invoke(prompt)
    print(f"\nVersuch {i+1}: {res.content}\n")

print("=" * 60)
print("TEMPERATUR 1.0 (kreativ)")
print("=" * 60)
model_temp1 = ChatGroq(model_name=MODEL_NAME, temperature=1.0, api_key=os.getenv('ki-chrizgl-learn'))
for i in range(3):
    res = model_temp1.invoke(prompt)
    print(f"\nVersuch {i+1}: {res.content}\n")