#%% packages
import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True))
# %%
# Model overview: https://console.groq.com/docs/models
MODEL_NAME = 'llama-3.3-70b-versatile'
model = ChatGroq(model_name=MODEL_NAME,
                   temperature=1.0, # controls creativity
                   api_key=os.getenv('ki-chrizgl-learn'))

# %% Run the model
res = model.invoke("Was hat Albert Einstain in seinem 40 Lebensjahr gemacht?")
# %% find out what is in the result
res.model_dump()
# %% only print content
print(res.content)
# %%
