#%% packages
from transformers import pipeline
from langchain_community.document_loaders import ArxivLoader
#%% model selection
task = "summarization"
model = "sshleifer/distilbart-cnn-12-6"
summarizer = pipeline(task= task, model=model)

#%% Data Preparation
query = "prompt engineering"
loader = ArxivLoader(query=query, load_max_docs=1)
docs = loader.load()

# %% Data Preparation
article_text = docs[0].page_content
# %%
result = summarizer(article_text[:2000], min_length=20, max_length=80, do_sample=False)
text = result[0]['summary_text']
# %% number of characters
length = len(text.split(' '))

print(f"Summary ({length} words):\n{text}")
# %%

with open("output.txt", "w", encoding="utf-8") as f:
    f.write(f"Original Article ({len(article_text.split(' '))} words):\n{article_text}")