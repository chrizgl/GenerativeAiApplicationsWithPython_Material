#%% packages
from sympy import true
from transformers import pipeline

#%% data
feedback = [
    "I recently bought the EcoSmart Kettle, and while I love its design, the heating element broke after just two weeks. Customer service was friendly, but I had to wait over a week for a response. It's frustrating, especially given the high price I paid.",
    "Die Lieferung war super schnell, und die Verpackung war großartig! Die Galaxy Wireless Headphones kamen in perfektem Zustand an. Ich benutze sie jetzt seit einer Woche, und die Klangqualität ist erstaunlich. Vielen Dank für ein tolles Einkaufserlebnis!",
    "Je ne suis pas satisfait de la dernière mise à jour de l'application EasyHome. L'interface est devenue encombrée et le chargement des pages prend plus de temps. J'utilise cette application quotidiennement et cela affecte ma productivité. J'espère que ces problèmes seront bientôt résolus."
]

candidate_labels = ['defect', 'delivery', 'interface']

# %% function
def process_feedback(feedbackList):
    sentiment = pipeline("text-classification", model="nlptown/bert-base-multilingual-uncased-sentiment")
    sentimentResult = sentiment(feedbackList)
    classifier = pipeline("zero-shot-classification", model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")
    classResult = classifier(feedbackList, candidate_labels, multi_label=False)
    result = zip(feedbackList, [x['label'] for x in sentimentResult], [x['labels'][0] for x in classResult])
    return result

#%% Test

print("Processing customer feedback...\n")
results = process_feedback(feedback)
for feedbackText, sentimentRes, classRes in results:
    print(f"Feedback: {feedbackText}")
    print(f"Sentiment: {sentimentRes}")
    print(f"Classes: {classRes}")
    print("\n")
