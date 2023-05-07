import os
import requests
import pandas as pd
import random
import json
import time
import seaborn as sns
import matplotlib.pyplot as plt


class OAAPI:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = 'https://api.originality.ai/api/v1'
    def text_predict(self, document):
        url = f'{self.base_url}/scan/ai'
        headers = {
            'X-OAI-API-KEY': self.api_key,
            'Content-Type': 'application/json'
        }
        data = {
            "title": "optional title",
            "content": document
        }
        response = requests.post(url, headers=headers, json=data)
        return response.json()
# Credits for this code go to https://github.com/Haste171/gptzero

GZA = OAAPI("----API from Originality.AI----")
df = pd.read_csv("AI_Detector\GPT-wiki-intro.csv")
df = df.loc[df["generated_intro_len"] > 100]
sampled_index = random.sample(range(0, len(df)), 200)
human_asset = df["wiki_intro"]
generated_asset = df["generated_intro"]


true_positive = 0
false_positive = 0
true_negative = 0
false_negative = 0
start = time.time()
iter = 0
for i in sampled_index:
    print(f"iteration {iter}")
    iter+=1
    positive_res = GZA.text_predict(str(human_asset.iloc[i]))
    negative_res = GZA.text_predict(str(generated_asset.iloc[i]))
    if len(positive_res) < 2 or len(negative_res) < 2:
        continue
    if positive_res["score"]["original"] > positive_res["score"]["ai"]:
        true_negative +=1
    else:
        false_positive += 1

    if negative_res["score"]["ai"] > negative_res["score"]["original"]:
        true_positive +=1
    else:
        false_negative += 1
end = time.time()

print(f"True_positive: {true_positive}\nFalse_positive: {false_positive}\nTrue_negative: {true_negative}\nFalse_negative: {false_negative}\n")

cm_data = {"Actual":[true_positive/(true_positive+false_negative), false_positive/(false_positive+true_negative)], "Predicted":[false_negative/(true_positive+false_negative), true_negative/(false_positive+true_negative)]}
df = pd.DataFrame(cm_data)
sns.heatmap(df, annot=True, fmt='.4f', xticklabels=["AI Generated", "Human Written"], yticklabels=["AI Generated", "Human Written"])
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.title("Confusion Matrix from Experiment")
plt.show()