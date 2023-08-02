import pandas as pd
import torch
from transformers import BertTokenizer, BertForMaskedLM
from tqdm import tqdm
import threading
import logging
from dijkstra.predictions import get_all_predictions
import datetime


# --------- args here ---------
test_dataset_path = '../../data/test.csv'
model_checkpoint_name = '../../carolina/random_tokens/best_model.pt'
top_k = 5
# -----------------------------

logging.getLogger("transformers").setLevel(logging.ERROR)

df = pd.read_csv(test_dataset_path)
sentences = df['sentence']
last_words = df['last_word']

correct = [0] * top_k
total_samples = [0]

def run_samples(samples: list[tuple[str, str]], device: str) -> list[int]:
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    model = BertForMaskedLM.from_pretrained('bert-base-multilingual-cased')
    model.load_state_dict(torch.load(model_checkpoint_name))

    loop = tqdm(samples, total=len(samples), leave=True)

    for sent, word in loop:
        suggestions = get_all_predictions(sent, tokenizer, model, top_k, False, device)

        total_samples[0] += 1

        try:
            # gotta add this extra space because
            # the prediction of a new word also
            # predics a preceding whitespace
            ix = suggestions.index(' '+word)
        except:
            continue

        correct[ix] += 1

zip1 = zip(sentences[:10], last_words[:10])
zip2 = zip(sentences[10:20], last_words[10:20])

t1 = threading.Thread(target=run_samples, args=(list(zip1), 'cuda:0'))
t2 = threading.Thread(target=run_samples, args=(list(zip2), 'cuda:1'))

start_time = datetime.datetime.now()

t1.start()
t2.start()

t1.join()
t2.join()

tot = 0

end_time = datetime.datetime.now()
time_difference = end_time - start_time

print(f'Took {time_difference} to finish.')

for ix, val in enumerate(correct):
    tot += val
    print(f'Top {ix+1}: {tot}/{total_samples[0]} = {tot/total_samples[0]}')