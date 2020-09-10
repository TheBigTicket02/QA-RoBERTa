from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import torch

from dataset import TweetDataset
from train import calculate_jaccard_score

def create_loader(tweet: str, sentiment: str):
    
    df = pd.DataFrame({'text': tweet, 'sentiment': sentiment}, index=[1])
    
    test_dataset = TweetDataset(
        tweet=df.text.values,
        sentiment=df.sentiment.values,
        selected_text=df.text.values
    )
    
    test_data_loader = DataLoader(
        test_dataset,
        batch_size=1
    )
    return test_data_loader

def predict(data_loader, model, device=torch.device('cpu')):
    final_output = []
    with torch.no_grad():
        for _, d in enumerate(data_loader):
            ids = d["ids"]
            token_type_ids = d["token_type_ids"]
            mask = d["mask"]
            sentiment = d["sentiment"]
            orig_selected = d["orig_selected"]
            orig_tweet = d["orig_tweet"]
            targets_start = d["targets_start"]
            targets_end = d["targets_end"]
            offsets = d["offsets"].numpy()

            ids = ids.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            targets_start = targets_start.to(device, dtype=torch.long)
            targets_end = targets_end.to(device, dtype=torch.long)

            outputs_start, outputs_end = model(
                ids=ids,
                mask=mask,
                token_type_ids=token_type_ids
            )
            
            for px, tweet in enumerate(orig_tweet):
                selected_tweet = orig_selected[px]
                tweet_sentiment = sentiment[px]
                _, output_sentence = calculate_jaccard_score(
                    original_tweet=tweet,
                    target_string=selected_tweet,
                    sentiment_val=tweet_sentiment,
                    idx_start=np.argmax(outputs_start[px, :]),
                    idx_end=np.argmax(outputs_end[px, :]),
                    offsets=offsets[px]
                )
                final_output.append(output_sentence)
            print(final_output)