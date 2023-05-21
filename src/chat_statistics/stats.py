from typing import Union
from pathlib import Path
import json

from hazm import word_tokenize, Normalizer
from wordcloud import WordCloud
import arabic_reshaper
from bidi.algorithm import get_display
import matplotlib.pyplot as plt
from src.data import DATA_DIR


class ChatStatistics:
    def __init__(self, chat_json: Union[str, Path]):

        #load chat data
        with open(chat_json) as f:
            self.chat_data = json.load(f)
        
        #load stop words
        self.normalizer = Normalizer()
        stop_words = open(DATA_DIR / 'stopwords.txt').readlines()
        stop_words = list(map(str.strip, stop_words))
        self.stop_words = list(map(self.normalizer.normalize, stop_words))
 
    def generate_word_cloud(self, output_dir: Union[str, Path]):
        text_contant = ''

        for msg in self.chat_data['messages']:
            if type(msg['text']) is str:
                tokens = word_tokenize(msg['text'])
                tokens = list(filter(lambda item: item not in self.stop_words, tokens))
                text_contant += f"{' '. join(tokens)}"
        
        # normalize, reshape for final word cloud
        text_contant = self.normalizer.normalize(text_contant)
        text_contant = arabic_reshaper.reshape(text_contant)
        text_contant = get_display(text_contant)

        # generate word cloud
        wordcloud = WordCloud(
            font_path=str(DATA_DIR / 'B Homa_0.ttf'),
            background_color='white'
        ).generate(text_contant)

        wordcloud.to_file(str(Path(output_dir) / 'wordcloud.png'))

if __name__ == "__main__":
    chat_stats = ChatStatistics(chat_json=DATA_DIR / 'Python_OG.json')
    chat_stats.generate_word_cloud(output_dir=DATA_DIR)

    print('Done!')
