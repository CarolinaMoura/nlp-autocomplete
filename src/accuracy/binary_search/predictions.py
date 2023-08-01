import torch
import string
from helpers.Token import Token


class Get_predictions:
    punctuation = string.punctuation + '[PAD]'

    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model

        self.tokens: list[Token] = []
        for i in range(tokenizer.vocab_size):
            char = tokenizer.decode(i)
            self.tokens.append(Token(i, char))

    def next_k(text_sentence: str, top_k: int) -> list[str]:
        pass

    def __get_next_tokens__(self, sentence: str) -> list[tuple(int, str)]:
        """
        Args:
            sentence: sentence from which to predict
                      the next word.
        Returns:
            list of pairs of next tokens and their
            associated probabilities, sorted from
            most probable to least probable.
        """
        # facilitate typing
        tokenizer = self.tokenizer
        model = self.model

        # mask end of the sentence
        sentence += tokenizer.token

        # get encoded sequence
        input_ids = tokenizer.encode(
            sentence,
            add_special_tokens=True,
            truncation=True,
            max_length=500,
        )
        input_ids = torch.tensor(input_ids)

        # get index of the masked token
        mask_idx = torch.where(input_ids == tokenizer.mask_token_id)[1].tolist()[0]

        # get the list of the probabilities
        with torch.no_grad():
            probs = model(input_ids)[0]
            probs = probs[0, mask_idx]
            probs = probs.softmax(dim=-1)
            probs = probs.tolist()

        pairs = list(zip(probs, self.tokens[:]))
        pairs.sort(key=lambda tup: -tup[0])

        return pairs

    def __binary_search_step__(self, sentence: str, min_prob: int) -> list[tuple(str, int)]:
        """
        Args:
            sentence: sentence from which to draw the next
                      token.
            min_prob: minimum probability that the word
                      has to have.
        Returns:
            list of tuples of words and their probabilities,
            and all of those probabilities will be bigger than
            or equal to 'min_prob'.
        """
        next_tokens: list[tuple(int ,str)] = self.__get_next_tokens__(sentence)
        all_drawn_words: set[str] = set()
        return_list: list[tuple(str, int)] = []

        def add_to_return_list(prob: int, token: str):
            if token in all_drawn_words:
                return 
            all_drawn_words.add(token)
            return_list.append((prob, token))

        for prob, next_token in next_tokens:
            if next_token in Get_predictions.punctuation:
                add_to_return_list(prob, '')
            


if __name__ == "__main__":
    from transformers import BertForMaskedLM, BertTokenizer

    model = BertForMaskedLM.from_pretrained("bert-base-multilingual-cased")
    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
