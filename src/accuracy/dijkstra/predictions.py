import string
import torch
from dijkstra.helpers.min_heap import Min_heap

def encode(
    tokenizer, text_sentence: str, add_special_tokens=True
) -> tuple[torch.Tensor, int]:
    """
    Args:
        tokenizer: tokenizer to use when masking the input.
        text_sentence: text to tokenize. Should have a single
                       appearance of '<mask>', that will be
                       replaced with the mask token of the
                       tokenizer.
        add_special_tokens: flag to put on the tokenizer options.
    Returns:
        a tuple containing the tokenized text sequence and the
        index of the masked token in the tokenized tensor.
    """

    text_sentence = text_sentence.replace("<mask>", tokenizer.mask_token)

    input_ids = torch.tensor(
        [
            tokenizer.encode(
                text_sentence,
                add_special_tokens=add_special_tokens,
                truncation=True,
                max_length=500,
            )
        ]
    )
    mask_idx = torch.where(input_ids == tokenizer.mask_token_id)[1].tolist()[0]
    return input_ids, mask_idx


def create_node_class(tokenizer, model, text_sentence: str, device: str):
    # id of a period in the tokenizer
    period_id: int = tokenizer.encode(".")[1]

    class Token:
        IGNORE_TOKENS = string.punctuation + "[PAD]"

        def __init__(self, id: int):
            self.id = id

            if id < 0:
                self.string = ""
                return

            # decode the id to get the string
            token = tokenizer.decode(id)
            token = token.replace(" ", "")

            if "##" in token:
                token = token.replace("##", "")
            elif token not in Token.IGNORE_TOKENS:
                token = " " + token
            ##

            self.string = token

        def is_punctuation(self) -> bool:
            return self.string in Token.IGNORE_TOKENS

        def __gt__(self, other):
            return self.id > other.id

        def __lt__(self, other):
            return self.id < other.id

    class Node:
        def __init__(
            self,
            parent: "Node",
            token: Token,
            previous_word: str,
            prob: int,
        ):
            self.parent = parent
            self.token = token
            self.previous_word = previous_word
            self.curr_word = previous_word + token.string
            self.prob = prob
            self.next_token_list: list[(int, Token)] | None = None

        def __find_next_token_list__(self):
            """
            Populates self.next_token_list with the token
            probabilities.
            """

            # to find the next token, i'm using my current word
            # and masking the next token
            input_ids, mask_idx = encode(
                tokenizer, f"{text_sentence}{self.curr_word}<mask>"
            )

            # send input id to device
            input_ids = input_ids.to(device)

            with torch.no_grad():
                predict = model(input_ids)[0]
                predict = predict[0, mask_idx]
                predict = predict.softmax(dim=-1)

            token_probs = predict.tolist()
            amount_of_tokens: int = len(token_probs)

            # tokenize all indicies
            indices = [i for i in range(amount_of_tokens)]
            indices.sort(key=lambda ix: -token_probs[ix])

            # create all pairs
            all_pairs = [(token_probs[i], Token(i)) for i in indices]

            all_pairs_filtered = []
            found_punctuation = False

            for pair in all_pairs:
                prob, token = pair

                if self.parent != None and " " in token.string:
                    # i'm not God and i'm trying to predict
                    # a whitespace: same thing as punctuation.
                    # i'll treat this token as if it was
                    # punctuation
                    token = Token(period_id)

                if token.is_punctuation():
                    # i want my model to predict
                    # only one punctuation (the one
                    # with the biggest probability),
                    # because, for all punctuations,
                    # i'll always predict the same previous word
                    if not found_punctuation:
                        all_pairs_filtered.append(pair)
                    found_punctuation = True
                else:
                    all_pairs_filtered.append(pair)

            all_pairs_filtered.reverse()
            self.next_token_list = all_pairs_filtered

        def get_long_neighbor(self):
            if self.next_token_list is None:
                self.__find_next_token_list__()
                
            if not self.next_token_list:
                return None

            next_prob, next_token = self.next_token_list.pop()
            next_prob = next_prob * self.prob

            return Node(self, next_token, self.curr_word, next_prob)

        def is_word_done(self):
            return self.token.is_punctuation()

        def get_short_neighbor(self):
            return self.parent.get_long_neighbor()

        def __gt__(self, other):
            return self.prob < other.prob

        def __lt__(self, other):
            return self.prob > other.prob

    return Node, Token


def get_all_predictions(
    text_sentence, tokenizer, model, top_k=5, verbose=False
) -> list[str]:
    """
    Args:
        text_sentence: text to predict mask element. Can only have ONE masked element.
        tokenizer: tokenizer used with the model
        model: model to finetuned to predict the next word
        top_k: amount of most probable replacements that we want to find.
        verbose: whether or not to print the algorithm's checkpoints
    Returns:
        the top_k most probable words to replace the masked element,
        sorted in order from the most probable to the least probable.
    """
    
    # send model to device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    Node, Token = create_node_class(tokenizer, model, text_sentence, device)
    root = Node(None, Token(-1), "", 1)
    nxt = root.get_long_neighbor()

    done = set()
    probs: list[tuple[int, str]] = []
    min_heap = Min_heap()

    min_heap.add(nxt)
    while len(done) < top_k:
        top = min_heap.top()
        min_heap.pop()

        neigh = top.get_short_neighbor()
        if neigh is not None:
            min_heap.add(neigh)

        if verbose:
            print(
                f"Current biggest prob eh {top.prob} com palavra {top.curr_word}\n"
                + f"Size of the heap {min_heap.size()}"
            )

            print(f"Printing heap:")
            for node in min_heap:
                print(f"{node.curr_word} has prob {node.prob}")
            print()

        if top.is_word_done():
            if top.previous_word in done:
                continue

            done.add(top.previous_word)
            probs.append((top.prob, top.previous_word))

            if verbose:
                print(f"Found a new completion {text_sentence}{top.previous_word}")

            continue

        neigh = top.get_long_neighbor()
        if neigh is not None:
            min_heap.add(neigh)

    probs.sort()
    return [string for prob, string in probs]
