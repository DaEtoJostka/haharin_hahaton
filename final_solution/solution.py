import typing as tp
import pandas as pd
import re
from transformers import BertTokenizer, BertForTokenClassification
import torch
from torch import cuda

class Pipeline:
    def __init__(self):
        self.type = None

    def data_preparation(self, data: pd.DataFrame) -> pd.DataFrame:
        '''
        param data contain DataFrame we need to clear and split
        return DataFrame with cleared and splited data
        '''
        all_tokens: list[str] = []
        for i in range(data.shape[0]):
            text: str = data.loc[i]['description'].replace('\n', ' ')
            text = re.sub(r'\?{2,}', '', text)
            text = re.sub(r'\u200b', '', text)
            text = re.sub(r'\[^ ]*', '', text)
            text = re.sub(r'@\S{2,}', '', text)
            text = re.sub(r'http\S{2,}', '', text)
            text = re.sub(r'\s{2,}', ' ', text)
            text = re.sub(r'"{2,}', '"', text)

            tokens: list[str] = []
            word: int = 0
            st: str = ''
            pos: int = 0
            for j in text:
                # if symbol is digit or letter, create a word
                if j.isdigit() or j.isalpha():
                    word = 1
                    st += j
                # if symbol is space, add formed word to list and skip space
                elif j == ' ':
                    if word:
                        tokens.append(st)
                        st = ''
                        word = 0
                # if any other symbol, add formed word to list and symbol as well
                else:
                    if word: tokens.append(st)
                    tokens.append(j)
                    st = ''
                    word = 0

                pos += 1

            all_tokens.append('repthing'.join(tokens))
        result = pd.DataFrame()
        result['tokens'] = all_tokens
        return result

    def prediction(self, data: pd.DataFrame) -> list[list]:
        '''
        param data contains DataFrame where vacancies description is placed.
        returns result of prediction for each vacancy.
        '''
        data_prepared: pd.DataFrame = self.data_preparation(data)
        model = Model()
        all_predictions: list = []
        for i in data_prepared['tokens'].tolist():
            all_predictions.append(model.predict(i))

        return all_predictions
    
class Model:
    def __init__(self):
        '''
        loading (if need it) models from HuggingFace and initialization of tokenizer and model
        '''
        self.tokenizer = BertTokenizer.from_pretrained('DeepPavlov/rubert-base-cased-conversational')
        self.model = torch.load('model_for_hackaton.pt', map_location=torch.device('cpu'))
        self.model.to(device)
        self.model.eval()

    def predict(self, sentence) -> list:
        '''
        param sentence contains single vacancy description.
        returns list of founded skills by model. List is DTObject named SkillsPrediction.
        '''
        sentence_split: str = sentence.split('repthing')
        inputs = self.tokenizer(sentence_split,
                                padding='max_length',
                                truncation=True,
                                max_length=MAX_LEN,
                                return_tensors="pt",
                                is_split_into_words=True)
        ids = inputs["input_ids"].to(device)
        mask = inputs["attention_mask"].to(device)

        with torch.no_grad():
            outputs = self.model(ids, mask)
        logits = outputs[0]

        active_logits = logits.view(-1, 3)  # shape (batch_size * seq_len, num_labels)
        flattened_predictions: torch.Tensor = torch.argmax(active_logits, axis=1)  # shape (batch_size*seq_len,) - predictions at the token level

        tokens: str | list[str] = self.tokenizer.convert_ids_to_tokens(ids.flatten().tolist())
        token_predictions: list[str] = [id2label[i] for i in flattened_predictions.cpu().numpy()]
        wp_preds: list = list(zip(tokens, token_predictions))  # list of tuples. Each tuple = (wordpiece, prediction)

        word_level_predictions: list[str] = []
        for pair in wp_preds:
            if (pair[0].startswith("##")) or (pair[0] in ['[CLS]', '[SEP]', '[PAD]']):
                # skip prediction
                continue
            else:
                word_level_predictions.append(pair[1])

        # delete all I-marks which not start with B-mark
        '''
        for i in range(len(word_level_predictions)):
            if (i > 0 and word_level_predictions[i] == 'I-skill' and word_level_predictions[i - 1] == 'O' or (i == 0 and word_level_predictions[i] == 'I-skill')):
                word_level_predictions[i] = 'O'
        '''

        # remake I-mark into B-mark if distance is more than 1
        if word_level_predictions[0] == 'I-skill': word_level_predictions[0] = 'B-skill'
        for i in range(len(word_level_predictions)):
            if i > 2 and word_level_predictions[i] == 'I-skill' and word_level_predictions[i - 1] == 'O' and word_level_predictions[i - 2] == 'O':
                word_level_predictions[i] = 'B-skill'

        # compare I-mark to end of last skill, if distance is equal to 1
        for i in range(len(word_level_predictions)):
            if i > 2 and word_level_predictions[i] == 'I-skill' and word_level_predictions[i - 1] == 'O' and (word_level_predictions[i - 2] == 'I-skill' or word_level_predictions[i - 2] == 'B-skill'):
                word_level_predictions[i - 1] = 'I-skill'


        test_text: list[str] = []
        for i in sentence_split:
            if i != '': test_text.append(i)

        # merging tokens into sentences which contain skills
        found_skills: list[str] = []
        it_was_here: int = 0
        skill: str = ''

        for i in range(len(word_level_predictions)):
            if word_level_predictions[i] == 'B-skill':
                if it_was_here != 0:
                    found_skills.append(skill[:-1])
                    skill = ''
                skill += test_text[i] + ' '
                it_was_here = 1
            elif word_level_predictions[i] == 'I-skill':
                skill += test_text[i] + ' '
            elif it_was_here != 0:
                found_skills.append(skill[:-1])
                skill = ''
                it_was_here = 0
        if skill != '': found_skills.append(skill)

        return found_skills
    
MAX_LEN: int = 512
device: str = 'cuda' if cuda.is_available() else 'cpu'
label2id: dict = {'O': 0, 'B-skill': 1, 'I-skill': 2}
id2label: dict = {0: 'O', 1: 'B-skill', 2: 'I-skill'}

EntityScoreType = tp.Tuple[int, float]  # (entity_id, entity_score)
MessageResultType = tp.List[EntityScoreType]  # list of entity scores

# Assuming CSV file named "company_names.csv"
df = pd.read_csv("final_solution\company_names2.csv", sep=";")

def process_mentions(mentions):
  """
  This function processes a list of mentions and returns a list of corresponding company indices with duplicates removed.
  """
  company_indices = []
  for mention_list in mentions:
    indices = []
    for mention in mention_list:
      # Find rows where any of the synonyms match the current mention
      matching_rows = df[df['l_syns'].str.contains(mention, case=False)]
      indices.extend(matching_rows['issuerid'].tolist())
    # Remove duplicates while preserving order
    unique_indices = list(dict.fromkeys(indices))
    company_indices.append(unique_indices)
  return company_indices

def score_texts(
    messages: tp.Iterable[str], *args, **kwargs
) -> tp.Iterable[MessageResultType]:
    """
    Main function (see tests for more clarifications)
    Args:
        messages (tp.Iterable[str]): any iterable of strings (utf-8 encoded text messages)
    Returns:
        tp.Iterable[tp.Tuple[int, float]]: for any messages returns MessageResultType object
    -------
    Clarifications:
    >>> assert all([len(m) < 10 ** 11 for m in messages]) # all messages are shorter than 2048 characters
    """
    pipeline = Pipeline()
    predictions = pipeline.prediction(messages)
    mentions = process_mentions(predictions)
    results = []
    for mention_list in mentions:
      if mention_list:
        results.append(mention_list)
      else:
        results.append([])

    return results