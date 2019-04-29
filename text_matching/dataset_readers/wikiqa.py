from typing import Dict
import csv

from overrides import overrides

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField, Field
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.tokenizers.word_splitter import JustSpacesWordSplitter
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer


@DatasetReader.register("wikiqa")
class WikiQADatasetReader(DatasetReader):
    """
    Reads a file from the WikiQA dataset. The train/validation/test split of the data
    comes from the paper `WikiQA: A Challenge Dataset for Open-Domain Question Answering`
    
    Each file of the data is a tsv file with header. 
    The columns are: QuestionID, Question, DocumentID, DocumentTitle, SentenceID, Sentence, Label.
    
    All questions are pre-tokenized and tokens are space separated. We convert these keys into
    fields named "label", "premise" and "hypothesis", so that it is compatible to some existing
    natural language inference algorithms.
    """
    def __init__(self,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer(JustSpacesWordSplitter())
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path):
        with open(file_path, "r") as data_file:
            tsv_in = csv.reader(data_file, delimiter='\t')
            next(tsv_in, None)  # skip the header
            for row in tsv_in:
                if len(row) == 7:
                    yield self.text_to_instance(premise=row[1], hypothesis=row[5], label=row[6])

    @overrides
    def text_to_instance(self,  # type: ignore
                         premise: str,
                         hypothesis: str,
                         label: str = None) -> Instance:
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}
        tokenized_premise = self._tokenizer.tokenize(premise)
        tokenized_hypothesis = self._tokenizer.tokenize(hypothesis)
        fields["premise"] = TextField(tokenized_premise, self._token_indexers)
        fields["hypothesis"] = TextField(tokenized_hypothesis, self._token_indexers)
        if label is not None:
            fields['label'] = LabelField(label)

        return Instance(fields)
