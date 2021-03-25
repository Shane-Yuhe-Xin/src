import data_utils
import pandas as pd
import torch
import unittest

from transformers import RobertaTokenizerFast


class TestDataUtils(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
        self.dataset = pd.DataFrame.from_dict(
            {
                "question": ["question 0", "question 1"],
                "passage": ["passage 0", "passage 1"],
                "idx": [0, 1],
                "label": [True, False],
            }
        )
        self.max_seq_len = 4

    def test_sample(self):
        ## An example of a basic unit test, using class variables initialized in
        ## setUpClass().
        self.assertEqual(self.max_seq_len, 4)

    def test_encode_data(self):
        encode = data_utils.encode_data(self.dataset, self.tokenizer, self.max_seq_len)
        self.assertEqual(list(encode[0].shape), [len(self.dataset), self.max_seq_len])
        self.assertEqual(list(encode[1].shape), [len(self.dataset), self.max_seq_len])
        self.assertEqual(encode[0].dtype, torch.long)
        self.assertEqual(encode[1].dtype, torch.long)

    def test_extract_labels(self):
        ## TODO: Write a unit test that asserts that extract_labels() outputs the
        ## correct labels, e.g. [1, 0].
        test = data_utils.extract_labels(self.dataset)
        self.assertEqual(test, [1,0])

if __name__ == "__main__":
    unittest.main()
