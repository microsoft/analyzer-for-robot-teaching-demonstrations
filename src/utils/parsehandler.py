from nltk.parse.stanford import StanfordDependencyParser
from nltk.tag.stanford import StanfordPOSTagger
from nltk.tag import StanfordNERTagger
from nltk.parse.stanford import StanfordParser
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import nltk
import json


class ParseHandler:
    def __init__(self, text="hello. put the dish here"):
        self.stanfordparser_path = '/stanford/stanford-parser-full-2020-11-17/stanford-parser.jar'
        self.stanfordparser_model_path = '/stanford/stanford-corenlp-4.2.0-models-english.jar'

        self.dep_parser = StanfordDependencyParser(
            path_to_jar=self.stanfordparser_path,
            path_to_models_jar=self.stanfordparser_model_path)
        self.text = text
        self.tokenized_sentences, self.tokenized_sentences_span, self.tokenized_words, self.tokenized_words_span = self.tokenize_raw_text(
            self.text)
        self.parse_result = self.dep_parser.parse_sents(self.tokenized_words)
        # nltk.sent_tokenize(self.text)
        #self.parse_result = self.dep_parser.raw_parse_sents(self.tokenized_sentences)
        self.parse_result_list = []
        for sent in self.parse_result:
            self.dep_graph_conll_str = next(sent).to_conll(10)
            self.parse_result_list.append(
                self.stanfordparse2list(self.dep_graph_conll_str))
        # parse_result_list is a list of sentences, which contain a list of dictionary
        # we assume self.text is not updated. only switching of tokenized words (e.g., here -> table, it -> apple) is allowed.
        # once updated, update the tags uding the following commands.
        self.parse_result_list = self.update_word_global_position(
            self.parse_result_list)
        self.parse_result_list = self.update_sentence_position(
            self.parse_result_list)
        self.parse_result_list = self.update_span_in_text(
            self.parse_result_list)

    def update_text(self, text="hello. put the dish here"):
        self.text = text
        self.tokenized_sentences, self.tokenized_sentences_span, self.tokenized_words, self.tokenized_words_span = self.tokenize_raw_text(
            self.text)
        self.parse_result = self.dep_parser.parse_sents(self.tokenized_words)
        self.parse_result_list = []
        for sent in self.parse_result:
            self.dep_graph_conll_str = next(sent).to_conll(10)
            self.parse_result_list.append(
                self.stanfordparse2list(self.dep_graph_conll_str))
        self.parse_result_list = self.update_word_global_position(
            self.parse_result_list)
        self.parse_result_list = self.update_sentence_position(
            self.parse_result_list)
        self.parse_result_list = self.update_span_in_text(
            self.parse_result_list)

    def print_parse_result_list(self):
        print(json.dumps(self.parse_result_list, indent=4))

    def tokenize_raw_text(self, text):
        tokenizer_sentence = nltk.tokenize.punkt.PunktSentenceTokenizer()
        sentences = tokenizer_sentence.tokenize(text)
        span_sentences = list(tokenizer_sentence.span_tokenize(text))
        tokenizer_word = nltk.TreebankWordTokenizer()
        words = tokenizer_word.tokenize_sents(sentences)
        span_words = list(tokenizer_word.span_tokenize_sents(sentences))
        span_words_offset = []
        for i, item_sentences in enumerate(span_words):
            span_offset = span_sentences[i][0]
            _span_words_offset = []
            for item in item_sentences:
                item_list = list(item)
                _span_words_offset.append(
                    tuple([span_offset + item_list[0], span_offset + item_list[1]]))
            span_words_offset.append(_span_words_offset)

        return sentences, span_sentences, words, span_words_offset

    def get_verb_list(self):
        ret_list = []
        for sentence in self.parse_result_list:
            for item in sentence:
                tag = item["tag"]
                if tag.startswith("V"):
                    ret_list.append(item)
        return ret_list

    def resolve_object_parent(self, parse_result_element):
        idx_position_sentence = parse_result_element["position_sentence"]
        idx_position_head = parse_result_element["position_head"]
        for sentence in self.parse_result_list:
            for item in sentence:
                if item["position_sentence"] == idx_position_sentence and item["position"] == idx_position_head:
                    return item
        return None

    def resolve_object_children(self, parse_result_element, relation="obj"):
        ret_list = []
        for sentence in self.parse_result_list:
            for item in sentence:
                if item["relation"] == relation:
                    parent = self.resolve_object_parent(item)
                    if parent == parse_result_element:
                        ret_list.append(item)
        return ret_list

    def update_word_global_position(self, parse_result_list):
        idx_position_global = 1  # 1 indexed to follow stanfordparser specification
        for sentence in self.parse_result_list:
            for item in sentence:
                item["position_global"] = idx_position_global
                idx_position_global = idx_position_global + 1
        return parse_result_list

    def update_sentence_position(self, parse_result_list):
        idx_position_sentence = 1  # 1 indexed to follow stanfordparser specification
        for sentence in self.parse_result_list:
            for item in sentence:
                item["position_sentence"] = idx_position_sentence
            idx_position_sentence = idx_position_sentence + 1
        return parse_result_list

    def update_span_in_text(self, parse_result_list):
        # traverse span across words in a sentence
        for _idx_sent, sentence in enumerate(self.parse_result_list):
            _idx_word = 0  # 0 indexed to search
            for item in sentence:
                word = item["word"]
                while _idx_word < len(self.tokenized_words_span[_idx_sent]):
                    if word == self.tokenized_words[_idx_sent][_idx_word]:
                        item["span_in_text"] = list(
                            self.tokenized_words_span[_idx_sent][_idx_word])
                        _idx_word = _idx_word + 1
                        break
                    else:
                        _idx_word = _idx_word + 1
        return parse_result_list

    def wordnet_pos_resolver(self, tag):
        if tag.startswith("J"):
            return wordnet.ADJ
        elif tag.startswith("V"):
            return wordnet.VERB
        elif tag.startswith("N"):
            return wordnet.NOUN
        elif tag.startswith("R"):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    def stanfordparse2list(self, dep_graph_conll_str):
        # Input: a dependency graph in CoNLL format
        # '{i}\t{word}\t{lemma}\t{ctag}\t{tag}\t{feats}\t{head}\t{rel}\t_\t_\n'
        # table includes : {i}, {word}, {tag}, {head}, {rel}
        ret_list = []
        dep_items = dep_graph_conll_str.split("\n")
        for item in dep_items:
            item_info = item.split("\t")
            if len(item_info) == 10:
                i_position = item_info[0]
                i_word = item_info[1]
                i_tag = item_info[4]
                i_head_position = item_info[6]
                i_rel = item_info[7]
                i_word_lemm = WordNetLemmatizer().lemmatize(
                    i_word, self.wordnet_pos_resolver(i_tag))
                ret_list.append({"position": int(i_position),
                                 "word": i_word,
                                 "tag": i_tag,
                                 "position_head": int(i_head_position),
                                 "relation": i_rel,
                                 "word_lemmatized": i_word_lemm.lower()})
        return ret_list

    def print_structure_parsing(self):
        parser = StanfordParser(
            path_to_jar=self.stanfordparser_path,
            path_to_models_jar=self.stanfordparser_model_path)
        for sentence in self.tokenized_sentences:
            next(parser.raw_parse(sentence)).pretty_print()
