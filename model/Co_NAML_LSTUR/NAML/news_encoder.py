import torch
import torch.nn as nn
import torch.nn.functional as F
from model.general.attention.additive import AdditiveAttention
from transformers import DistilBertModel, DistilBertTokenizer

# Updated device selection to prefer MPS on Mac
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")


class TextEncoder(torch.nn.Module):
    def __init__(
        self,
        word_embedding,
        word_embedding_dim,
        num_filters,
        window_size,
        query_vector_dim,
        dropout_probability,
    ):
        super(TextEncoder, self).__init__()
        self.word_embedding = word_embedding
        self.dropout_probability = dropout_probability
        self.CNN = nn.Conv2d(
            1,
            num_filters,
            (window_size, word_embedding_dim),
            padding=(int((window_size - 1) / 2), 0),
        )
        self.additive_attention = AdditiveAttention(query_vector_dim, num_filters)

    def forward(self, text):
        # batch_size, num_words_text, word_embedding_dim
        text_vector = F.dropout(
            self.word_embedding(text),
            p=self.dropout_probability,
            training=self.training,
        )
        # batch_size, num_filters, num_words_title
        convoluted_text_vector = self.CNN(text_vector.unsqueeze(dim=1)).squeeze(dim=3)
        # batch_size, num_filters, num_words_title
        activated_text_vector = F.dropout(
            F.relu(convoluted_text_vector),
            p=self.dropout_probability,
            training=self.training,
        )

        # batch_size, num_filters
        text_vector = self.additive_attention(activated_text_vector.transpose(1, 2))
        return text_vector


class DistilBertTextEncoder(torch.nn.Module):
    def __init__(
        self,
        num_filters,
        query_vector_dim,
        dropout_probability,
        max_length=64,
    ):
        super(DistilBertTextEncoder, self).__init__()
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        self.bert_model = DistilBertModel.from_pretrained("distilbert-base-uncased")

        # Explicitly move BERT model to the correct device
        self.bert_model = self.bert_model.to(device)

        # Freeze DistilBERT parameters to avoid training them
        for param in self.bert_model.parameters():
            param.requires_grad = False

        self.dropout_probability = dropout_probability
        self.max_length = max_length

        # Project BERT's hidden size to the same dimension as filters in the CNN approach
        self.projection = nn.Linear(self.bert_model.config.hidden_size, num_filters)

        # Attention mechanism to get a weighted sum of token embeddings
        self.additive_attention = AdditiveAttention(query_vector_dim, num_filters)

        # Map word IDs to text for processing
        # We'll use a simple vocab for demo, in production you'd use the actual vocab
        self.id_to_word = None  # Will be loaded lazily

    def _ensure_id_to_word_mapping(self):
        if self.id_to_word is None:
            try:
                # Try to load the vocabulary mapping
                vocab = self.load_vocabulary()
                # Create a function to map IDs to words
                self.id_to_word = lambda x: vocab.get(x, f"token_{x}")
            except Exception as e:
                # Fallback to simple token placeholders
                print(
                    f"Warning: Could not load vocabulary mapping, using placeholder tokens. Error: {e}"
                )
                self.id_to_word = lambda x: f"token_{x}"

    def load_vocabulary(self):
        """
        Load the vocabulary mapping from ID to word.
        In a production system, this would load from your actual vocab file.
        """
        try:
            import os
            import json

            # Look for a vocabulary file in various locations
            vocab_file_paths = [
                "data/train/word_dict.json",
                "data/word_dict.json",
                "word_dict.json",
            ]

            for path in vocab_file_paths:
                if os.path.exists(path):
                    with open(path, "r", encoding="utf-8") as f:
                        word_dict = json.load(f)
                        # Convert from word->id to id->word
                        id_to_word = {v: k for k, v in word_dict.items()}
                        return id_to_word

            # If we didn't find a vocab file, try to create a simple one
            # This is just a fallback for demonstration purposes
            simple_vocab = {}
            return simple_vocab
        except Exception as e:
            print(f"Error loading vocabulary: {e}")
            return {}

    def forward(self, text):
        # text is a tensor containing word IDs from the original vocab
        batch_size = text.size(0)

        # Get the original tokenizer's vocab size for context
        self._ensure_id_to_word_mapping()

        # Create a batch of texts to process with DistilBERT
        batch_texts = []
        for i in range(batch_size):
            # Get non-padding word IDs (filter out zeros)
            word_ids = text[i][text[i] > 0].cpu().tolist()

            # Map IDs to placeholder tokens and join with spaces
            # In a real system, you would map to actual words here
            text_str = " ".join([self.id_to_word(wid) for wid in word_ids])
            batch_texts.append(text_str)

        # Tokenize with DistilBERT tokenizer
        encoded = self.tokenizer(
            batch_texts,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Explicitly move tensors to device and allocate properly for MPS
        input_ids = encoded["input_ids"].contiguous().to(device)
        attention_mask = encoded["attention_mask"].contiguous().to(device)

        # Get DistilBERT embeddings
        with torch.no_grad():  # We can freeze DistilBERT for efficiency
            outputs = self.bert_model(
                input_ids=input_ids, attention_mask=attention_mask
            )
            bert_embeddings = (
                outputs.last_hidden_state
            )  # [batch_size, seq_len, hidden_size]

        # Project to the desired dimension
        projected_embeddings = F.dropout(
            self.projection(bert_embeddings),
            p=self.dropout_probability,
            training=self.training,
        )  # [batch_size, seq_len, num_filters]

        # Use attention to get a weighted sum of token embeddings
        text_vector = self.additive_attention(
            projected_embeddings
        )  # [batch_size, num_filters]

        return text_vector


class ElementEncoder(torch.nn.Module):
    def __init__(self, embedding, linear_input_dim, linear_output_dim):
        super(ElementEncoder, self).__init__()
        self.embedding = embedding
        self.linear = nn.Linear(linear_input_dim, linear_output_dim)

    def forward(self, element):
        return F.relu(self.linear(self.embedding(element)))


class DistilBertElementEncoder(torch.nn.Module):
    def __init__(
        self,
        num_filters,
        query_vector_dim,
        dropout_probability,
        max_length=16,
        element_type="category",
    ):
        super(DistilBertElementEncoder, self).__init__()
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        self.bert_model = DistilBertModel.from_pretrained("distilbert-base-uncased")

        # Explicitly move BERT model to the correct device
        self.bert_model = self.bert_model.to(device)

        # Freeze DistilBERT parameters to avoid training them
        for param in self.bert_model.parameters():
            param.requires_grad = False

        self.dropout_probability = dropout_probability
        self.max_length = max_length
        self.element_type = element_type

        # Project BERT's hidden size to the same dimension as num_filters
        self.projection = nn.Linear(self.bert_model.config.hidden_size, num_filters)

        # Category/subcategory dictionaries for mapping IDs to text
        self.id_to_name = None  # Will be loaded lazily

    def _ensure_id_to_name_mapping(self):
        if self.id_to_name is None:
            try:
                # Try to load category/subcategory mapping
                element_dict = self.load_element_dict()
                # Create mapping function
                self.id_to_name = lambda x: element_dict.get(
                    x, f"{self.element_type}_{x}"
                )
            except Exception as e:
                # Fallback to simple placeholders
                print(
                    f"Warning: Could not load {self.element_type} mapping, using placeholders. Error: {e}"
                )
                self.id_to_name = lambda x: f"{self.element_type}_{x}"

    def load_element_dict(self):
        """
        Load the category/subcategory mapping from ID to name.
        """
        try:
            import os
            import json

            # Possible file paths
            element_file_paths = [
                f"data/train/{self.element_type}_dict.json",
                f"data/{self.element_type}_dict.json",
                f"{self.element_type}_dict.json",
            ]

            for path in element_file_paths:
                if os.path.exists(path):
                    with open(path, "r", encoding="utf-8") as f:
                        element_dict = json.load(f)
                        # Convert from name->id to id->name
                        id_to_name = {v: k for k, v in element_dict.items()}
                        return id_to_name

            # Fallback
            return {}
        except Exception as e:
            print(f"Error loading {self.element_type} dictionary: {e}")
            return {}

    def forward(self, element_ids):
        # element_ids is a tensor of category/subcategory IDs
        batch_size = element_ids.size(0)

        # Load mapping
        self._ensure_id_to_name_mapping()

        # Convert IDs to text
        element_texts = []
        for i in range(batch_size):
            # Filter out padding (0)
            elem_id = element_ids[i].item()
            if elem_id > 0:
                # Map ID to text name
                elem_text = self.id_to_name(elem_id)
                element_texts.append(elem_text)
            else:
                element_texts.append("unknown")

        # Encode with DistilBERT
        encoded = self.tokenizer(
            element_texts,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Explicitly move tensors to device and allocate properly for MPS
        input_ids = encoded["input_ids"].contiguous().to(device)
        attention_mask = encoded["attention_mask"].contiguous().to(device)

        # Get embeddings
        with torch.no_grad():
            outputs = self.bert_model(
                input_ids=input_ids, attention_mask=attention_mask
            )
            # Use CLS token embedding as the category/subcategory representation
            embeddings = outputs.last_hidden_state[:, 0, :]

        # Project to desired dimension and apply dropout
        element_vector = F.dropout(
            F.relu(self.projection(embeddings)),
            p=self.dropout_probability,
            training=self.training,
        )

        return element_vector


class NewsEncoder(torch.nn.Module):
    def __init__(self, config, pretrained_word_embedding):
        super(NewsEncoder, self).__init__()
        self.config = config

        # Use DistilBERT instead of GloVe word embeddings
        use_distilbert = getattr(config, "use_distilbert", False)

        text_encoders_candidates = ["title", "abstract"]
        if use_distilbert:
            self.text_encoders = nn.ModuleDict(
                {
                    name: DistilBertTextEncoder(
                        config.num_filters,
                        config.query_vector_dim,
                        config.dropout_probability,
                    )
                    for name in (
                        set(config.dataset_attributes["news"])
                        & set(text_encoders_candidates)
                    )
                }
            )

            # Also use DistilBERT for category and subcategory
            element_encoders_candidates = ["category", "subcategory"]
            self.element_encoders = nn.ModuleDict(
                {
                    name: DistilBertElementEncoder(
                        config.num_filters,
                        config.query_vector_dim,
                        config.dropout_probability,
                        element_type=name,
                    )
                    for name in (
                        set(config.dataset_attributes["news"])
                        & set(element_encoders_candidates)
                    )
                }
            )
        else:
            if pretrained_word_embedding is None:
                word_embedding = nn.Embedding(
                    config.num_words, config.word_embedding_dim, padding_idx=0
                )
            else:
                word_embedding = nn.Embedding.from_pretrained(
                    pretrained_word_embedding, freeze=False, padding_idx=0
                )
            self.text_encoders = nn.ModuleDict(
                {
                    name: TextEncoder(
                        word_embedding,
                        config.word_embedding_dim,
                        config.num_filters,
                        config.window_size,
                        config.query_vector_dim,
                        config.dropout_probability,
                    )
                    for name in (
                        set(config.dataset_attributes["news"])
                        & set(text_encoders_candidates)
                    )
                }
            )

            # Use regular embedding for category/subcategory when not using DistilBERT
            category_embedding = nn.Embedding(
                config.num_categories, config.category_embedding_dim, padding_idx=0
            )
            element_encoders_candidates = ["category", "subcategory"]
            self.element_encoders = nn.ModuleDict(
                {
                    name: ElementEncoder(
                        category_embedding,
                        config.category_embedding_dim,
                        config.num_filters,
                    )
                    for name in (
                        set(config.dataset_attributes["news"])
                        & set(element_encoders_candidates)
                    )
                }
            )

        if len(config.dataset_attributes["news"]) > 1:
            self.final_attention = AdditiveAttention(
                config.query_vector_dim, config.num_filters
            )

    def forward(self, news):
        """
        Args:
            news:
                {
                    "category": batch_size,
                    "subcategory": batch_size,
                    "title": batch_size * num_words_title,
                    "abstract": batch_size * num_words_abstract,
                }
        Returns:
            (shape) batch_size, num_filters
        """
        text_vectors = [
            encoder(news[name].to(device))
            for name, encoder in self.text_encoders.items()
        ]
        element_vectors = [
            encoder(news[name].to(device))
            for name, encoder in self.element_encoders.items()
        ]

        all_vectors = text_vectors + element_vectors

        if len(all_vectors) == 1:
            final_news_vector = all_vectors[0]
        else:
            final_news_vector = self.final_attention(torch.stack(all_vectors, dim=1))
        return final_news_vector
