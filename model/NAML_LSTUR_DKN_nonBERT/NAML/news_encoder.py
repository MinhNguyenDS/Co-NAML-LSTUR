import torch
import torch.nn as nn
import torch.nn.functional as F
from model.general.attention.additive import AdditiveAttention
from transformers import DistilBertModel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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
    ):
        super(DistilBertTextEncoder, self).__init__()
        self.bert_model = DistilBertModel.from_pretrained("distilbert-base-uncased")

        # Freeze DistilBERT parameters to avoid training them
        for param in self.bert_model.parameters():
            param.requires_grad = False

        self.dropout_probability = dropout_probability

        # Project BERT's hidden size to the same dimension as filters in the CNN approach
        self.projection = nn.Linear(self.bert_model.config.hidden_size, num_filters)

        # Attention mechanism to get a weighted sum of token embeddings
        self.additive_attention = AdditiveAttention(query_vector_dim, num_filters)

    def forward(self, input_ids, attention_mask):
        # input_ids: batch_size, seq_len
        # attention_mask: batch_size, seq_len
        # These are now expected to be pre-tokenized and passed directly

        # Get DistilBERT embeddings
        with torch.no_grad():  # We can freeze DistilBERT for efficiency
            outputs = self.bert_model(
                input_ids=input_ids.to(device), attention_mask=attention_mask.to(device)
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
        dropout_probability,
    ):
        super(DistilBertElementEncoder, self).__init__()
        self.bert_model = DistilBertModel.from_pretrained("distilbert-base-uncased")

        # Freeze DistilBERT parameters to avoid training them
        for param in self.bert_model.parameters():
            param.requires_grad = False

        self.dropout_probability = dropout_probability

        # Project BERT's hidden size to the same dimension as num_filters
        self.projection = nn.Linear(self.bert_model.config.hidden_size, num_filters)

    def forward(self, input_ids, attention_mask):
        # input_ids: batch_size, seq_len
        # attention_mask: batch_size, seq_len
        # These are now expected to be pre-tokenized and passed directly

        # Get embeddings
        with torch.no_grad():
            outputs = self.bert_model(
                input_ids=input_ids.to(device), attention_mask=attention_mask.to(device)
            )
            # Use CLS token embedding as the category/subcategory representation
            embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token

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
        element_encoders_candidates = ["category", "subcategory"]
        if use_distilbert:
            self.text_encoders = nn.ModuleDict(
                {
                    name: DistilBertTextEncoder(
                        config.num_filters,
                        config.query_vector_dim,
                        config.dropout_probability,
                    )
                    for name in text_encoders_candidates
                    if name in config.dataset_attributes["news"]
                }
            )

            self.element_encoders = nn.ModuleDict(
                {
                    name: DistilBertElementEncoder(
                        config.num_filters,
                        config.dropout_probability,
                    )
                    for name in element_encoders_candidates
                    if name in config.dataset_attributes["news"]
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
            news: A dictionary where keys are news attributes (e.g., 'title', 'category')
                  If use_distilbert is True, for text attributes like 'title', it expects:
                      news['title_input_ids']: batch_size * num_words_title (tensor of token IDs)
                      news['title_attention_mask']: batch_size * num_words_title (tensor of attention masks)
                  For element attributes like 'category', it expects:
                      news['category_input_ids']: batch_size (tensor of token IDs, e.g., for CLS token)
                      news['category_attention_mask']: batch_size (tensor of attention masks)
                  If use_distilbert is False, it expects original ID tensors:
                      news['title']: batch_size * num_words_title (tensor of word IDs)
                      news['category']: batch_size (tensor of category IDs)
        Returns:
            (shape) batch_size, num_filters
        """
        use_distilbert = getattr(self.config, "use_distilbert", False)
        all_vectors = []

        if use_distilbert:
            for name, encoder in self.text_encoders.items():
                # Construct key names for input_ids and attention_mask
                input_ids_key = f"{name}_input_ids"
                attention_mask_key = f"{name}_attention_mask"
                if input_ids_key in news and attention_mask_key in news:
                    input_ids = news[input_ids_key].to(device)
                    attention_mask = news[attention_mask_key].to(device)
                    all_vectors.append(encoder(input_ids, attention_mask))
                else:
                    # Fallback or error if pre-tokenized data is not found for a configured encoder
                    # For now, let's print a warning. In a real scenario, this should be handled robustly.
                    print(
                        f"Warning: Pre-tokenized data for '{name}' not found in news input."
                    )

            for name, encoder in self.element_encoders.items():
                input_ids_key = f"{name}_input_ids"
                attention_mask_key = f"{name}_attention_mask"
                if input_ids_key in news and attention_mask_key in news:
                    input_ids = news[input_ids_key].to(device)
                    attention_mask = news[attention_mask_key].to(device)
                    all_vectors.append(encoder(input_ids, attention_mask))
                else:
                    print(
                        f"Warning: Pre-tokenized data for element '{name}' not found in news input."
                    )
        else:
            # Original logic for non-DistilBERT encoders
            text_vectors = [
                encoder(news[name].to(device))
                for name, encoder in self.text_encoders.items()
                if name in news  # Ensure key exists
            ]
            element_vectors = [
                encoder(news[name].to(device))
                for name, encoder in self.element_encoders.items()
                if name in news  # Ensure key exists
            ]
            all_vectors = text_vectors + element_vectors

        if not all_vectors:  # Handle case where no vectors were generated
            # This might happen if news attributes in config don't match keys in news dict,
            # or if 'use_distilbert' is true but no tokenized data is provided.
            # Return a zero tensor of expected shape or raise an error.
            # For now, let's assume batch_size can be inferred or is 1 if not.
            # This part needs careful handling based on expected behavior.
            # Placeholder:
            print("Warning: No news vectors were generated. Returning a zero vector.")
            # Determine batch size from one of the inputs if possible, or use a default/error
            # This needs a robust way to get batch_size
            # For now, let's assume we can get it from a config or a common input.
            # This part is tricky without knowing the exact batch structure if news is empty.
            # Fallback: create a zeros tensor. This might not be correct for all batch sizes.
            # You'll need to ensure 'news' is never empty in a way that leads here without a clear batch_size.
            # A better approach would be to ensure news always has some reference tensor to get batch_size.
            # For now, to avoid crashing, but this needs review:
            example_key = next(iter(news))  # Get an example key
            # Try to get batch_size from a non-empty tensor in news
            batch_size = 1  # Default, needs improvement
            found_batch_size = False
            for key_val in news.values():
                if isinstance(key_val, torch.Tensor) and key_val.dim() > 0:
                    batch_size = key_val.size(0)
                    found_batch_size = True
                    break
            if not found_batch_size and hasattr(
                self.config, "batch_size"
            ):  # Fallback to config if available
                batch_size = self.config.batch_size

            return torch.zeros(batch_size, self.config.num_filters).to(device)

        if len(all_vectors) == 1:
            final_news_vector = all_vectors[0]
        else:
            # Stack tensors only if there's more than one
            try:
                stacked_vectors = torch.stack(all_vectors, dim=1)
                final_news_vector = self.final_attention(stacked_vectors)
            except RuntimeError as e:
                print(f"Error stacking vectors: {e}")
                print(f"Number of vectors: {len(all_vectors)}")
                for i, vec in enumerate(all_vectors):
                    print(f"Vector {i} shape: {vec.shape}")
                # Handle error, e.g., by returning a zero tensor or re-raising
                # This might happen if vectors have inconsistent batch sizes.
                # For now, re-raise to highlight the issue.
                raise e
        return final_news_vector
