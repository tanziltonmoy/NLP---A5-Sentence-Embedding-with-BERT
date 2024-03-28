import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class Embedding(nn.Module):
    """
    Embedding layer that combines token embeddings, position embeddings, and segment embeddings,
    followed by layer normalization. Designed for use in transformer models.
    """
    def __init__(self, vocab_size, max_len, n_segments, d_model, device):
        """
        Initializes the embedding layer.

        Args:
            vocab_size (int): Size of the vocabulary.
            max_len (int): Maximum length of the input sequences.
            n_segments (int): Number of distinct segments or types of tokens.
            d_model (int): Dimensionality of the embeddings.
            device (torch.device): The device (CPU/GPU) where the tensors will be allocated.
        """
        super(Embedding, self).__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)  # Embedding layer for token IDs.
        self.pos_embed = nn.Embedding(max_len, d_model)     # Embedding layer for positional encoding.
        self.seg_embed = nn.Embedding(n_segments, d_model)  # Embedding layer for segment IDs.
        self.norm = nn.LayerNorm(d_model)  # Layer normalization to stabilize the inputs to the subsequent layers.
        self.device = device  # Specifies the device for tensor allocation.

    def forward(self, x, seg):
        """
        Forward pass of the embedding layer.

        Args:
            x (Tensor): Input tensor with token IDs. Shape: (batch_size, sequence_length).
            seg (Tensor): Segment ID tensor to indicate different parts of the input. Shape: (batch_size, sequence_length).

        Returns:
            Tensor: The combined embeddings with normalization applied. Shape: (batch_size, sequence_length, d_model).
        """
        seq_len = x.size(1)  # Determine the sequence length from the input tensor.
        
        # Generate a position tensor, move it to the specified device, and match its shape with the input tensor.
        pos = torch.arange(seq_len, dtype=torch.long).to(self.device)
        pos = pos.unsqueeze(0).expand_as(x)  # Shape transformation to match `x`.
        
        # Compute the sum of token embeddings, position embeddings, and segment embeddings.
        embedding = self.tok_embed(x) + self.pos_embed(pos) + self.seg_embed(seg)
        
        # Apply layer normalization to the combined embeddings before returning.
        return self.norm(embedding)


def get_attn_pad_mask(seq_q, seq_k, device):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1).to(device)  # batch_size x 1 x len_k(=len_q), one is masking
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # batch_size x len_q x len_k

class EncoderLayer(nn.Module):
    """
    Represents a single layer within a transformer encoder.
    
    This layer consists of two main components:
    1. Multi-head self-attention mechanism.
    2. Position-wise feed-forward network.
    
    Each encoder layer processes the input sequence using self-attention and then applies a position-wise feed-forward neural network to the result.
    """
    def __init__(self, n_heads, d_model, d_ff, d_k, device):
        """
        Initializes the encoder layer with specified parameters.
        
        Args:
            n_heads (int): Number of attention heads.
            d_model (int): Dimensionality of the model's output space.
            d_ff (int): Dimensionality of the feed-forward network's inner layer.
            d_k (int): Dimensionality of the key/query vectors in the attention mechanism.
            device (torch.device): Device (CPU/GPU) on which the computations will be executed.
        """
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(n_heads, d_model, d_k, device)  # Initializes the multi-head self-attention.
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff)  # Initializes the position-wise feed-forward network.

    def forward(self, enc_inputs, enc_self_attn_mask):
        """
        Forward pass of the encoder layer.
        
        Args:
            enc_inputs (Tensor): Input tensor to the encoder layer. Shape: (batch_size, sequence_length, d_model).
            enc_self_attn_mask (Tensor): Mask tensor for self-attention mechanism to ignore specific positions within the input. Shape: (batch_size, sequence_length).
        
        Returns:
            Tuple[Tensor, Tensor]: A tuple containing:
                - The output of the encoder layer after processing `enc_inputs`. Shape: (batch_size, sequence_length, d_model).
                - The attention weights from the multi-head self-attention mechanism. Shape varies based on implementation.
        """
        # Applies self-attention to the input. The same tensor `enc_inputs` is used as queries, keys, and values.
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        
        # Applies the position-wise feed-forward network to the output of the self-attention mechanism.
        enc_outputs = self.pos_ffn(enc_outputs)
        
        return enc_outputs, attn

class ScaledDotProductAttention(nn.Module):
    """
    Implements the scaled dot-product attention mechanism.

    The attention function used here is the dot product of queries and keys, scaled by the square root of the dimensionality of keys, followed by application of a softmax function to obtain the weights on the values.
    """
    def __init__(self, d_k, device):
        """
        Initializes the ScaledDotProductAttention layer.

        Args:
            d_k (int): Dimensionality of the key vectors. It is used to scale the dot product of the queries and keys.
            device (torch.device): Specifies the device for computation. This is important for
                                   transferring the scaling factor to the same device as the input tensors.
        """
        super(ScaledDotProductAttention, self).__init__()
        # Scale factor for the dot products, sqrt(d_k), moved to the specified device.
        self.scale = torch.sqrt(torch.FloatTensor([d_k])).to(device)

    def forward(self, Q, K, V, attn_mask):
        """
        Forward pass of the scaled dot-product attention.

        Args:
            Q (Tensor): Queries. Shape: [batch_size, n_heads, len_q, d_k].
            K (Tensor): Keys. Shape: [batch_size, n_heads, len_k, d_k].
            V (Tensor): Values. Shape: [batch_size, n_heads, len_v(=len_k), d_v].
            attn_mask (Tensor): An attention mask to prevent attention to certain positions. This is important for masking out padding tokens. Shape: [batch_size, n_heads, len_q, len_k].

        Returns:
            Tuple[Tensor, Tensor]: A tuple containing:
                - The context tensor after applying attention to the value vectors. Shape: [batch_size, n_heads, len_q, d_v].
                - The attention weights. Shape: [batch_size, n_heads, len_q, len_k].
        """
        # Calculate the dot products of Q and K, scale them, and apply the attention mask.
        scores = torch.matmul(Q, K.transpose(-1, -2)) / self.scale
        scores.masked_fill_(attn_mask, -1e9)  # Apply attention mask with a large negative number to softmax.

        # Apply softmax to get the attention weights.
        attn = nn.Softmax(dim=-1)(scores)

        # Multiply the attention weights with the value vectors to get the context.
        context = torch.matmul(attn, V)

        return context, attn


class MultiHeadAttention(nn.Module):
    """
    Implements the Multi-Head Attention mechanism, a crucial component in Transformers. 
    This mechanism allows the model to jointly attend to information from different representation subspaces at different positions.
    """
    def __init__(self, n_heads, d_model, d_k, device):
        """
        Initializes the MultiHeadAttention layer.
        
        Args:
            n_heads (int): Number of attention heads.
            d_model (int): Dimensionality of the model's output space.
            d_k (int): Dimensionality of the key/query vectors in each attention head.
            device (torch.device): Device (CPU/GPU) on which the computations will be executed.
        """
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_k  # For simplicity, size of 'v' is made equal to that of 'k'.
        # Linear transformations for queries, keys, and values
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_k * n_heads)  # Note: d_v is used as d_k for consistency.
        self.device = device
        # Output linear transformation
        self.fc = nn.Linear(n_heads * d_k, d_model).to(device)
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model).to(device)

    def forward(self, Q, K, V, attn_mask):
        """
        Forward pass of the MultiHeadAttention layer.
        
        Args:
            Q (Tensor): Queries tensor. Shape: [batch_size, len_q, d_model].
            K (Tensor): Keys tensor. Shape: [batch_size, len_k, d_model].
            V (Tensor): Values tensor. Shape: [batch_size, len_v(=len_k), d_model].
            attn_mask (Tensor): Tensor indicating positions to be masked with negative infinity for softmax. Shape: [batch_size, len_q, len_k].
        
        Returns:
            Tuple[Tensor, Tensor]: A tuple containing:
                - Output tensor after applying multi-head attention and residual connection followed by layer normalization. Shape: [batch_size, len_q, d_model].
                - Attention weights across the heads. Shape: [batch_size, n_heads, len_q, len_k].
        """
        residual, batch_size = Q.clone(), Q.size(0)
        # Prepare query, key, value tensors for attention mechanism
        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1,2)
        # Repeat attn_mask for each attention head
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)

        # Apply scaled dot product attention
        context, attn = ScaledDotProductAttention(self.d_k, self.device)(q_s, k_s, v_s, attn_mask)
        # Concatenate heads and apply final linear transformation
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v)
        output = self.fc(context)
        # Apply layer normalization and residual connection
        return self.layer_norm(output + residual), attn


class PoswiseFeedForwardNet(nn.Module):
    """
    Implements a position-wise feedforward network as described in the Transformer model architecture.
    This network is applied to each position separately and identically. It consists of two fully-connected layers
    with a GELU non-linearity between them.
    """
    def __init__(self, d_model, d_ff):
        """
        Initializes the position-wise feedforward network.

        Args:
            d_model (int): The number of expected features in the input (model dimension).
            d_ff (int): The dimensionality of the feed-forward network's inner layer.
        """
        super(PoswiseFeedForwardNet, self).__init__()
        # First fully connected layer increases dimensionality from d_model to d_ff
        self.fc1 = nn.Linear(d_model, d_ff)
        # Second fully connected layer decreases dimensionality back from d_ff to d_model
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        """
        Forward pass through the position-wise feedforward network.

        Args:
            x (Tensor): The input tensor with shape [batch_size, sequence_length, d_model].

        Returns:
            Tensor: The output tensor with the same shape as the input, [batch_size, sequence_length, d_model].
        """
        # Apply the first linear transformation followed by a GELU non-linearity
        # Then apply the second linear transformation to project the dimensions back
        return self.fc2(F.gelu(self.fc1(x)))
    

class BERT(nn.Module):
    def __init__(self, n_layers, n_heads, d_model, d_ff, d_k, n_segments, vocab_size, max_len, device):
        super(BERT, self).__init__()
        self.params = {'n_layers': n_layers, 'n_heads': n_heads, 'd_model': d_model,
                       'd_ff': d_ff, 'd_k': d_k, 'n_segments': n_segments,
                       'vocab_size': vocab_size, 'max_len': max_len}
        self.embedding = Embedding(vocab_size, max_len, n_segments, d_model, device)
        self.layers = nn.ModuleList([EncoderLayer(n_heads, d_model, d_ff, d_k, device) for _ in range(n_layers)])
        self.fc = nn.Linear(d_model, d_model)
        self.activ = nn.Tanh()
        self.linear = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, 2)
        # decoder is shared with embedding layer
        embed_weight = self.embedding.tok_embed.weight
        n_vocab, n_dim = embed_weight.size()
        self.decoder = nn.Linear(n_dim, n_vocab, bias=False)
        self.decoder.weight = embed_weight
        self.decoder_bias = nn.Parameter(torch.zeros(n_vocab))
        self.device = device

    def forward(self, input_ids, segment_ids, masked_pos):
        output = self.embedding(input_ids, segment_ids)
        enc_self_attn_mask = get_attn_pad_mask(input_ids, input_ids, self.device)
        for layer in self.layers:
            output, enc_self_attn = layer(output, enc_self_attn_mask)
        # output : [batch_size, len, d_model], attn : [batch_size, n_heads, d_mode, d_model]
        
        # 1. predict next sentence
        # it will be decided by first token(CLS)
        h_pooled   = self.activ(self.fc(output[:, 0])) # [batch_size, d_model]
        logits_nsp = self.classifier(h_pooled) # [batch_size, 2]

        # 2. predict the masked token
        masked_pos = masked_pos[:, :, None].expand(-1, -1, output.size(-1)) # [batch_size, max_pred, d_model]
        h_masked = torch.gather(output, 1, masked_pos) # masking position [batch_size, max_pred, d_model]
        h_masked  = self.norm(F.gelu(self.linear(h_masked)))
        logits_lm = self.decoder(h_masked) + self.decoder_bias # [batch_size, max_pred, n_vocab]

        return logits_lm, logits_nsp
    
    def get_last_hidden_state(self, input_ids, segment_ids):
        output = self.embedding(input_ids, segment_ids)
        enc_self_attn_mask = get_attn_pad_mask(input_ids, input_ids, self.device)
        for layer in self.layers:
            output, enc_self_attn = layer(output, enc_self_attn_mask)

        return output
    
     
import re
from sklearn.metrics.pairwise import cosine_similarity

# define mean pooling function
def mean_pool(token_embeds, attention_mask):
    # reshape attention_mask to cover 768-dimension embeddings
    in_mask = attention_mask.unsqueeze(-1).expand(
        token_embeds.size()
    ).float()
    # perform mean-pooling but exclude padding tokens (specified by in_mask)
    pool = torch.sum(token_embeds * in_mask, 1) / torch.clamp(
        in_mask.sum(1), min=1e-9
    )
    return pool

def get_inputs(sentence, tokenizer, vocab, max_seq_length):
    tokens = tokenizer(re.sub("[.,!?\\-]", '', sentence.lower()))
    input_ids = [vocab['[CLS]']] + [vocab[token] for token in tokens] + [vocab['[SEP]']]
    n_pad = max_seq_length - len(input_ids)
    attention_mask = ([1] * len(input_ids)) + ([0] * n_pad)
    input_ids = input_ids + ([0] * n_pad)

    return {'input_ids': torch.LongTensor(input_ids).reshape(1, -1),
            'attention_mask': torch.LongTensor(attention_mask).reshape(1, -1)}

def calculate_similarity(model, tokenizer, vocab, max_seq_length, sentence_a, sentence_b, device):
    # Tokenize and convert sentences to input IDs and attention masks
    inputs_a = get_inputs(sentence_a, tokenizer, vocab, max_seq_length)
    inputs_b = get_inputs(sentence_b, tokenizer, vocab, max_seq_length)
    

    # Move input IDs and attention masks to the active device
    inputs_ids_a = inputs_a['input_ids'].to(device)
    attention_a = inputs_a['attention_mask'].to(device)
    inputs_ids_b = inputs_b['input_ids'].to(device)
    attention_b = inputs_b['attention_mask'].to(device)
    segment_ids = torch.zeros(1, max_seq_length, dtype=torch.int32).to(device)

    # Extract token embeddings from BERT
    u = model.get_last_hidden_state(inputs_ids_a, segment_ids)
    v = model.get_last_hidden_state(inputs_ids_b, segment_ids)

    u = mean_pool(u, attention_a).detach().cpu().numpy().reshape(-1)  # batch_size, hidden_dim
    v = mean_pool(v, attention_b).detach().cpu().numpy().reshape(-1)  # batch_size, hidden_dim

    # Calculate cosine similarity
    similarity_score = cosine_similarity(u.reshape(1, -1), v.reshape(1, -1))[0, 0]

    return similarity_score