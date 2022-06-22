from bitorch.layers.qembedding import QEmbedding, QEmbeddingBag
from bitorch.quantizations import quantizations_by_name
from torch.nn.functional import embedding, embedding_bag
import pytest
import torch
import numpy as np

TEST_INPUT_DATA = [
    (10, 10),
    (100, 10),
    (1000, 100),
    (30000, 300),
] * 3
TEST_QUANTIZATION_FUNCTIONS = list(quantizations_by_name.values())


@pytest.mark.parametrize("vocab_size, embedding_size", TEST_INPUT_DATA)
@pytest.mark.parametrize("quantization_function", TEST_QUANTIZATION_FUNCTIONS)
def test_qembedding(vocab_size, embedding_size, quantization_function):
    qembedding = QEmbedding(
        num_embeddings=vocab_size,
        embedding_dim=embedding_size,
        weight_quantization=quantization_function(),
        output_quantization=quantization_function(),
        sparse=False,
    )

    example_input = torch.zeros(vocab_size, dtype=int)
    example_input[np.random.randint(vocab_size)] = 1
    quantization = quantization_function()

    output = qembedding(example_input)

    binarized_embedding_table = quantization(qembedding.weight)

    raw_embeddings = embedding(
        example_input,
        binarized_embedding_table,
        qembedding.padding_idx,
        qembedding.max_norm,
        qembedding.norm_type,
        qembedding.scale_grad_by_freq,
        False,
    )
    assert torch.equal(output, quantization(raw_embeddings))

    # now sparse tests
    qembedding = QEmbedding(
        num_embeddings=vocab_size,
        embedding_dim=embedding_size,
        weight_quantization=quantization_function(),
        output_quantization=quantization_function(),
        sparse=True,
    )

    example_input = torch.tensor(np.random.randint(vocab_size), dtype=int)

    output = qembedding(example_input)

    binarized_embedding_table = quantization(qembedding.weight)

    raw_embeddings = embedding(
        example_input,
        binarized_embedding_table,
        qembedding.padding_idx,
        qembedding.max_norm,
        qembedding.norm_type,
        qembedding.scale_grad_by_freq,
        True,
    )
    assert torch.equal(output, quantization(raw_embeddings))


@pytest.mark.parametrize("vocab_size, embedding_size", TEST_INPUT_DATA)
@pytest.mark.parametrize("quantization_function", TEST_QUANTIZATION_FUNCTIONS)
def test_qembeddingbag(vocab_size, embedding_size, quantization_function):

    qembeddingbag = QEmbeddingBag(
        num_embeddings=vocab_size,
        embedding_dim=embedding_size,
        weight_quantization=quantization_function(),
        output_quantization=quantization_function(),
        sparse=False,
        mode="mean",
    )

    example_input = torch.zeros(vocab_size, dtype=int)
    for _ in range(np.random.randint(vocab_size)):
        example_input[np.random.randint(vocab_size)] = 1
    example_offset = torch.tensor((0,), dtype=int)
    quantization = quantization_function()

    output = qembeddingbag(example_input, example_offset)

    binarized_embedding_table = quantization(qembeddingbag.weight)

    # necessary for torch 1.8 compliance
    if hasattr(qembeddingbag, "padding_idx"):
        raw_embeddings = embedding_bag(
            example_input,
            binarized_embedding_table,
            example_offset,
            qembeddingbag.max_norm,
            qembeddingbag.norm_type,
            qembeddingbag.scale_grad_by_freq,
            qembeddingbag.mode,
            False,
            None,
            qembeddingbag.include_last_offset,
            qembeddingbag.padding_idx,
        )
    else:
        raw_embeddings = embedding_bag(
            example_input,
            binarized_embedding_table,
            example_offset,
            qembeddingbag.max_norm,
            qembeddingbag.norm_type,
            qembeddingbag.scale_grad_by_freq,
            qembeddingbag.mode,
            False,
            None,
            qembeddingbag.include_last_offset,
        )
    assert torch.equal(output, quantization(raw_embeddings))

    # now sparse tests

    qembeddingbag = QEmbeddingBag(
        num_embeddings=vocab_size,
        embedding_dim=embedding_size,
        weight_quantization=quantization_function(),
        output_quantization=quantization_function(),
        sparse=True,
        mode="mean",
    )

    example_input = torch.tensor(np.random.randint(vocab_size, size=np.random.randint(vocab_size)), dtype=int)

    output = qembeddingbag(example_input, example_offset)

    binarized_embedding_table = quantization(qembeddingbag.weight)

    # necessary for torch 1.8 compliance
    if hasattr(qembeddingbag, "padding_idx"):
        raw_embeddings = embedding_bag(
            example_input,
            binarized_embedding_table,
            example_offset,
            qembeddingbag.max_norm,
            qembeddingbag.norm_type,
            qembeddingbag.scale_grad_by_freq,
            qembeddingbag.mode,
            True,
            None,
            qembeddingbag.include_last_offset,
            qembeddingbag.padding_idx,
        )
    else:
        raw_embeddings = embedding_bag(
            example_input,
            binarized_embedding_table,
            example_offset,
            qembeddingbag.max_norm,
            qembeddingbag.norm_type,
            qembeddingbag.scale_grad_by_freq,
            qembeddingbag.mode,
            True,
            None,
            qembeddingbag.include_last_offset,
        )
    output = torch.nan_to_num(output, nan=0.0)
    output_raw = torch.nan_to_num(quantization(raw_embeddings), nan=0.0)
    assert torch.equal(output, output_raw)
