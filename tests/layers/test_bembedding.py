import numpy as np
import pytest
import torch
from bitorch.layers.bembedding import BEmbedding, BEmbeddingBag
from bitorch.quantizations import ApproxSign, Sign, SwishSign
from torch.nn.functional import embedding, embedding_bag
from torch.optim import SGD, Adam

TEST_INPUT_DATA = [
    (10, 10),
    (100, 10),
    (1000, 100),
    (30000, 300),
] * 3
TEST_QUANTIZATION_FUNCTIONS = [ApproxSign, Sign, SwishSign]
TEST_OPTIMIZERS = [Adam, SGD]


@pytest.mark.parametrize("vocab_size, embedding_size", TEST_INPUT_DATA)
@pytest.mark.parametrize("quantization_function", TEST_QUANTIZATION_FUNCTIONS)
def test_bembedding(vocab_size, embedding_size, quantization_function):
    qembedding = BEmbedding(
        num_embeddings=vocab_size,
        embedding_dim=embedding_size,
        weight_quantization=quantization_function(),
    )
    example_input = torch.zeros(vocab_size, dtype=int)
    example_input[np.random.randint(vocab_size)] = 1

    output = qembedding(example_input)

    binarized_embedding_table = qembedding.weight.to(dtype=torch.float32)

    raw_embeddings = embedding(
        input=example_input,
        weight=binarized_embedding_table,
        sparse=False,
    )
    assert torch.equal(output, raw_embeddings)

    # now sparse tests
    qembedding = BEmbedding(
        num_embeddings=vocab_size,
        embedding_dim=embedding_size,
        weight_quantization=quantization_function(),
    )

    example_input = torch.tensor(np.random.randint(vocab_size), dtype=int)
    output = qembedding(example_input)

    binarized_embedding_table = qembedding.weight.to(dtype=torch.float32)
    raw_embeddings = embedding(
        input=example_input,
        weight=binarized_embedding_table,
        sparse=True,
    )
    assert torch.equal(output, raw_embeddings)


@pytest.mark.parametrize("vocab_size, embedding_size", TEST_INPUT_DATA)
@pytest.mark.parametrize("quantization_function", TEST_QUANTIZATION_FUNCTIONS)
def test_batched_bembedding(vocab_size, embedding_size, quantization_function):
    qembedding = BEmbedding(
        num_embeddings=vocab_size,
        embedding_dim=embedding_size,
        weight_quantization=quantization_function(),
    )
    example_input = torch.zeros(vocab_size, dtype=int)
    example_input[np.random.randint(vocab_size)] = 1
    example_input[np.random.randint(vocab_size)] = 1

    output = qembedding(example_input)

    binarized_embedding_table = qembedding.weight.to(dtype=torch.float32)

    raw_embeddings = embedding(
        input=example_input,
        weight=binarized_embedding_table,
        sparse=False,
    )
    assert torch.equal(output, raw_embeddings)

    # now sparse tests
    qembedding = BEmbedding(
        num_embeddings=vocab_size,
        embedding_dim=embedding_size,
        weight_quantization=quantization_function(),
    )

    example_input = torch.randint(0, vocab_size, (np.random.randint(1, 100), 1), dtype=int)
    output = qembedding(example_input)
    binarized_embedding_table = qembedding.weight.to(dtype=torch.float32)
    raw_embeddings = embedding(
        input=example_input,
        weight=binarized_embedding_table,
        sparse=True,
    )
    assert torch.equal(output, raw_embeddings)


@pytest.mark.parametrize("vocab_size, embedding_size", TEST_INPUT_DATA)
@pytest.mark.parametrize("quantization_function", TEST_QUANTIZATION_FUNCTIONS)
@pytest.mark.parametrize("optimizer", TEST_OPTIMIZERS)
def test_bembedding_training(vocab_size, embedding_size, quantization_function, optimizer):
    example_input = torch.randint(0, vocab_size, (1,), dtype=int)
    example_output = torch.rand(size=(1, 1))
    assert_equal_train(example_input, example_output, vocab_size, embedding_size, quantization_function, optimizer)


@pytest.mark.parametrize("vocab_size, embedding_size", TEST_INPUT_DATA)
@pytest.mark.parametrize("quantization_function", TEST_QUANTIZATION_FUNCTIONS)
@pytest.mark.parametrize("optimizer", TEST_OPTIMIZERS)
def test_batched_bembedding_training(vocab_size, embedding_size, quantization_function, optimizer):
    batch_size = np.random.randint(1, 100)
    example_input = torch.randint(0, vocab_size, (batch_size,), dtype=int)
    example_output = torch.rand((batch_size, 1))
    assert_equal_train(example_input, example_output, vocab_size, embedding_size, quantization_function, optimizer)


@pytest.mark.parametrize("vocab_size, embedding_size", TEST_INPUT_DATA)
@pytest.mark.parametrize("quantization_function", TEST_QUANTIZATION_FUNCTIONS)
@pytest.mark.parametrize("optimizer", TEST_OPTIMIZERS)
def test_batched_bembedding_training_duplicate(vocab_size, embedding_size, quantization_function, optimizer):
    example_input = torch.tensor([0, 1, 2, 3, 1])
    example_output = torch.rand((len(example_input), 1))
    assert_equal_train(example_input, example_output, vocab_size, embedding_size, quantization_function, optimizer)


def assert_equal_train(input, output, vocab_size, embedding_size, quantization_function, optimizer_class):
    input.requires_grad_(False)
    output.requires_grad_(False)
    weights = (torch.rand((vocab_size, embedding_size)) * 100) - 5
    linear = torch.nn.Linear(embedding_size, 1).requires_grad_(False)
    below_zero = False
    if quantization_function()(weights).min().item() < 0:
        below_zero = True
    qembedding = BEmbedding(
        num_embeddings=vocab_size,
        embedding_dim=embedding_size,
        weight_quantization=quantization_function(),
        sign_bool=below_zero,
    )
    qembedding.set_weight(weights)
    model = torch.nn.Sequential(qembedding, linear)
    optimizer = optimizer_class(model.parameters(), lr=0.03)
    qembedding.set_optimizer(optimizer)
    model_output1 = model(input)
    torch.nn.functional.l1_loss(model_output1, output).backward()
    optimizer.step()
    qembedding.step()
    model_output1 = model(input)
    torch.nn.functional.l1_loss(model_output1, output).backward()
    optimizer.step()
    qembedding.step()

    class NormalEmbedding(torch.nn.Module):
        def __init__(self, weight, q_function) -> None:
            super().__init__()
            self.weight = weight
            self.q_function = q_function

        def forward(self, x):
            return embedding(x, self.q_function(self.weight), sparse=False)

    normal_embedding = NormalEmbedding(torch.clone(weights).requires_grad_(True), quantization_function())
    model = torch.nn.Sequential(normal_embedding, linear)
    optimizer = optimizer_class(model.parameters(), lr=0.03)
    model_output2 = model(input)
    torch.nn.functional.l1_loss(model_output2, output).backward()
    optimizer.step()
    model_output2 = model(input)
    torch.nn.functional.l1_loss(model_output2, output).backward()
    optimizer.step()
    qweight = qembedding.weight.clone().to(torch.float32)
    if below_zero:
        qweight[qweight == 0] = -1
    nweight = quantization_function()(normal_embedding.weight)
    assert torch.equal(model_output1, model_output2)
    assert torch.equal(qweight, nweight)


@pytest.mark.parametrize("vocab_size, embedding_size", TEST_INPUT_DATA)
@pytest.mark.parametrize("quantization_function", TEST_QUANTIZATION_FUNCTIONS)
@pytest.mark.parametrize("optimizer", TEST_OPTIMIZERS)
def test_optimizer_is_cleared(vocab_size, embedding_size, quantization_function, optimizer):
    model = BEmbedding(
        num_embeddings=vocab_size,
        embedding_dim=embedding_size,
        weight_quantization=quantization_function(),
    )
    example_input = torch.tensor([0, 1, 2, 3, 1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    before_size = len(optimizer.param_groups)
    model.set_optimizer(optimizer)
    model(example_input)
    assert before_size + 1 == len(optimizer.param_groups)
    model(example_input)
    assert before_size + 1 == len(optimizer.param_groups)


@pytest.mark.parametrize("vocab_size, embedding_size", TEST_INPUT_DATA)
@pytest.mark.parametrize("quantization_function", TEST_QUANTIZATION_FUNCTIONS)
@pytest.mark.parametrize("optimizer", TEST_OPTIMIZERS)
def test_batched_bembedding_bag_training_duplicate(vocab_size, embedding_size, quantization_function, optimizer):
    example_input = torch.tensor([1, 2, 3, 1, 1, 2])
    example_offsets = torch.tensor([0, 2, 3, 4, 5])
    example_output = torch.rand((len(example_offsets), 1))
    assert_equal_train_embedding_bag(
        example_input, example_offsets, example_output, vocab_size, embedding_size, quantization_function, optimizer
    )


def assert_equal_train_embedding_bag(
    input_indices, input_offsets, output, vocab_size, embedding_size, quantization_function, optimizer_class
):
    input_indices.requires_grad_(False)
    input_offsets.requires_grad_(False)
    output.requires_grad_(False)
    weights = (torch.rand((vocab_size, embedding_size)) * 100) - 50
    linear = torch.nn.Linear(embedding_size, 1).requires_grad_(False)
    below_zero = False
    if quantization_function()(weights).min().item() < 0:
        below_zero = True
    qembedding = BEmbeddingBag(
        num_embeddings=vocab_size,
        embedding_dim=embedding_size,
        weight_quantization=quantization_function(),
        sign_bool=below_zero,
    )
    qembedding.set_weight(weights)
    model = torch.nn.Sequential(qembedding, linear)
    qoptimizer = optimizer_class(model.parameters(), lr=0.03)
    qembedding.set_optimizer(qoptimizer)

    class NormalEmbeddingBag(torch.nn.Module):
        def __init__(self, weight, q_function) -> None:
            super().__init__()
            self.weight = weight
            self.q_function = q_function

        def forward(self, indices, offsets):
            return embedding_bag(input=indices, offsets=offsets, weight=self.q_function(self.weight), sparse=False)

    normal_embedding = NormalEmbeddingBag(torch.clone(weights).requires_grad_(True), quantization_function())
    model = torch.nn.Sequential(normal_embedding, linear)
    optimizer = optimizer_class(model.parameters(), lr=0.03)

    # Check if weights match before
    qweight = qembedding.weight.clone().to(torch.float32)
    if below_zero:
        qweight[qweight == 0] = -1
    nweight = quantization_function()(normal_embedding.weight)
    assert torch.equal(qweight, nweight)

    # First pass
    model_output1_1 = linear(qembedding(input_indices, input_offsets))
    torch.nn.functional.l1_loss(model_output1_1, output).backward()
    optimizer.step()
    qembedding.step()
    model_output2_1 = linear(normal_embedding(input_indices, input_offsets))
    torch.nn.functional.l1_loss(model_output2_1, output).backward()
    optimizer.step()

    qweight = qembedding.weight.clone().to(torch.float32)
    if below_zero:
        qweight[qweight == 0] = -1
    nweight = quantization_function()(normal_embedding.weight)
    assert torch.equal(qweight, nweight)

    # Second pass
    model_output1_2 = linear(qembedding(input_indices, input_offsets))
    torch.nn.functional.l1_loss(model_output1_2, output).backward()
    optimizer.step()
    qembedding.step()
    model_output2_2 = linear(normal_embedding(input_indices, input_offsets))
    torch.nn.functional.l1_loss(model_output2_2, output).backward()
    optimizer.step()

    qweight = qembedding.weight.clone().to(torch.float32)
    if below_zero:
        qweight[qweight == 0] = -1
    nweight = quantization_function()(normal_embedding.weight)
    assert torch.equal(model_output1_2, model_output2_2)
    assert torch.equal(qweight, nweight)


@pytest.mark.parametrize("vocab_size, embedding_size", TEST_INPUT_DATA)
@pytest.mark.parametrize("quantization_function", TEST_QUANTIZATION_FUNCTIONS)
def test_bembedding_bag(vocab_size, embedding_size, quantization_function):
    qembedding = BEmbeddingBag(
        num_embeddings=vocab_size,
        embedding_dim=embedding_size,
        weight_quantization=quantization_function(),
    )
    example_input = torch.tensor([0, 1, 2])
    example_offsets = torch.tensor([0])

    output = qembedding(example_input, example_offsets)

    binarized_embedding_table = qembedding.weight.to(dtype=torch.float32)

    raw_embeddings = embedding_bag(
        input=example_input,
        offsets=example_offsets,
        weight=binarized_embedding_table,
        sparse=False,
    )
    assert torch.equal(output, raw_embeddings)

    qembedding = BEmbeddingBag(
        num_embeddings=vocab_size,
        embedding_dim=embedding_size,
        weight_quantization=quantization_function(),
    )
    example_input = torch.tensor([0, 1, 2, 1, 2, 3])
    example_offsets = torch.tensor([0, 3])

    output = qembedding(example_input, example_offsets)

    binarized_embedding_table = qembedding.weight.to(dtype=torch.float32)

    raw_embeddings = embedding_bag(
        input=example_input,
        offsets=example_offsets,
        weight=binarized_embedding_table,
        sparse=False,
    )
    assert torch.equal(output, raw_embeddings)
