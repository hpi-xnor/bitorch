import bitorch
from bitorch.models import model_from_name, model_names


def test_all_model_names():
    wrong_model_names = []
    for model_name in model_names():
        model = model_from_name(model_name)
        assert model_from_name(model.name) == model
        assert model_from_name(model.name.lower()) == model
        if model.name != model.__name__:
            wrong_model_names.append((model.name, model.__name__))
    assert len(wrong_model_names) == 0
