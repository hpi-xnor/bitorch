import torch
from bitorch.models import LeNet
from bitorch.quantizations import Quantization_Scheduler, Sign, WeightDoReFa
from bitorch.quantizations.quantization_scheduler import MixLinearScheduling

INPUT_SHAPE = (10, 1, 28, 28)


def test_scheduler():
    torch.manual_seed(123)
    model = LeNet(INPUT_SHAPE, 10, 0)
    torch.manual_seed(123)
    model_unscheduled = LeNet(INPUT_SHAPE, 10, 0)
    torch.manual_seed(123)
    model_dorefa = LeNet(INPUT_SHAPE, 10, 1)

    par1 = list(model.parameters())[0]
    par2 = list(model_unscheduled.parameters())[0]
    par3 = list(model_dorefa.parameters())[0]
    assert torch.equal(par1, par2)
    assert torch.equal(par2, par3)

    scheduler = Quantization_Scheduler(model, 2, [Sign(), WeightDoReFa(), Sign()], scheduling_procedure="mix_linear")
    assert scheduler.scheduled_quantizer is MixLinearScheduling

    input_data = torch.rand(INPUT_SHAPE)
    sign_output = model_unscheduled(input_data)
    dorefa_output = model_dorefa(input_data)

    scheduled_output = model(input_data)
    assert torch.equal(scheduled_output, sign_output)
    scheduler.step()

    scheduled_output = model(input_data)
    assert torch.equal(scheduled_output, dorefa_output)
    scheduler.step()

    scheduled_output = model(input_data)
    assert torch.equal(scheduled_output, sign_output)
