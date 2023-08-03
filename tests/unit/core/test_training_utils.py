import numpy as np
import torch

from fgvc.core.training.training_utils import concat_arrays, to_device, to_numpy


def test_to_device():
    """Test converting pytorch tensors to device."""
    # test multiple tensors
    input_tensors = [
        torch.randn(1, 3, 24, 24),
        dict(p1=torch.randn(1, 3, 24, 24), p2=torch.randn(4, 3, 24, 24)),
    ]
    output_tensors = to_device(*input_tensors, device=torch.device("cpu"))
    assert torch.allclose(input_tensors[0], output_tensors[0])
    assert torch.allclose(input_tensors[1]["p1"], output_tensors[1]["p1"])
    assert torch.allclose(input_tensors[1]["p2"], output_tensors[1]["p2"])

    # test a single tensor
    input_tensor = torch.randn(1, 3, 24, 24)
    output_tensor = to_device(input_tensor, device=torch.device("cpu"))
    assert torch.allclose(input_tensor, output_tensor)

    # test a single dictionary with tensors
    input_dict = dict(p1=torch.randn(1, 3, 24, 24), p2=torch.randn(4, 3, 24, 24))
    output_dict = to_device(input_dict, device=torch.device("cpu"))
    assert torch.allclose(input_dict["p1"], output_dict["p1"])
    assert torch.allclose(input_dict["p2"], output_dict["p2"])


def test_to_numpy():
    """Test converting pytorch tensors to numpy arrays."""
    # test multiple tensors
    input_tensors = [
        torch.randn(1, 3, 24, 24, requires_grad=True),
        dict(p1=torch.randn(1, 3, 24, 24), p2=torch.randn(4, 3, 24, 24, requires_grad=True)),
    ]
    output_arrays = to_numpy(*input_tensors)
    assert np.allclose(input_tensors[0].detach().numpy(), output_arrays[0])
    assert np.allclose(input_tensors[1]["p1"].numpy(), output_arrays[1]["p1"])
    assert np.allclose(input_tensors[1]["p2"].detach().numpy(), output_arrays[1]["p2"])

    # test a single tensor
    input_tensor = torch.randn(1, 3, 24, 24, requires_grad=True)
    output_array = to_numpy(input_tensor)
    assert np.allclose(input_tensor.detach().numpy(), output_array)

    # test a single dictionary with tensors
    input_dict = dict(
        p1=torch.randn(1, 3, 24, 24), p2=torch.randn(4, 3, 24, 24, requires_grad=True)
    )
    output_dict = to_numpy(input_dict)
    assert np.allclose(input_dict["p1"].numpy(), output_dict["p1"])
    assert np.allclose(input_dict["p2"].detach().numpy(), output_dict["p2"])


def test_concat_arrays():
    """Test concatenating numpy arrays."""
    # test concatenating lists of arrays
    input_arrays_1 = [np.random.randn(4, 128) for _ in range(5)]
    input_arrays_2 = [np.random.randint(0, 10, size=4) for _ in range(5)]
    output_array_1, output_array_2 = concat_arrays(input_arrays_1, input_arrays_2)
    assert np.allclose(np.concatenate(input_arrays_1), output_array_1)
    assert np.allclose(np.concatenate(input_arrays_2), output_array_2)
    output_array_1 = concat_arrays(input_arrays_1)
    assert np.allclose(np.concatenate(input_arrays_1), output_array_1)

    # test concatenating lists of dictionaries with arrays
    input_dicts_1 = [dict(p1=np.random.randn(4, 128), p2=np.random.randn(4, 512)) for _ in range(5)]
    input_arrays_2 = [np.random.randint(0, 10, size=4) for _ in range(5)]
    output_dict_1, output_array_2 = concat_arrays(input_dicts_1, input_arrays_2)
    assert np.allclose(np.concatenate([x["p1"] for x in input_dicts_1]), output_dict_1["p1"])
    assert np.allclose(np.concatenate([x["p2"] for x in input_dicts_1]), output_dict_1["p2"])
    assert np.allclose(np.concatenate(input_arrays_2), output_array_2)
