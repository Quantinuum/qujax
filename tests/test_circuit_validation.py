import pytest

import qujax


@pytest.mark.parametrize(
    "builder",
    [
        qujax.get_params_to_statetensor_func,
        qujax.get_params_to_densitytensor_func,
        qujax.get_params_to_unitarytensor_func,
    ],
)
@pytest.mark.parametrize("param_inds", [[], [[]], [[], [], []]])
def test_reject_mismatched_parameter_index_sequences(builder, param_inds):
    with pytest.raises(TypeError, match="must have matching lengths"):
        builder(["X", "H"], [[0], [0]], param_inds)
