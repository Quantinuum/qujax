import jax
import jax.numpy as jnp
import numpy as np
import pytest

from qujax.statetensor_observable import sample_probs


@pytest.mark.parametrize("n_samps", [10000, 100000])
@pytest.mark.parametrize("compiled", [False, True])
def test_one_outcome_empirical_distribution_is_normalized(n_samps, compiled):
    probabilities = jnp.array([1.0], dtype=jnp.float32)
    sampler = jax.jit(sample_probs, static_argnums=2) if compiled else sample_probs
    result = sampler(probabilities, jax.random.PRNGKey(0), n_samps)
    assert jnp.allclose(result, probabilities, rtol=0, atol=1e-7)


@pytest.mark.parametrize("compiled", [False, True])
def test_empirical_distribution_matches_integer_sample_frequencies(compiled):
    probabilities = jnp.array([[0.1, 0.2], [0.3, 0.4]], dtype=jnp.float32)
    key = jax.random.PRNGKey(1)
    n_samps = 100000
    samples = jax.random.choice(
        key, jnp.arange(probabilities.size), shape=(n_samps,), p=probabilities.flatten()
    )
    frequencies = np.bincount(np.asarray(samples), minlength=probabilities.size)
    expected = (frequencies / n_samps).reshape(probabilities.shape)
    sampler = jax.jit(sample_probs, static_argnums=2) if compiled else sample_probs
    result = sampler(probabilities, key, n_samps)
    assert result.shape == probabilities.shape
    assert result.dtype == probabilities.dtype
    np.testing.assert_allclose(result, expected, rtol=0, atol=1e-7)
    assert jnp.isclose(result.sum(), 1.0, rtol=0, atol=1e-7)
