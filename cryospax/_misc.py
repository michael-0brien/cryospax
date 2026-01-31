import equinox as eqx
import jax


def filter_device_get(pytree):
    dynamic, static = eqx.partition(pytree, eqx.is_array)
    dynamic = jax.device_get(dynamic)
    return eqx.combine(dynamic, static)
