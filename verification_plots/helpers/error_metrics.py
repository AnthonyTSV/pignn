import numpy as np

def compute_relative_field_error(
    exact_field: np.ndarray, predicted_field: np.ndarray
) -> float:
    r"""
    Compute the relative L2 error between the exact and predicted fields.
    \varepsilon_{L^2}(u) =
    \frac{\left\|u_{\mathrm{pred}} - u_{\mathrm{ref}}\right\|_{L^2(\Omega)}}
    {\left\|u_{\mathrm{ref}}\right\|_{L^2(\Omega)}}
    """
    # Compute the L2 norm of the error
    error = np.linalg.norm(exact_field - predicted_field)

    # Compute the L2 norm of the exact field
    exact_norm = np.linalg.norm(exact_field)

    # Avoid division by zero
    if exact_norm == 0:
        return 0.0

    # Compute the relative error
    relative_error = error / exact_norm

    return relative_error


def compute_maximum_pointwise_error(
    exact_field: np.ndarray, predicted_field: np.ndarray
) -> float:
    r"""
    Compute the maximum pointwise error between the exact and predicted fields.
    \varepsilon_{\infty}(u) =
    \frac{\left\|u_{\mathrm{pred}} - u_{\mathrm{ref}}\right\|_{L^\infty(\Omega)}}
    {\left\|u_{\mathrm{ref}}\right\|_{L^\infty(\Omega)} + \epsilon}
    """
    # Compute the maximum pointwise error
    pointwise_error = np.max(np.abs(exact_field - predicted_field))

    # Compute the maximum value of the exact field
    exact_max = np.max(np.abs(exact_field))

    # Add a small epsilon to avoid division by zero
    epsilon = 1e-8

    # Compute the relative maximum pointwise error
    relative_pointwise_error = pointwise_error / (exact_max + epsilon)

    return relative_pointwise_error


def compute_probe_point_error(
    exact_field: np.ndarray, predicted_field: np.ndarray, probe_index: int
):
    r"""
    Compute the pointwise error at a specific probe location.
    \varepsilon_{\mathrm{probe}}(t_i) =
    \left|u_{\mathrm{pred}}(\mathbf{x}_p, t_i) - u_{\mathrm{ref}}(\mathbf{x}_p, t_i)\right|
    """
    if probe_index < 0 or probe_index >= len(exact_field):
        raise ValueError("Probe index is out of bounds.")

    # Compute the pointwise error at the probe location
    probe_error = np.abs(predicted_field[probe_index] - exact_field[probe_index])

    return probe_error
