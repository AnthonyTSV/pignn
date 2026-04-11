import numpy as np

def compute_relative_error(exact_field: np.ndarray, predicted_field: np.ndarray) -> np.ndarray:
    r"""
    Compute the relative error between the exact and predicted fields.
    \varepsilon_{\mathrm{rel}}(t_i) =
    \frac{\left\|u_{\mathrm{pred}}(\cdot, t_i) - u_{\mathrm{ref}}(\cdot, t_i)\right\|_{L^2(\Omega)}}
    {\left\|u_{\mathrm{ref}}(\cdot, t_i)\right\|_{L^2(\Omega)} + \epsilon} \times 100\%
    """
    # Compute the absolute error
    abs_err = np.abs(predicted_field - exact_field)

    # Compute the relative error
    rel_err = abs_err / (np.max(exact_field) - np.min(exact_field)) * 100.0

    return rel_err


def compute_l2_error(predicted_field: np.ndarray, exact_field: np.ndarray) -> float:
    """
    Compute normalized L2 error between the predicted and exact fields for all time steps.
    """
    if isinstance(predicted_field, list):
        predicted_field = np.array(predicted_field)
    if isinstance(exact_field, list):
        exact_field = np.array(exact_field)
    l2_err = np.sqrt(np.mean(np.sum((predicted_field - exact_field) ** 2, axis=tuple(range(1, predicted_field.ndim))))) / np.sqrt(np.mean(np.sum(exact_field ** 2, axis=tuple(range(1, exact_field.ndim)))))
    return l2_err