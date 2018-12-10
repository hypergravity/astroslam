import numpy as np

"""

define error processing functions

"""


# jacobian matrix to covariance
def jac_to_cov(jac):
    from scipy.linalg import svd
    _, s, VT = svd(jac, full_matrices=False)
    threshold = np.finfo(float).eps * max(jac.shape) * s[0]
    s = s[s > threshold]
    VT = VT[:s.size]
    pcov = np.dot(VT.T / s**2, VT)
    return pcov


# covariance to standard deviation
def cov_to_err(cov):
    return np.sqrt(np.diag(cov))


# jacobian matrix to standard deviation
def jac_to_err(jac):
    from scipy.linalg import svd
    _, s, VT = svd(jac, full_matrices=False)
    threshold = np.finfo(float).eps * max(jac.shape) * s[0]
    s = s[s > threshold]
    VT = VT[:s.size]
    pcov = np.dot(VT.T / s**2, VT)
    return np.sqrt(np.diag(pcov))


# inversed hessian matrix to standard deviation
def hessinv_to_err(hess_inv):
    return np.sqrt(np.diag(hess_inv))


# least_squares[jac] --> jacobian matrix --> standard deviation
def do_post(ls_result, label_scaler=None):
    """

    Parameters
    ----------
    ls_result:
        least_squares full output
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html

    label_scaler:
        SLAM label scale for training labels

    Returns
    -------
    pp_result:
        post-processed result

    """
    pp_result = dict()
    pp_result["cost"] = ls_result["cost"]
    pp_result["grad"] = ls_result["grad"]

    pp_result["pcov"] = jac_to_cov(ls_result["jac"]) * \
                        np.dot(label_scaler.scale_.reshape(-1, 1), label_scaler.scale_.reshape(1, -1))
    pp_result["pstd"] = cov_to_err(pp_result["pcov"])

    pp_result["message"] = ls_result["message"]
    pp_result["nfev"] = ls_result["nfev"]
    pp_result["optimality"] = ls_result["optimality"]
    pp_result["status"] = ls_result["status"]
    pp_result["success"] = ls_result["success"]

    pp_result["x"] = label_scaler.inverse_transform(ls_result["x"].reshape(1, -1)).flatten()

    return pp_result
