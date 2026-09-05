import torch

def mup_invariant_diagnostics(widths, activation_rms,
                              activation_change_rms, slope_tolerance):
    """
    Returns: dictionary containing fitted slopes, ratios, and decisions
    """
    log_widths = torch.log(widths.double())
    design = torch.stack((torch.ones_like(log_widths), log_widths), dim=1)
    activation_fit = torch.linalg.lstsq(
        design, torch.log(activation_rms.double())
    ).solution
    change_fit = torch.linalg.lstsq(
        design, torch.log(activation_change_rms.double())
    ).solution
    activation_slope = float(activation_fit[1])
    change_slope = float(change_fit[1])
    return {
        "activation_slope": activation_slope,
        "activation_intercept": float(activation_fit[0]),
        "change_slope": change_slope,
        "change_intercept": float(change_fit[0]),
        "activation_ratio": float(activation_rms.max() / activation_rms.min()),
        "change_ratio": float(activation_change_rms.max() / activation_change_rms.min()),
        "activation_invariant": abs(activation_slope) <= slope_tolerance,
        "change_invariant": abs(change_slope) <= slope_tolerance,
    }
