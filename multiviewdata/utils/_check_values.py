def _process_parameter(parameter_name: str, parameter, default, n_views: int):
    if parameter is None:
        parameter = [default] * n_views
    elif not isinstance(parameter, (list, tuple)):
        parameter = [parameter] * n_views
    _check_parameter_number(parameter_name, parameter, n_views)
    return parameter


def _check_parameter_number(parameter_name: str, parameter, n_views: int):
    if len(parameter) != n_views:
        raise ValueError(
            f"number of views passed should match number of parameter {parameter_name}"
            f"len(views)={n_views} and "
            f"len({parameter_name})={len(parameter)}"
        )
