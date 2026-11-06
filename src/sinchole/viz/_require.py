"""Import guard for the optional plotting dependency."""


def eyepiece():
    """Return eyepiece or raise with the install hint for this extra."""
    try:
        import eyepiece as ep
    except ImportError as err:
        msg = "sinchole.viz requires eyepiece: pip install 'sinchole[viz]'"
        raise ImportError(msg) from err
    return ep
