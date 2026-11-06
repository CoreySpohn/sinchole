"""Pytest configuration for the fenced-code run over the docs tree."""

# The generator scripts write the committed validation data and figures; they are
# run by hand, not executed as documentation examples.
collect_ignore = ["scripts"]
