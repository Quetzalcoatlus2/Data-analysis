import warnings


def pytest_configure() -> None:
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        message=r"All support for the `google\.generativeai` package has ended\..*",
    )
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        module=r"app",
    )
