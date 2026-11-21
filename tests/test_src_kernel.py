"""prep/kernel/test_kernel.py against src/kernel.py (see conftest.py).

    pytest tests -v -rs                                   # GPU
    TRITON_INTERPRET=1 pytest tests -v -k "12-3"          # CPU, small cases
    modal run tests/modal_tests.py                         # on a Modal GPU
"""

from test_kernel import *  # noqa: F401,F403  (pytest collects the imported tests)
