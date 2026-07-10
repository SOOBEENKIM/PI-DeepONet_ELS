"""Minimal smoke test to verify the container environment.

Run inside the container:

    python -m els_hedging.smoke_test
"""

from __future__ import annotations


def main() -> None:
    import numpy as np
    import pandas as pd
    import scipy

    print("=== ELS hedging_2 smoke test ===")
    print(f"numpy   : {np.__version__}")
    print(f"pandas  : {pd.__version__}")
    print(f"scipy   : {scipy.__version__}")

    try:
        import torch

        print(f"torch   : {torch.__version__}")
        print(f"cuda ok : {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"device  : {torch.cuda.get_device_name(0)}")
    except Exception as exc:  # noqa: BLE001
        print(f"torch   : unavailable ({exc})")

    # tiny sanity computation
    x = np.linspace(0.0, 1.0, 5)
    print(f"sanity  : mean={x.mean():.3f}")
    print("OK")


if __name__ == "__main__":
    main()
