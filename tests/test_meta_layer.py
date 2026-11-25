for corr, liq, age, dis, div, dd, cost in itertools.product(
        [None, 0.436, 0.65, 0.80, 0.90],       # unchanged
        [0.1, 1.0],                              # REMOVED None
        [None, 10, 500],
        [None, 0.1, 0.9],
        [False, True],
        [0, 2000],
        [False, True],
    ):