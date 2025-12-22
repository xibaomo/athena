import numpy as np
def golden_section_search(f, a, b, tol=1e-6, max_iter=200):
    gr = (np.sqrt(5) - 1) / 2  # golden ratio ≈ 0.618
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc, fd = f(c), f(d)

    for _ in range(max_iter):
        if abs(b - a) < tol:
            break

        if fc < fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = f(d)

    xmin = (b + a) / 2
    return xmin, f(xmin)