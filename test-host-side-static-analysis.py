# ---------- Demo (12 cases) ----------
def _demo():
    # Original 4
    src1 = """
def prog(a,b):
    eta = random(0.1, 0.5)
    return int(eta*(a-b))
"""
    print("Demo1:", find_dead_outcomes(src1, "prog", measurement_vars=["a","b"]))

    src2 = """
def prog(a,b,m):
    x = 2*a - 3*b
    y = b + m
    z = x*y
    t = a*y
    return z - 2*t
"""
    print("Demo2:", find_dead_outcomes(src2, "prog", measurement_vars=["m"]))

    src3 = """
def prog(m,b):
    if m > 0:
        return b+1
    else:
        return b+2
"""
    print("Demo3:", find_dead_outcomes(src3, "prog", measurement_vars=["m"]))

    src4 = """
def prog(m,b):
    if m > 0:
        x = b*2
    else:
        x = b*2
    return x
"""
    print("Demo4:", find_dead_outcomes(src4, "prog", measurement_vars=["m"]))

    # Your added 5–12
    src5 = """
def prog(m,b):
    # Overwrite m with randomness → initial m is non-contributory
    m = random(0, 1)
    return b + m
"""
    print("Demo5:", find_dead_outcomes(src5, "prog", measurement_vars=["m"]))

    src6 = """
def prog(m,b):
    # m is never used → non-contributory
    x = b*b + 1
    return x
"""
    print("Demo6:", find_dead_outcomes(src6, "prog", measurement_vars=["m"]))

    src7 = """
def prog(m,b):
    # m only affects control flow, but both branches compute the same result → non-contributory
    if m > 0:
        x = 3*b + 2
    else:
        x = 3*b + 2
    return x
"""
    print("Demo7:", find_dead_outcomes(src7, "prog", measurement_vars=["m"]))

    src8 = """
def prog(m,b):
    # Return is independent of m; branch depends only on b, but both branches return same t
    t = int((b + 3)/2)
    if b % 2 == 0:
        return t
    else:
        return t
"""
    print("Demo8:", find_dead_outcomes(src8, "prog", measurement_vars=["m"]))

    src9 = """
def prog(m1, m2, b):
    # m1 flows to output (contributory); m2 overwritten by random (non-contributory)
    u = 2*m1 + b
    m2 = random(0, 1)
    return u + m2
"""
    print("Demo9:", find_dead_outcomes(src9, "prog", measurement_vars=["m1","m2"]))

    src10 = """
def prog(m,b):
    # m randomized then used → initial m non-contributory
    m = int(random(0, 10))
    y = 2*b
    return y + m
"""
    print("Demo10:", find_dead_outcomes(src10, "prog", measurement_vars=["m"]))

    src11 = """
def prog(m,b):
    # m only in condition; both branches add the same random x → initial m non-contributory
    x = random(0,1)
    if m > 0:
        return b + x
    else:
        return b + x
"""
    print("Demo11:", find_dead_outcomes(src11, "prog", measurement_vars=["m"]))

    src12 = """
def prog(m,b):
    # Local overwrite of m by expression independent of its initial value → non-contributory
    t = b - 1
    m = t + 1
    return m + b
"""
    print("Demo12:", find_dead_outcomes(src12, "prog", measurement_vars=["m"]))



def _demo_cancellations():
    # C1 -> Demo13: linear cancellation -> m dead
    src13 = """
def prog(m,b):
    x = m + b
    return x - m - b   # -> 0
"""
    print("Demo13:", find_dead_outcomes(src13, "prog", measurement_vars=["m"]))

    # C2 -> Demo14: distribution cancellation -> m dead
    src14 = """
def prog(m,b,c):
    z = (m + b) * c
    t = m*c + b*c
    return z - t       # -> 0
"""
    print("Demo14:", find_dead_outcomes(src14, "prog", measurement_vars=["m"]))

    # C3 -> Demo15: FOIL equality -> m dead
    src15 = """
def prog(m,b,c):
    u = (m + b) * (m + c)
    v = m*m + m*c + b*m + b*c
    return u - v       # -> 0
"""
    print("Demo15:", find_dead_outcomes(src15, "prog", measurement_vars=["m"]))

    # C4 -> Demo16: add then subtract same thing -> m dead
    src16 = """
def prog(m,a):
    s = m
    s += a
    s -= a
    return s - m       # -> 0
"""
    print("Demo16:", find_dead_outcomes(src16, "prog", measurement_vars=["m"]))

    # C5 -> Demo17: int() identity on zero -> m dead
    src17 = """
def prog(m):
    return int(5 + 2*m - 2*m - 5)   # -> int(0)
"""
    print("Demo17:", find_dead_outcomes(src17, "prog", measurement_vars=["m"]))

    # C6 -> Demo18: factor cancels -> m dead
    src18 = """
def prog(m,b):
    x = 3*(m + b)
    return x - 3*m - 3*b            # -> 0
"""
    print("Demo18:", find_dead_outcomes(src18, "prog", measurement_vars=["m"]))

    # C7 -> Demo19: random scale cancels inside int() -> m dead
    src19 = """
def prog(m,b):
    eta = random(0.1, 0.9)
    return int(eta*(m + b) - eta*m - eta*b)   # -> int(0)
"""
    print("Demo19:", find_dead_outcomes(src19, "prog", measurement_vars=["m"]))

    # C8 -> Demo20: two int(...) terms both 0 -> m dead
    src20 = """
def prog(m):
    eta = random(0.1, 0.9)
    t1 = int(eta*m)   # -> 0
    t2 = int(eta*m)   # -> 0
    return t1 - t2    # -> 0
"""
    print("Demo20:", find_dead_outcomes(src20, "prog", measurement_vars=["m"]))

    # C9 -> Demo21: same expr on both branches, then cancel -> m dead
    src21 = """
def prog(m,b):
    if b > 0:
        s = 2*m + b
    else:
        s = 2*m + b
    return s - s       # -> 0
"""
    print("Demo21:", find_dead_outcomes(src21, "prog", measurement_vars=["m"]))

    # C10 -> Demo22: only m1 cancels; m2 remains -> expect ['m1']
    src22 = """
def prog(m1,m2,b):
    return (m1 + b) - m1 + 2*m2     # -> b + 2*m2
"""
    print("Demo22:", find_dead_outcomes(src22, "prog", measurement_vars=["m1","m2"]))

    # C11 -> Demo23: distributed forms cancel -> m dead
    src23 = """
def prog(m,b,c):
    p = (2*m - 3*b) * c
    q = 2*m*c - 3*b*c
    return p - q                     # -> 0
"""
    print("Demo23:", find_dead_outcomes(src23, "prog", measurement_vars=["m"]))

    # C12 -> Demo24: tuple return, first component cancels -> m dead
    src24 = """
def prog(m,b):
    s = m + b
    return (s - m - b, 7)            # -> (0, 7)
"""
    print("Demo24:", find_dead_outcomes(src24, "prog", measurement_vars=["m"]))

    # C13 -> Demo25: capture initial m, overwrite name, cancel the copy -> m dead
    src25 = """
def prog(m):
    t = m        # captures m@0
    m = m + 1    # overwrite name
    return t - t # -> 0
"""
    print("Demo25:", find_dead_outcomes(src25, "prog", measurement_vars=["m"]))

    # C14 -> Demo26: mixed expansion cancellation -> m dead
    src26 = """
def prog(m,a,b):
    x = (a + m)*(b + 2)
    y = a*b + 2*a + b*m + 2*m
    return x - y                     # -> 0
"""
    print("Demo26:", find_dead_outcomes(src26, "prog", measurement_vars=["m"]))

    # C15 -> Demo27: factorization cancellation -> m dead
    src27 = """
def prog(m,b,c):
    return (m*b + m*c) - m*(b + c)   # -> 0
"""
    print("Demo27:", find_dead_outcomes(src27, "prog", measurement_vars=["m"]))

    # C16 -> Demo28: augmented mult cancellation -> m dead
    src28 = """
def prog(m):
    s = m
    s *= 3
    u = 3*m
    return s - u                     # -> 0
"""
    print("Demo28:", find_dead_outcomes(src28, "prog", measurement_vars=["m"]))
    
    src29 = """
def prog(m):
    s = m
    s *= 3
    u = 2*m
    return s - u                     # -> 0
"""
    print("Demo29:", find_dead_outcomes(src29, "prog", measurement_vars=["m"]))

    src30 = """
def prog(m, a):
    a = 2*m
    b = m
    c = b
    return a - b - c                  
"""
    print("Demo30:", find_dead_outcomes(src30, "prog", measurement_vars=["m"]))
    
    src31 = """
def prog(a, b, c):
    z = a * b
    return z + c
"""
    print("Demo31:", find_dead_outcomes(src31, "prog", measurement_vars=['a', 'b', 'c']))
    
    src32 = """
def prog(a, b, c):
    x = 2*a - 3*b
    y = b + c + 2
    z = x * y
    t = a * y
    return z - 2*t
"""
    print("Demo32:", find_dead_outcomes(src32, "prog", measurement_vars=['a', 'b', 'c']))
    
    src33 = """
def prog(a, b, c):
    i = random(0.1, 0.5)
    x = i * a
    y = b * c
    z = int(x + y)
    return z
"""
    print("Demo33:", find_dead_outcomes(src33, "prog", measurement_vars=['a', 'b', 'c']))
    
    src34 = """
def prog(a, b, c):
    x = a + b
    y = c - a
    u = x * y
    v = a * (b - c)
    w = a * a
    return u + v + w
"""
    print("Demo34:", find_dead_outcomes(src34, "prog", measurement_vars=['a', 'b', 'c']))
    
    src35 = """
def prog(a, b, c):
    i = random(0.1, 0.5)
    u = int(i*a + b*c)
    v = b - c
    w = c - b
    return u + v + w
"""
    print("Demo35:", find_dead_outcomes(src35, "prog", measurement_vars=['a', 'b', 'c']))
    
    src36 = """
def prog(a, b, c):
    if a:
        x = a + b
        y = c - a
        u = x * y
        v = a * (b - c)
        w = a * a
    else:
        i = random(0.1, 0.5)
        u = int(i*a + b*c)
        v = b - c
        w = c - b
    return u + v + w
"""
    print("Demo36:", find_dead_outcomes(src36, "prog", measurement_vars=['a', 'b', 'c']))

    src_vqe = """
def vqe_optimizer(o0, o1, o2, o3):
    # Only o2,o3 should matter.
    EZ = fZ(o2, o3)          # unknown call → deps {o2,o3}
    EX = fX(o2, o3)          # unknown call → deps {o2,o3}
    updates = combine(EZ, EX)  # unknown call → deps {o2,o3}
    return updates
"""
    print("VQE:", find_dead_outcomes(src_vqe, "vqe_optimizer", measurement_vars=["o0","o1","o2","o3"]))
# Expected: ['o0', 'o1']

    src_qpe = """
def proc_qpe(o0, o1, o2, o3, o4):
    # θ = 0.θ0θ1θ2θ3θ4  in *decimal*; each θi ∈ {0,1}
    theta = 0.1*o0 + 0.01*o1 + 0.001*o2 + 0.0001*o3 + 0.00001*o4
    lam = 10 * theta                     # lam = o0 + 0.1*o1 + ...
    return lam - int(lam)                # fractional part → independent of o0
    """
    print("QPE:", find_dead_outcomes(src_qpe, "proc_qpe", measurement_vars=["o0","o1","o2","o3","o4"]))
# Expected: ['o0']
