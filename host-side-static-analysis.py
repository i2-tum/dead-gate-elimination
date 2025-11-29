# Version 09.11.2025
# dead_start_analysis.py
# Host-side static analysis for "dead measurement outcomes" (initial values that are non-contributory)

import ast, math
from fractions import Fraction
from typing import Dict, Tuple, List, Optional, Union, Set

# ---------- Algebra over polynomials (Fractions) ----------
Monomial = Tuple[Tuple[str, int], ...]           # sorted tuple of (symbol, power)
Poly = Dict[Monomial, Fraction]
UNKNOWN = object()

def fconst(x: Union[int, float, Fraction]) -> Fraction:
    if isinstance(x, Fraction): return x
    if isinstance(x, int):      return Fraction(x, 1)
    return Fraction(str(x))  # safer than binary float

def poly_const(c) -> Poly:      return {(): fconst(c)}
def poly_var(sym: str) -> Poly: return {((sym, 1),): Fraction(1, 1)}

def poly_add(a: Poly, b: Poly) -> Poly:
    out = dict(a)
    for m, c in b.items():
        out[m] = out.get(m, Fraction(0)) + c
        if out[m] == 0: del out[m]
    return out

def poly_neg(a: Poly) -> Poly:  return {m: -c for m, c in a.items()}
def poly_sub(a: Poly, b: Poly) -> Poly: return poly_add(a, poly_neg(b))

def mul_monos(m1: Monomial, m2: Monomial) -> Monomial:
    d: Dict[str, int] = {}
    for v,p in m1: d[v] = d.get(v,0)+p
    for v,p in m2: d[v] = d.get(v,0)+p
    return tuple(sorted((v,p) for v,p in d.items() if p))

def poly_mul(a: Poly, b: Poly) -> Poly:
    out: Poly = {}
    for m1,c1 in a.items():
        for m2,c2 in b.items():
            m = mul_monos(m1,m2)
            out[m] = out.get(m, Fraction(0)) + c1*c2
            if out[m] == 0: del out[m]
    return out

def poly_vars(a: Poly) -> Set[str]:
    s: Set[str] = set()
    for m in a:
        for (v,_) in m: s.add(v)
    return s

# ---------- Abstract value ----------
class AV:
    __slots__ = ("poly", "deps")
    def __init__(self, poly: Optional[Poly], deps: Optional[Set[str]] = None):
        self.poly = poly
        s = set()
        if poly is not None:
            s |= poly_vars(poly)
        if deps:
            s |= deps
        self.deps: Set[str] = s

    def __repr__(self):
        return f"AV(poly={'yes' if self.poly is not None else 'no'}, deps={sorted(self.deps)})"

def av_from_poly(p: Poly) -> AV:     return AV(p, None)
def av_const(c: Union[int, bool]) -> AV: return AV(poly_const(int(c)), None)
def av_union(a: AV, b: AV) -> Set[str]: return a.deps | b.deps
def _deps_from_poly(p: Optional[Poly]) -> Set[str]:
    return poly_vars(p) if p is not None else set()

def av_add(a: AV, b: AV) -> AV:
    if a.poly is not None and b.poly is not None:
        rp = poly_add(a.poly, b.poly)
        return AV(rp, _deps_from_poly(rp))
    return AV(None, a.deps | b.deps)

def av_sub(a: AV, b: AV) -> AV:
    if a.poly is not None and b.poly is not None:
        rp = poly_sub(a.poly, b.poly)
        return AV(rp, _deps_from_poly(rp))
    return AV(None, a.deps | b.deps)

def av_mul(a: AV, b: AV) -> AV:
    if a.poly is not None and b.poly is not None:
        rp = poly_mul(a.poly, b.poly)
        return AV(rp, _deps_from_poly(rp))
    return AV(None, a.deps | b.deps)

def av_neg(a: AV) -> AV:
    if a.poly is not None:
        rp = poly_neg(a.poly)
        return AV(rp, _deps_from_poly(rp))
    return AV(None, set(a.deps))

# ---------- Domains and intervals ----------
Interval = Tuple[Fraction, Fraction]

def poly_is_integer_valued(p: Poly, domain: Dict[str, str]) -> bool:
    # integer-valued if all coeffs are integers and no 'real' symbols appear
    if any(c.denominator != 1 for c in p.values()):
        return False
    for v in poly_vars(p):
        if domain.get(v, 'int') == 'real':
            return False
    return True

def has_finite_bounds(bounds: Dict[str, Interval], p: Poly) -> bool:
    return all(v in bounds for v in poly_vars(p))

def ipow_interval(iv: Interval, k: int) -> Interval:
    L,U = iv; assert L <= U and k >= 0
    if k == 0: return Fraction(1), Fraction(1)
    if k % 2 == 1: return (L**k, U**k)
    cand = [L**k, U**k]
    if L <= 0 <= U: cand.append(Fraction(0))
    return (min(cand), max(cand))

def imul(a: Interval, b: Interval) -> Interval:
    L1,U1=a; L2,U2=b
    prods = [L1*L2, L1*U2, U1*L2, U1*U2]
    return (min(prods), max(prods))

def iadd(a: Interval, b: Interval) -> Interval:
    L1,U1=a; L2,U2=b
    return (L1+L2, U1+U2)

def poly_interval(p: Poly, bounds: Dict[str, Interval]) -> Optional[Interval]:
    if not has_finite_bounds(bounds, p): return None
    total: Optional[Interval] = (Fraction(0), Fraction(0))
    for mono, coeff in p.items():
        term_iv: Interval = (Fraction(1), Fraction(1))
        for (v,powk) in mono:
            term_iv = imul(term_iv, ipow_interval(bounds[v], powk))
        term_iv = imul(term_iv, (coeff, coeff))
        total = iadd(total, term_iv)  # type: ignore
    return total

def int_trunc_constant(iv: Interval) -> Optional[int]:
    L,U = iv
    if L >= 0:
        kL,kU = math.floor(L), math.floor(U)
        return kL if kL == kU else None
    if U <= 0:
        cL,cU = math.ceil(L), math.ceil(U)
        return cL if cL == cU else None
    if L > -1 and U < 1: return 0
    return None

# NEW: split polynomial into "integer-valued part" vs "rest"
def split_poly_integer_vs_rest(p: Poly, domain: Dict[str, str]) -> Tuple[Poly, Poly]:
    p_int: Poly = {}
    p_rest: Poly = {}
    for mono, coeff in p.items():
        # integer coefficient?
        if coeff.denominator != 1:
            p_rest[mono] = coeff
            continue
        # any real-valued symbol present?
        has_real = any(domain.get(v, 'int') == 'real' for (v, _) in mono)
        if has_real:
            p_rest[mono] = coeff
        else:
            # product of ints/binaries remains integer-valued
            p_int[mono] = coeff
    return p_int, p_rest

# ---------- Analyzer ----------
class Analyzer(ast.NodeVisitor):

    def __init__(self, func: ast.FunctionDef, measurement_vars: Set[str]):
        self.func = func
        self.params = [a.arg for a in func.args.args]

        self.env: Dict[str, AV] = {}
        self.domain: Dict[str, str] = {}       # symbol -> domain kind
        self.bounds: Dict[str, Interval] = {}  # symbol -> interval

        self.meas: Set[str] = set(measurement_vars)
        self.version: Dict[str, int] = {}      # name -> current version (only bumped on random())
        self.returns: List[Union[AV, Tuple[AV, ...]]] = []
        self._rand_counter = 0

        # NEW: pending branch environments for an immediate following return
        self._pending_envs_after_if: Optional[Tuple[Dict[str, AV], Dict[str, AV]]] = None
        self._pending_if_test_deps: Set[str] = set()

        # initial bind
        for p in self.params:
            self.version[p] = 0
            sym0 = self._sym(p, 0)
            av = av_from_poly(poly_var(sym0))
            self.env[p] = av
            if p in self.meas:
                self.domain[sym0] = 'binary'
                self.bounds[sym0] = (Fraction(0), Fraction(1))
            else:
                self.domain[sym0] = 'int'

    def _sym(self, v: str, k: int) -> str:
        return f"{v}@{k}"

    def _fresh_random(self, L: Fraction, U: Fraction) -> AV:
        self._rand_counter += 1
        sym = f"rand@{self._rand_counter}"
        self.domain[sym] = 'real'
        self.bounds[sym] = (L, U)
        return av_from_poly(poly_var(sym))

    # NEW: helpers for path-sensitive return joining -------------------------
    def _eval_in_env(self, env: Dict[str, AV], node: ast.AST) -> AV:
        """Evaluate expression 'node' under a temporary environment."""
        save = self.env
        try:
            self.env = env
            return self._eval(node)
        finally:
            self.env = save

    def _av_equal(self, a: AV, b: AV) -> bool:
        if a.poly is not None and b.poly is not None:
            return a.poly == b.poly
        return a.deps == b.deps

    def _join_av(self, a: AV, b: AV) -> AV:
        if a.poly is not None and b.poly is not None and a.poly == b.poly:
            return a
        if a.deps == b.deps:
            return AV(None, set(a.deps))
        return AV(None, a.deps | b.deps)

    def _join_out(self, ra: Union[AV, Tuple[AV, ...]],
                         rb: Union[AV, Tuple[AV, ...]]) -> Union[AV, Tuple[AV, ...]]:
        """Join two return AVs (possibly tuples) elementwise."""
        if isinstance(ra, tuple) and isinstance(rb, tuple):
            n = min(len(ra), len(rb))
            return tuple(self._join_av(ra[i], rb[i]) for i in range(n))
        if not isinstance(ra, tuple) and not isinstance(rb, tuple):
            return self._join_av(ra, rb)
        # Fallback: shape mismatch -> dependence-only on union of all deps
        def deps_of(x):
            if isinstance(x, tuple):
                d = set()
                for e in x: d |= e.deps
                return d
            return set(x.deps)
        return AV(None, deps_of(ra) | deps_of(rb))

    # --- Expression evaluation to AV ---
    def _eval(self, node) -> AV:
        if isinstance(node, ast.Constant):
            v = node.value
            if isinstance(v, bool):  return av_const(v)
            if isinstance(v, (int,float)): return AV(poly_const(v))
            return AV(None, set())

        if isinstance(node, ast.Name):
            return self.env.get(node.id, AV(None, set()))

        if isinstance(node, ast.BinOp):
            A, B = self._eval(node.left), self._eval(node.right)
            if isinstance(node.op, ast.Add):  return av_add(A,B)
            if isinstance(node.op, ast.Sub):  return av_sub(A,B)
            if isinstance(node.op, ast.Mult): return av_mul(A,B)
            # unsupported (/, %, **, etc.): keep dependency-only info
            return AV(None, av_union(A,B))

        if isinstance(node, ast.BoolOp):
            deps: Set[str] = set()
            for v in node.values:
                deps |= self._eval(v).deps
            return AV(None, deps)

        if isinstance(node, ast.Compare):
            deps: Set[str] = set()
            deps |= self._eval(node.left).deps
            for comp in node.comparators:
                deps |= self._eval(comp).deps
            return AV(None, deps)

        if isinstance(node, ast.UnaryOp):
            A = self._eval(node.operand)
            if isinstance(node.op, ast.USub): return av_neg(A)
            if isinstance(node.op, ast.UAdd): return A
            return AV(None, set(A.deps))

        if isinstance(node, ast.Tuple):
            # A tuple expression value is opaque in this analysis
            return AV(None, set())

        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            fname = node.func.id
            if fname == "int":
                if len(node.args) != 1:
                    return AV(None, set())
                A = self._eval(node.args[0])

                # Case 1: exact integer-valued polynomial -> identity
                if A.poly is not None and poly_is_integer_valued(A.poly, self.domain):
                    return A

                # Case 2: constant-fold via global interval
                if A.poly is not None:
                    iv = poly_interval(A.poly, self.bounds)
                    if iv is not None:
                        k = int_trunc_constant(iv)
                        if k is not None:
                            return av_const(k)

                    # NEW Case 3: split p = p_int + p_rest, with p_rest ∈ [0,1) ⇒ int(p)=p_int
                    p_int, p_rest = split_poly_integer_vs_rest(A.poly, self.domain)
                    if p_rest:
                        iv_rest = poly_interval(p_rest, self.bounds)
                        if iv_rest is not None:
                            Lr, Ur = iv_rest
                            if Lr >= 0 and Ur < 1:
                                # safe to drop p_rest inside int()
                                return AV(p_int, _deps_from_poly(p_int))

                # Fallback: dependence-only
                return AV(None, set(A.deps))

            if fname == "random":
                # expression-position random(L,U): fresh symbol with bounds if numeric literals
                if len(node.args) == 2:
                    Lnode, Unode = node.args
                    if (isinstance(Lnode, ast.Constant) and isinstance(Unode, ast.Constant) and
                        isinstance(Lnode.value, (int,float)) and isinstance(Unode.value, (int,float))):
                        L, U = fconst(Lnode.value), fconst(Unode.value)
                        if L > U: L, U = U, L
                        return self._fresh_random(L, U)
                # otherwise, conservative union of arg deps
                deps = set()
                for a in node.args:
                    deps |= self._eval(a).deps
                for kw in getattr(node, "keywords", []):
                    if kw.value is not None:
                        deps |= self._eval(kw.value).deps
                return AV(None, deps)

            # Other calls: union of arg/kw deps
            deps = set()
            for a in node.args:
                deps |= self._eval(a).deps
            for kw in getattr(node, "keywords", []):
                if kw.value is not None:
                    deps |= self._eval(kw.value).deps
            return AV(None, deps)

        return AV(None, set())

    # --- Random assignment creates a fresh symbol version (independent of prior one) ---
    def _assign_random(self, targets, call_node: ast.Call) -> bool:
        if len(call_node.args) != 2: return False
        Lnode, Unode = call_node.args
        if not (isinstance(Lnode, ast.Constant) and isinstance(Unode, ast.Constant)): return False
        if not (isinstance(Lnode.value, (int,float)) and isinstance(Unode.value, (int,float))): return False
        L, U = fconst(Lnode.value), fconst(Unode.value)
        if L > U: L, U = U, L
        ok = False
        for t in targets:
            if isinstance(t, ast.Name):
                v = t.id
                k = self.version.get(v, 0) + 1
                self.version[v] = k
                symk = self._sym(v, k)
                self.env[v] = av_from_poly(poly_var(symk))
                self.domain[symk] = 'real'
                self.bounds[symk] = (L, U)
                ok = True
            else:
                for n in ast.walk(t):
                    if isinstance(n, ast.Name):
                        self.env[n.id] = AV(None, set())
        return ok

    # --- Assignments / control flow / returns ---
    def _assign(self, target, value_av: AV):
        if isinstance(target, ast.Name):
            v = target.id
            self.env[v] = value_av
            return
        for n in ast.walk(target):
            if isinstance(n, ast.Name):
                self.env[n.id] = AV(None, set())

    def visit_Assign(self, node: ast.Assign):
        # NEW: any non-return statement clears pending-if info
        self._pending_envs_after_if = None
        self._pending_if_test_deps = set()

        if (len(node.targets) == 1 and isinstance(node.targets[0], (ast.Tuple, ast.List))
            and isinstance(node.value, ast.Tuple) and len(node.targets[0].elts) == len(node.value.elts)):
            for t, v in zip(node.targets[0].elts, node.value.elts):
                if isinstance(v, ast.Call) and isinstance(v.func, ast.Name) and v.func.id == "random":
                    if self._assign_random([t], v): continue
                self._assign(t, self._eval(v))
            return

        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == "random":
            if self._assign_random(node.targets, node.value):
                return

        val = self._eval(node.value)
        for t in node.targets:
            self._assign(t, val)

    def visit_AugAssign(self, node: ast.AugAssign):
        # NEW: clear pending-if info
        self._pending_envs_after_if = None
        self._pending_if_test_deps = set()

        base = self._eval(node.target)
        inc  = self._eval(node.value)
        if   isinstance(node.op, ast.Add): self._assign(node.target, av_add(base, inc))
        elif isinstance(node.op, ast.Sub): self._assign(node.target, av_sub(base, inc))
        elif isinstance(node.op, ast.Mult): self._assign(node.target, av_mul(base, inc))
        else: self._assign(node.target, AV(None, set()))

    def visit_If(self, node: ast.If):
        # NEW: clear old pending-if info first
        self._pending_envs_after_if = None
        self._pending_if_test_deps = set()

        save_env = dict(self.env)
        before_then = len(self.returns)
        for s in node.body: self.visit(s)
        env_then = dict(self.env)
        then_returns = self.returns[before_then:]

        self.env = dict(save_env)
        before_else = len(self.returns)
        for s in node.orelse: self.visit(s)
        env_else = dict(self.env)
        else_returns = self.returns[before_else:]

        def av_tuple_equal(a: Union[AV, Tuple[AV,...]], b: Union[AV, Tuple[AV,...]]) -> bool:
            def norm(x):
                return x if isinstance(x, tuple) else (x,)
            A, B = norm(a), norm(b)
            if len(A) != len(B): return False
            for u, v in zip(A, B):
                if (u.poly is not None) and (v.poly is not None):
                    if u.poly != v.poly: return False
                else:
                    if u.deps != v.deps: return False
            return True

        # Control contributors when there are differing in-branch returns
        if then_returns and else_returns:
            equal = True
            for ra in then_returns:
                for rb in else_returns:
                    if not av_tuple_equal(ra, rb):
                        equal = False
                        break
                if not equal: break
            if not equal:
                test_syms = self._eval(node.test).deps
                self._control_contributors = getattr(self, "_control_contributors", set())
                for m in self.meas:
                    if f"{m}@0" in test_syms:
                        self._control_contributors.add(m)

        # Merge environments (variable-wise) for continued execution
        merged: Dict[str, AV] = {}
        keys = set(env_then.keys()) | set(env_else.keys())
        for k in keys:
            a = env_then.get(k); b = env_else.get(k)
            if a is None and b is None:
                continue
            if a is None or b is None:
                only = a or b
                merged[k] = AV(None, set() if only is None else set(only.deps))
                continue
            if (a.poly is not None) and (b.poly is not None) and (a.poly == b.poly):
                merged[k] = a
            elif a.deps == b.deps:
                merged[k] = AV(None, set(a.deps))
            else:
                merged[k] = AV(None, a.deps | b.deps)
        self.env = merged

        # NEW: remember branch envs and the condition deps for a possible *immediate* return
        self._pending_envs_after_if = (env_then, env_else)
        self._pending_if_test_deps = self._eval(node.test).deps

    def visit_Return(self, node: ast.Return):
        # NEW: path-sensitive return immediately after an 'if'
        pend = self._pending_envs_after_if
        if node.value is not None and pend is not None:
            env_t, env_e = pend
            at = self._eval_in_env(env_t, node.value)
            ae = self._eval_in_env(env_e, node.value)
            out = self._join_out(at, ae)

            # if the two returns differ, record control contributors
            if (isinstance(out, AV) and not self._av_equal(at, ae)) or \
               (isinstance(out, tuple) and any(not self._av_equal(x,y) for x,y in zip(at if isinstance(at, tuple) else (at,),
                                                                                      ae if isinstance(ae, tuple) else (ae,)))):
                test_syms = set(self._pending_if_test_deps)
                self._control_contributors = getattr(self, "_control_contributors", set())
                for m in self.meas:
                    if f"{m}@0" in test_syms:
                        self._control_contributors.add(m)

            self.returns.append(out)
            # clear pending info
            self._pending_envs_after_if = None
            self._pending_if_test_deps = set()
            return

        # No pending-if special case → original logic
        if node.value is None:
            self.returns.append(AV(None, set()))
            return
        if isinstance(node.value, ast.Tuple):
            self.returns.append(tuple(self._eval(elt) for elt in node.value.elts))
        else:
            self.returns.append(self._eval(node.value))

        # Clear any stale pending info to be safe
        self._pending_envs_after_if = None
        self._pending_if_test_deps = set()

# ---------- Public API ----------
def find_dead_outcomes(py_src: str, func_name: Optional[str], measurement_vars: List[str]) -> List[str]:
    mod = ast.parse(py_src)
    funcs = [n for n in mod.body if isinstance(n, ast.FunctionDef)]
    if func_name is None:
        if len(funcs) != 1:
            raise ValueError("Provide func_name when multiple functions are present.")
        func = funcs[0]
    else:
        func = next((f for f in funcs if f.name == func_name), None)
        if func is None:
            raise ValueError(f"Function {func_name!r} not found.")

    A = Analyzer(func, set(measurement_vars))
    for s in func.body:
        A.visit(s)

    outs: List[AV] = []
    for r in A.returns or []:
        if isinstance(r, tuple):
            outs.extend(r)
        else:
            outs.append(r)

    if not outs:
        return []

    control = getattr(A, "_control_contributors", set())
    dead: List[str] = []
    for m in measurement_vars:
        if m in control:
            continue
        sym0 = f"{m}@0"
        if all(sym0 not in av.deps for av in outs):
            dead.append(m)
    return dead
