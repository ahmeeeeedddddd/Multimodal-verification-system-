from permissions import execute

cases = [
    ("yes",  "basmala", True),    # admin unlock    -> OK
    ("on",   "basmala", True),    # admin lights    -> OK
    ("no",   "someone", True),    # user delete     -> DENIED
    ("on",   "someone", True),    # user lights     -> OK
    ("yes",  "basmala", False),   # not verified    -> DENIED
    (None,   "basmala", True),    # not understood
]
for word, ident, ok in cases:
    print(f"{str(word):5s} {ident:8s} verified={ok!s:5s} ->",
          execute(word, ident, ok).message)