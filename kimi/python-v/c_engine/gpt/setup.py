from setuptools import setup, Extension

module = Extension(
    "xqai",
    sources=[
        "xqai.c",
        "board.c",
        "movegen.c",
        "evaluate.c",
        "search.c",
        "rules.c",
        "zobrist.c",
    ],
    extra_compile_args=["-O2"],
)

setup(
    name="xqai",
    version="1.0",
    description="Chinese Chess AI (C backend)",
    ext_modules=[module],
)
