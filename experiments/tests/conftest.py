import sys

# Upstream tracks __pycache__/*.pyc in git; importing its modules during tests must not rewrite them.
sys.dont_write_bytecode = True
