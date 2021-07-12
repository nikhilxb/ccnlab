find ccnlab scripts -name '*.py' -print0 |\
xargs -0 \
yapf --in-place --verbose --exclude \"venv/*,__pycache__/*,\"