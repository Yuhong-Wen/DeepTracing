project = "DeepTracing"

author = "DeepTracing contributors"

extensions = [
    "nbsphinx",
]

html_theme = "sphinx_rtd_theme"

# Show saved notebook outputs without executing the cells.
nbsphinx_execute = "never"

# Keep navigation concise.
html_title = "DeepTracing Documentation"

exclude_patterns = [
    "_build",
    "**/.ipynb_checkpoints",
]