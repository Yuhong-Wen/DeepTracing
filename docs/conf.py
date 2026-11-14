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

html_static_path = ["_static"]
html_css_files = ["custom.css"]

rst_prolog = """
.. role:: coming-soon
   :class: coming-soon
"""