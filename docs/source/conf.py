# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from datetime import datetime
from importlib.metadata import version

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Stable Gym"
copyright = f"{datetime.now().year}, Rick Staa"
author = "Rick Staa"
release = version("stable_gym")
version = ".".join(release.split(".")[:3])
print("Doc release: ", release)
print("Doc version: ", version)

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# If your documentation needs a minimal Sphinx version, state it here.
needs_sphinx = "3.0"

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.napoleon",  # Add google docstring support.
    "sphinx.ext.extlinks",  # Simplify linking to external documentation.
    "sphinx.ext.githubpages",  # Allow GitHub Pages rendering.
    "sphinx.ext.intersphinx",  # Link to other Sphinx documentation.
    "sphinx.ext.viewcode",  # Add a link to the Python source code for python objects.
    "myst_parser",  # Support for MyST Markdown syntax.
    "autoapi.extension",  # Generate API documentation from code.
    "sphinx.ext.autodoc",  # Include documentation from docstrings.
]
autoapi_dirs = ["../../stable_gym"]
myst_heading_anchors = 2  # Add anchors to headings.
myst_enable_extensions = ["dollarmath", "html_image"]

# Extensions settings.
autodoc_member_order = "bysource"

# Add mappings.
intersphinx_mapping = {
    "gymnasium": ("https://www.gymlibrary.dev/", None),
    "python3": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path.
exclude_patterns = []

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_logo = "images/logo.svg"
html_favicon = "_static/favicon.ico"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {"logo_only": True}
html_context = {
    "display_github": True,  # Add 'Edit on Github' link instead of 'View page source'
    "github_user": "rickstaa",
    "github_repo": "stable-gym",
    "github_version": "main",
    "conf_py_path": "/docs/source/",  # needs leading and trailing slashes!
}

# -- External links dictionary -----------------------------------------------
# Here you will find some often used global url definitions.
extlinks = {
    "stable_gym": ("https://github.com/rickstaa/stable-gym/%s", None),
    "gymnasium": ("https://gymnasium.farama.org/%s", None),
    "stable_learning_control": (
        "https://github.com/rickstaa/stable-learning-control/%s",
        None,
    ),
    "ros_gazebo_gym": ("https://github.com/rickstaa/ros-gazebo-gym/%s", None),
}


# -- Add extra style sheets --------------------------------------------------
def setup(app):
    app.add_css_file("css/modify.css")
