# -*- coding: utf-8 -*-
#

import sys
import os

sys.path.insert(0, os.path.abspath("../"))

extensions = [
    "autoapi.extension",
    "sphinx.ext.todo",
    "sphinx.ext.napoleon",
    "sphinx.ext.graphviz",
    "sphinx.ext.inheritance_diagram",
    "sphinx_multiversion",
    "m2r",
]
source_suffix = [".rst", ".md"]
master_doc = "index"
project = "wandb-allennlp"
copyright = "Dhruvesh Patel"
exclude_patterns = ["_build", "**/docs", "**/.docs", "**/tests", "tests/**"]
pygments_style = "sphinx"
templates_path = ["templates"]  # needed for multiversion
autoclass_content = "class"
html_baseurl = "http://dhruveshp.com/wandb-allennlp/"
html_logo = "images/banner.png"
html_theme_options = {
    "github_user": "dhruvdcoder",
    "github_repo": "wandb-allennlp",
    "github_banner": True,
    "github_button": True,
}

# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#confval-autoclass_content
autoclass_content = "both"
# autodoc_default_options = {'undoc-members': True}

# API Generation
autoapi_dirs = ["../wandb_allennlp"]
autoapi_root = "."
autoapi_options = [
    "members",
    "inherited-members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
]
autoapi_add_toctree_entry = False
autoapi_keep_files = True

# see: https://github.com/data-describe/data-describe/blob/master/docs/source/conf.py
# and https://github.com/data-describe/data-describe/blob/master/docs/make.py
# multiversion
# Multiversioning
smv_tag_whitelist = r"^v\d+\.\d+\.[456789]+b?\d*$"
smv_branch_whitelist = r"^.*master$"
smv_remote_whitelist = r"^.*$"
templates_path = [
    "templates",
]
html_sidebars = {
    "**": [
        "about.html",
        "navigation.html",
        "relations.html",
        "searchbox.html",
        "versioning.html",
    ]
}
