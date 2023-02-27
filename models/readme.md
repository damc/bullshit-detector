# Models

This is a package that is an abstraction layer over large language models. It contains functions that wrap requests to large language model API (e.g. OpenAI API) endpoints with useful functionality.

Specifically, there are few main concepts that this package is introducing:
a) The prompts are stored in Jinja2 templates. When the user of the package calls `complete`, they pass the name of the template as argument. Those templates are stored in some folder.
b) Each prompt has a config - it stores the parameters used with that prompt.
c) `search` module contains some useful functions for search based on embeddings from OpenAI API.