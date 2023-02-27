from json import loads
from os.path import join
from typing import Dict, Any

from jinja2 import FileSystemLoader, Environment
from openai.openai_object import OpenAIObject

from models.config import config
from models.transformations import include_prompt_last_line, strip_last_line_if_too_long, strip


def render_template(template_name: str, **kwargs) -> str:
    """Render template given template name and template arguments

    Args:
        template_name: template name
        **kwargs: arguments for the template

    Returns:
        constructed prompt
    """
    templates_path = join(config('inputs_path'), config('templates_dir'))
    template_loader = FileSystemLoader(searchpath=templates_path)
    template_env = Environment(loader=template_loader)
    template = template_env.get_template(template_name + ".txt")
    return template.render(**kwargs)


def template_config(name: str) -> Dict[str, Any]:
    """Load configuration for a given template

    Args:
        name: the name of the template

    Returns:
        configuration
    """
    inputs_path = config('inputs_path')
    config_dir = config('config_dir')
    file_path = join(inputs_path, config_dir, name + '.json')
    return loads(open(file_path).read())


TRANSFORMATIONS = {
    'include_prompt_last_line': include_prompt_last_line,
    'strip_last_line_if_too_long': strip_last_line_if_too_long,
    'strip': strip
}

ALLOWED_PARAMETERS = (
    "model",
    "prompt",
    "suffix",
    "max_tokens",
    "temperature",
    "top_p",
    "n",
    "stream",
    "logprobs",
    "echo",
    "stop",
    "presence_penalty",
    "frequency_penalty",
    "best_of",
    "logit_bias",
    "user",
    "input",
    "instruction"
)


def parameters(template_config_: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """Construct parameters to send in the request

    Args:
        template_config_: configuration for the template
        template_config_: configuration coming from the template

    Returns:
        parameters to send in the request
    """
    module_config = config('default_parameters')
    additional = {
        key: value
        for key, value in kwargs.items()
        if key in ALLOWED_PARAMETERS
    }
    parameters_ = {**module_config, **template_config_, **additional}
    for transformation in TRANSFORMATIONS:
        if transformation in parameters_:
            parameters_.pop(transformation)
    if 'provider' in parameters_ and parameters_['provider'] == 'bloom':
        parameters_ = convert_to_bloom_parameters(parameters_)
    return parameters_


BLOOM_PARAMETERS_MAP = {
    "max_tokens": "max_new_tokens"
}


def convert_to_bloom_parameters(
        parameters_: Dict[str, Any]
) -> Dict[str, any]:
    result = {}
    for key, value in parameters_.items():
        if key in BLOOM_PARAMETERS_MAP:
            result[BLOOM_PARAMETERS_MAP[key]] = value
        else:
            result[key] = value
    return result


def template_config_value_equals(
        template_config_: Dict[str, Any],
        key: str,
        value: Any
) -> bool:
    """Check if the template config value equals to some value

    Args:
        template_config_: configuration
        key: key for which we check
        value: value that we want

    Returns:
        True, if configuration contains a key with the given name and
            value.
    """
    return key in template_config_ and template_config_[key] == value


def transformed_text(
        template_config_: Dict[str, Any],
        prompt_: str,
        response: OpenAIObject
) -> str:
    """Transform text from the response from OpenAI API

    Args:
        template_config_: config that contains information which
         transformations we should apply
        prompt_: prompt that has been sent to OpenAI API
        response: response from OpenAI API

    Returns:
        transformed text
    """
    text = response['text']
    for transformation in TRANSFORMATIONS:
        if template_config_value_equals(template_config_, transformation, True):
            text = TRANSFORMATIONS[transformation](prompt_, text, response)
    return text
