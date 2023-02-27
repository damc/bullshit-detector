from typing import Dict, Any, Union

from openai.openai_object import OpenAIObject


def include_prompt_last_line(
        prompt_: str,
        text: str,
        _response: Union[OpenAIObject, Dict[str, Any]]
) -> str:
    if prompt_.endswith(('\r\n', '\n')):
        return text
    return prompt_.splitlines()[-1] + text


def strip_last_line_if_too_long(
        _prompt_: str,
        text: str,
        response: Union[OpenAIObject, Dict[str, Any]]
) -> str:
    if response['finish_reason'] == 'length':
        return "\n".join(text.splitlines()[:-1])
    return text


def strip(
        _prompt_: str,
        text: str,
        _response: Union[OpenAIObject, Dict[str, Any]]
) -> str:
    return text.strip()
