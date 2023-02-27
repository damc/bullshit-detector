from logging import getLogger
from os import environ
from time import sleep
from typing import Dict, Any, List, Optional

from huggingface_hub import InferenceApi
from openai import Completion, Edit, Embedding as OpenAIEmbedding, Moderation
from openai.error import RateLimitError, InvalidRequestError, APIError
from openai.openai_object import OpenAIObject

from .events import dispatch_event
from .templates import render_template, template_config, parameters, transformed_text


# from .redis import cache


def complete(input_name: str, **kwargs) -> str:
    """Send a request to API to generate completion

    Args:
        input_name: name of the prompt to use
        **kwargs: arguments passed to the prompt template

    Returns:
        text generated in the response from OpenAI API
    """
    getLogger("models.basic").debug(f"Complete: {input_name}")
    prompt_ = render_template(input_name + "_prompt", **kwargs)
    prompt_ = prompt_[-22000:]
    template_config_ = template_config(input_name)
    parameters_ = parameters(template_config_, prompt=prompt_, **kwargs)
    try:
        if 'provider' in parameters_ and parameters_['provider'] == 'bloom':
            response = bloom_raw_complete(parameters_)
        else:
            response = openai_raw_complete(parameters_)
    except RateLimitError:
        getLogger("models.basic").debug("Retrying due to rate limit")
        sleep(5)
        return complete(input_name, **kwargs)
    log_complete_response(prompt_, response)
    return transformed_text(template_config_, prompt_, response)


class BloomError(Exception):
    pass


def bloom_raw_complete(parameters_: Dict[str, Any]) -> Dict[str, Any]:
    if 'provider' in parameters_:
        parameters_.pop('provider')
    try:
        token = environ['HUGGING_FACE_TOKEN']
    except KeyError:
        raise KeyError(
            "HUGGING_FACE_TOKEN environment variable must be set"
        )
    inference = InferenceApi(repo_id="bigscience/bloom", token=token)
    print(parameters_)
    prompt = parameters_['prompt']
    parameters_.pop('prompt')
    response = inference(prompt, parameters_)
    if 'error' in response:
        raise BloomError(response['error'])
    result = {'text': response[0]['generated_text']}
    result['text'] = result['text'][len(prompt):]
    result['text'] = result['text'].split(parameters_['stop'], 1)[0]
    getLogger("models.basic").debug("Complete")
    getLogger("models.basic").debug(f"Prompt: {result['text']}")
    return result


def openai_raw_complete(
        parameters_: Dict[str, Any],
        retry_if_not_stop: int = 0,
        delay: int = 0
) -> Optional[OpenAIObject]:
    if delay:
        sleep(delay)
    if 'provider' in parameters_:
        parameters_.pop('provider')
    response = None
    for i in range(retry_if_not_stop + 1):
        try:
            response = Completion.create(**parameters_)['choices'][0]
        except RateLimitError:
            delay = delay * 2 if delay else 5
            getLogger("models.basic").info("Retrying due to rate limit")
            getLogger("models.basic").info(f"Delay: {str(delay)}")
            dispatch_event("rate_limit")
            return openai_raw_complete(
                parameters_,
                retry_if_not_stop - i,
                delay
            )
        except APIError:
            delay = delay * 2 if delay else 5
            getLogger("models.basic").debug("Retrying due to APIError")
            getLogger("models.basic").info(f"Delay: {str(delay)}")
            dispatch_event("rate_limit")
            return openai_raw_complete(
                parameters_,
                retry_if_not_stop - i,
                delay
            )
        if response['finish_reason'] == 'stop':
            break
        if i != retry_if_not_stop:
            getLogger("models.basic").debug("Retrying due to finish reason")
    return response


def log_complete_response(prompt_: str, response: OpenAIObject):
    """Log response from API

    Args:
        prompt_: prompt that has been sent
        response: response from API
    """
    getLogger("models.basic").debug("Complete")
    getLogger("models.basic").debug(f"Prompt: {prompt_}")
    getLogger("models.basic").debug(f"Response: {response['text']}")
    getLogger("models.basic").debug(f"Response: {response['finish_reason']}")


def complete_with_suffix(
        input_name: str,
        retry_if_not_stop: int = 0,
        **kwargs
) -> str:
    """Send a request to OpenAI API to generate completion with suffix

    The model will generate completion in place of '[insert]' from the
    prompt.

    Args:
        input_name: name of the prompt to use
        retry_if_not_stop: if the finish reason is not stop, then it
            will generate once again. This is the maximum number of
            retries.
        **kwargs: arguments passed to the prompt template

    Returns:
        text generated in the response from OpenAI API
    """
    if retry_if_not_stop < 0:
        ValueError("retry_if_not_stop must be at least 0")
    prompt_ = render_template(input_name + "_prompt", **kwargs)
    try:
        prompt_, suffix = prompt_.split('[insert]', 1)
    except ValueError:
        raise ValueError(
            "When using complete_with_suffix prompt must contain '[insert]'."
        )
    template_config_ = template_config(input_name)
    parameters_ = parameters(
        template_config_,
        prompt=prompt_,
        suffix=suffix,
        **kwargs
    )
    response = openai_raw_complete(parameters_, retry_if_not_stop)
    log_complete_with_suffix_response(prompt_, suffix, response)
    return transformed_text(template_config_, prompt_, response)


def log_complete_with_suffix_response(
        prompt_: str,
        suffix: str,
        response: OpenAIObject
):
    """Log response from API

    Args:
        prompt_: prompt that has been sent
        suffix: suffix that has been sent
        response: response from API
    """
    getLogger("models.basic").debug("Complete with suffix")
    getLogger("models.basic").debug(f"Prompt: {prompt_}")
    getLogger("models.basic").debug(f"Suffix: {suffix}")
    getLogger("models.basic").debug(f"Response: {response['text']}")
    getLogger("models.basic").debug(f"Response: {response['finish_reason']}")


class CouldNotEditError(Exception):
    pass


def edit(input_name: str, **kwargs) -> str:
    """Send a request to OpenAI API to generate edition

    Args:
        input_name: name of the prompt to use
        **kwargs: arguments passed to the prompt template

    Returns:
        text generated in the response from OpenAI API
    """
    input_ = render_template(input_name + "_input", **kwargs)
    instruction = render_template(input_name + "_instruction", **kwargs)
    template_config_ = template_config(input_name)
    additional_parameters = {
        'input': input_,
        'instruction': instruction,
        **kwargs
    }
    parameters_ = parameters(
        template_config_,
        **additional_parameters
    )
    try:
        response = Edit.create(**parameters_)['choices'][0]
    except RateLimitError:
        getLogger("models.basic").debug("Retrying due to rate limit")
        sleep(5)
        return edit(input_name, **kwargs)
    except APIError:
        getLogger("models.basic").debug("Retrying due to APIError")
        sleep(5)
        return edit(input_name, **kwargs)
    except InvalidRequestError as error:
        if str(error).startswith("Could not edit text"):
            getLogger("models.basic").debug("Could not edit")
            raise CouldNotEditError
        raise error
    log_edit_response(input_, instruction, response)
    return response["text"]


def log_edit_response(input_: str, instruction: str, response: OpenAIObject):
    """Log edit response from API

    Args:
        input_: input passed to API
        instruction: instruction passed to API
        response: response from API
    """
    getLogger("models.basic").debug("Edit")
    getLogger("models.basic").debug(f"Input: {input_}")
    getLogger("models.basic").debug(f"Instruction: {instruction}")
    getLogger("models.basic").debug(f"Response: {response['text']}")


Embedding = List[float]


# @cache.cache()
def embedding(model: Optional[str], input_: str, **kwargs) -> Embedding:
    if model is None:
        model = 'text-embedding-ada-002'
    try:
        full_response = OpenAIEmbedding.create(
            model=model,
            input=input_,
            **kwargs
        )
        response = full_response['data'][0]
    except RateLimitError:
        getLogger("models.basic").debug("Retrying due to rate limit")
        sleep(5)
        return embedding(model, input_, **kwargs)
    log_embedding_response(input_)
    return response['embedding']


def log_embedding_response(input_: str):
    getLogger("models.basic").debug("Embedding")
    getLogger("models.basic").debug(f"Input: {input_}")


def flagged(input_: str, **kwargs) -> bool:
    try:
        response = Moderation.create(input_, **kwargs)
    except RateLimitError:
        getLogger("models.basic").debug("Retrying due to rate limit")
        sleep(5)
        return flagged(input_, **kwargs)
    except APIError:
        getLogger("models.basic").debug("Retrying due to api error")
        sleep(5)
        return flagged(input_, **kwargs)
    log_flagged_response(input_, response)
    return response["results"][0]["flagged"]


def log_flagged_response(input_: str, response: OpenAIObject):
    """Log flagged response from API

    Args:
        input_: input passed to API
        response: response from API
    """
    getLogger("models.basic").debug("Flagged")
    getLogger("models.basic").debug(f"Input: {input_}")
    getLogger("models.basic").debug(
        f"Response: {response['results'][0]['flagged']}"
    )
