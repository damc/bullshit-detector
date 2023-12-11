from logging import getLogger
from os.path import join, dirname

from models.basic import chat_complete
from models.config import update_config


def confidence(content: str, answers: int = 3) -> float:
    """Confidence that the `content` is not a hallucination"""
    question = chat_complete('question', answer=content).strip()
    same_answers = 0
    for i in range(answers):
        answer = chat_complete('answer', question=question).strip()
        if convey_same_information(content, answer):
            same_answers += 1
    return same_answers / answers


def convey_same_information(a: str, b: str) -> bool:
    """Does `a` and `b` mean the same but with different words"""
    answer_ = chat_complete('convey_same_information', a=a, b=b)
    return 'yes' in answer_.lower()


model_inputs_path = join(dirname(__file__), 'model_inputs')
models_config = {'inputs_path': model_inputs_path}
update_config(models_config)
