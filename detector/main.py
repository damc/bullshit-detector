from logging import getLogger
from os.path import join, dirname

from models.basic import chat_complete
from models.config import update_config


def confidence(content: str, answers: int = 3) -> float:
    question = chat_complete('question', answer=content).strip()
    getLogger("detector.main").info(f"Question: {question}")
    same_answers = 0
    for i in range(answers):
        answer = chat_complete('answer', question=question).strip()
        getLogger("detector.main").debug(f"Answer: {answer}")
        if convey_same_information(content, answer):
            getLogger("detector.main").debug("It conveys the same information")
            same_answers += 1
        else:
            getLogger("detector.main").debug(
                "It doesn't convey the same information"
            )
    result = same_answers / answers
    getLogger("detector.main").info(f"Confidence: {result}")
    return result


def convey_same_information(a: str, b: str) -> bool:
    answer_ = chat_complete('convey_same_information', a=a, b=b)
    return 'yes' in answer_.lower()


model_inputs_path = join(dirname(__file__), 'model_inputs')
models_config = {'inputs_path': model_inputs_path}
update_config(models_config)
