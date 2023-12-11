This repository is the source code of https://bullshitdetector.tech .

The actual code detecting hallucination is contained by `detector/main.py` -> `confidence` function.

It uses `models` package which is a an abstraction layer over the use of LLMs that I have developed when working on CodeAssist ( https://codeassist.tech ) project. The `models` package contains a lot of code that is not needed in this project, because I simply copied over that package from the CodeAssist project (which is much more complex than this project and requires more functionality).

The templates of the used prompts (that are executed with `chat_complete` function) are contained by `detector/model_inputs/templates` folder.

The config of the prompts is contained by `detector/model_inputs/config` folder.

# How does it work?

It programmatically reverses the content to a question and then generates few answers to the question with a high softmax temperature. If the answers convey the same message as the content, the content is likely to be true because that means that the model has high confidence that this is the truth. If the model is not confident about the answer, it will generate a different answer every time.

I think there's also another, cheaper way to do this that I've described somewhere else.
