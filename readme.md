This repository is the source code of https://bullshitdetector.tech .

The actual code detecting hallucination is contained by `detector/main.py` -> `confidence` function.

It uses `models` package which is a an abstraction layer over the use of LLMs that I have developed when working on CodeAssist ( https://codeassist.tech ) project. The `models` package contains a lot of code that is not needed in this project, because I simply copied over that package from the CodeAssist project (which is much more complex than this project and requires more functionality).

The templates of the used prompts (that are executed with `chat_complete` function) are contained by `detector/model_inputs/templates` folder.

The config of the prompts is contained by `detector/model_inputs/config` folder.