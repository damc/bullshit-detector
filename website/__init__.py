from logging import basicConfig, WARNING, INFO, getLogger, DEBUG
from sys import stdout

from flask import Flask, render_template, request, jsonify

from detector.main import confidence

basicConfig(level=WARNING, stream=stdout)
getLogger("detector.main").setLevel(DEBUG)

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')


@app.route('/detect', methods=['POST'])
def detect():
    content = request.form['content']
    message_ = message(content)
    return jsonify({'message': message_})


def message(content: str) -> str:
    result = evaluation_message(content)
    if len(content) > 200:
        result += (
            "\nHowever, this content is long. For best results, keep it "
            "short and make it contain only the information that you want to "
            "verify."
        )
    result += (
        "\nKeep in mind that I have knowledge as of 2021, so if anything has "
        "changed after that year, my answer will not reflect that."
    )
    return result


def evaluation_message(content: str) -> str:
    confidence_ = confidence(content)
    if confidence_ < 0.3:
        return "The chances that the information is true are about 25%."
    if confidence_ < 0.6:
        return "The chances that the information is true are about 40%."
    if confidence_ < 0.9:
        return "The chances that the information is true are about 70%."
    return "The chances that the information is true are about 95%."


if __name__ == '__main__':
    app.run(debug=True)
