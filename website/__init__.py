from logging import basicConfig, WARNING, DEBUG, getLogger
from sys import stdout

from flask import Flask, render_template, request

from detector.main import confidence

basicConfig(level=WARNING, stream=stdout)
getLogger("detector.main").setLevel(DEBUG)

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    content = ''
    confidence_ = None
    if request.method == 'POST':
        content = request.form['content']
        confidence_ = confidence(content)
    return render_template(
        'index.html',
        content=content,
        confidence=confidence_
    )

if __name__ == '__main__':
    app.run(debug=True)

