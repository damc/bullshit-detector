from logging import basicConfig, WARNING, INFO, getLogger, DEBUG
from os import environ
from sys import stdout
import random

from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import Column, Integer, String, Float

from detector.main import confidence

DEBUG_MODE = bool(environ['FLASK_DEBUG'])

basicConfig(level=WARNING, stream=stdout)
getLogger("detector.main").setLevel(DEBUG if DEBUG_MODE else INFO)

app = Flask(__name__)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_DATABASE_URI'] = environ['DATABASE_URI']
db = SQLAlchemy(app, engine_options={'pool_pre_ping': True})


class Content(db.Model):
    __tablename__ = 'content'

    id = Column(Integer, primary_key=True)
    content = Column(String(255))
    confidence = Column(Float)

    def __repr__(self):
        return (
                "<Content(content='%s', confidence='%s')>" %
                (self.content, self.confidence)
        )


def save_to_database(content: str, confidence_: float) -> None:
    content_ = Content(content=content, confidence=confidence_)
    db.session.add(content_)
    db.session.commit()


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')


@app.route('/detect', methods=['POST'])
def detect():
    content = request.form['content']
    confidence_ = confidence(content)
    message_ = message(content, confidence_)
    save_to_database(content, confidence_)
    return jsonify({'message': message_})


def message(content: str, confidence_: float) -> str:
    result = evaluation_message(confidence_)
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
    result += (
        "\nI (Bullshit Detector) haven't been empirically validated on a"
        " large dataset, so don't rely on me when the stakes are high."
        " I have been tested on a very small dataset (with good results)."
    )
    return result


def evaluation_message(confidence_: float) -> str:
    if confidence_ < 0.3:
        return (
            "The content is most likely false, especially if it was "
            "generated by AI."
        )
    if confidence_ < 0.6:
        return "The content is more likely to be false, but I'm not sure."
    if confidence_ < 0.9:
        return "The content is more likely to be true, but I'm not sure."
    return (
        "I have strong confidence that the information is true (I still "
        "might be wrong sometimes)."
    )


def create_tables():
    with app.app_context():
        print("Tables to be created:", db.metadata.tables)
        db.create_all()


@app.before_first_request
def init_app():
    create_tables()

if __name__ == '__main__':
    create_tables()
    app.run(debug=DEBUG_MODE)
