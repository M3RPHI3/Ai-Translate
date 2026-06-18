from flask import Flask, render_template, jsonify
import cv2
import mediapipe as mp
import tst

app = Flask(__name__)

latest_word = "Waiting..."

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/gesture")
def gesture():
    return jsonify({
        "word": latest_word
    })

if __name__ == "__main__":
    app.run(debug=True)