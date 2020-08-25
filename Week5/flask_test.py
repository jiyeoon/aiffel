from flask import Flask
app = Flask(__name__)

@app.route("/")
def hello():
    return "플라스크 동작 확인!"

# 코드 선언부
def datetime_decorator(func):
        def decorated():
                print(datetime.datetime.now())
                func()
                print(datetime.datetime.now())
        return decorated

@datetime_decorator
def main_function():
        print ("test function")

if __name__ == "__main__":
    app.run()