activate:
	.venv\Scripts\activate

install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt
isort:
	isort src/

lint:
	flake8 src

test:
	pytest tests

format:
	black src/*.py tests/*.py

deploy:
	echo "Deploy Placeholder"

all: install format isort lint test