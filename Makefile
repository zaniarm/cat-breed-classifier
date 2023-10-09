install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

lint:
	pylint --disable=R,C src

test:
	pytest tests

format:
	black src/*.py tests/*.py

deploy:
	echo "Deploy Placeholder"

all: install lint test format