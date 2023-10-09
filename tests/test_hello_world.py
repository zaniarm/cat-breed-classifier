from src.hello_world import hello_world


def test_hello_world():
    return_value = hello_world()
    assert return_value == 0
