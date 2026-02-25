import pytest
import random
from DLL import zeros, ones, rand


@pytest.fixture(params=range(100))
def setup_data():
    rows = random.randint(1, 20)
    cols = random.randint(1, 20)
    shape = [rows, cols]
    
    a = rand(shape)
    b = rand(shape)
    s = random.uniform(2.0, 5.0) 
    
    return a, b, s, shape

def test_initialization(setup_data):
    _, _, _, shape = setup_data
    size = shape[0] * shape[1]
    
    assert zeros(shape).data == pytest.approx([0.0] * size)
    assert ones(shape).data == pytest.approx([1.0] * size)

def test_addition(setup_data):
    a, b, s, _ = setup_data
    
    assert (a + b).data == pytest.approx([x + y for x, y in zip(a.data, b.data)])
    assert (s + a).data == pytest.approx([s + x for x in a.data])

def test_subtraction(setup_data):
    a, b, s, _ = setup_data
    
    assert (a - b).data == pytest.approx([x - y for x, y in zip(a.data, b.data)])
    assert (s - a).data == pytest.approx([s - x for x in a.data])

def test_multiplication(setup_data):
    a, b, s, _ = setup_data
    
    assert (a * b).data == pytest.approx([x * y for x, y in zip(a.data, b.data)])
    assert (a * s).data == pytest.approx([x * s for x in a.data])

def test_division(setup_data):
    a, b, s, _ = setup_data
    
    assert (a / s).data == pytest.approx([x / s for x in a.data])
    assert (s / a).data == pytest.approx([s / x for x in a.data])

def test_power(setup_data):
    a, b, s, _ = setup_data
    
    assert (a ** s).data == pytest.approx([x ** s for x in a.data])
    assert (s ** a).data == pytest.approx([s ** x for x in a.data])
    assert (a ** b).data == pytest.approx([x ** y for x, y in zip(a.data, b.data)])
