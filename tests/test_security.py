from app.core.security import constant_time_equal

def test_constant_time_equal():
    assert constant_time_equal('secret','secret')
    assert not constant_time_equal('secret','wrong')
