from adapdomain.runtime.ring_buffer import RingBuffer


def test_ring_buffer_drops_oldest_after_capacity():
    rb = RingBuffer(capacity=3)
    for v in [1, 2, 3, 4]:
        rb.push(v)
    assert list(rb.snapshot()) == [2, 3, 4]
