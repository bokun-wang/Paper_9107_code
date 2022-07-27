"""
Elias Omega code in Python
http://en.wikipedia.org/wiki/Elias_omega_coding
Peter Elias, "Universal codeword sets and representations of the integers", IEEE Trans. Information Theory 21(2):194-203, Mar 1975.
Also known as the log* (or "logstar") code in Rohan Baxter's 1996 PhD thesis.
Type 'python EliasOmega.py' to run the tests.
"""


# A Python function decorator to remember computed values
def __memoize(function):
    memo = {}

    def wrapper(*args):
        if args in memo:
            return memo[args]
        else:
            rv = function(*args)
            memo[args] = rv
            return rv

    return wrapper


@__memoize
def __recursive_encode(n):
    s = ""
    if n > 1:
        b = bin(n)[2:]
        s += __recursive_encode(len(b) - 1) + b
    return s


@__memoize
def __recursive_decode(s, n):
    if s[0] == "0":
        return [n, s[1:]]
    else:
        m = int(s[:n + 1], 2)
        return __recursive_decode(s[n + 1:], m)


def encode(n):
    """Encode a string using the Elias Omega code.
    >>> encode(16)
    '10100100000'
    >>> encode(100)
    '1011011001000'
    >>> encode(1000)
    '11100111111010000'
    >>> encode(1000000)
    '1010010011111101000010010000000'
    >>> [encode(n) for n in range(1, 18)]
    ['0', '100', '110', '101000', '101010', '101100', '101110', '1110000', '1110010', '1110100', '1110110', '1111000', '1111010', '1111100', '1111110', '10100100000', '10100100010']
    """
    return __recursive_encode(n) + "0"


def decode(s):
    """Decode a string containing a binary number encoded using the Elias Omega code.

    >>> decode('1011011001000')
    [100, '']
    >>> all([decode(encode(n))[0]==n for n in range(1, 10000)])
    True
    """

    return __recursive_decode(s, 1)


def codelength(n):
    """Calculate the Elias Omega codelength.
    >>> codelength(17)
    11
    """

    return len(encode(n))


def probability(n):
    """Approximate the probability of n implied by the Elias Omega code.
    >>> probability(2)==1/8.
    True
    """

    return pow(2, -float(codelength(n)))


def _test():
    import doctest
    doctest.testmod()


if __name__ == "__main__":
    _test()