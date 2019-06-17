## Author: Ryan Gibbard
## Algorithm reference:
## https://en.wikibooks.org/wiki/Vehicle_Identification_Numbers_(VIN_codes)/Check_digit

import os
import logging
import operator
import unittest
import random
import string
from typing import Iterable, Callable, Tuple, Iterator, Union
from functools import reduce, partial

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)

trans_key = {
    'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8,
    'J': 1, 'K': 2, 'L': 3, 'M': 4, 'N': 5, 'O': 6, 'P': 7, 'R': 9,
    'S': 2, 'T': 3, 'U': 4, 'V': 5, 'W': 6, 'X': 7, 'Y': 8, 'Z': 9
}

# applys functions in order of appearance, the return of
# each is passed to the next.
def compose(*funcs: Callable) -> Callable:
    def inner(f_inner, f_outer):
        return lambda x: f_outer(f_inner(x))
    return reduce(inner, funcs, lambda x: x)


def is_legal_value(val: str) -> bool:
    return (
        val.upper() in trans_key.keys() or
        str.isdigit(val)
    )


def is_legal_string(vin: str) -> bool:
    return all(is_legal_value(i) for i in vin)


def normalize(vin: str) -> Iterator[Union[int, str]]:
    return map(lambda x: x.upper() if isinstance(x, str) else x, vin)


def translate(val: Union[int, str]) -> str:
    return str(trans_key.get(val, val))


def transliterate(vin: Iterable[Union[int, str]]) -> str:
    return reduce(operator.concat, map(translate, vin))


def multiply_pair(pair: Tuple[str]) -> int:
    return int(pair[0]) * int(pair[1])


def apply_weights(vin: str) -> Tuple[int]:
    weights = (8, 7, 6, 5, 4, 3, 2, 10, 0, 9, 8, 7, 6, 5, 4, 3, 2)
    return tuple(map(multiply_pair, zip(vin, weights)))


def apply_modulus(val: int, mod=11) -> int:
    return val % mod


def is_chk_digit_equal(vin: str, chk: str) -> bool:
    return vin[8] == chk


def is_seventeen_digits(val: str) -> bool:
    return len(val) == 17


def vin_chk_digit_to_str(val: int) -> str:
    return str(val) if val != 10 else "X"


def random_digit_str(len_: str) -> str:
    return ''.join(random.choices(string.digits, k=len_))


def new_random_digit_str(old_val: str) -> str:
    new_val = random_digit_str(len(old_val))

    if new_val != old_val:
        return new_val
    else:
        return new_random_digit_str(len(old_val))


def log_return(*vals: str) -> str:
    logger.debug(vals)
    return vals[-1]


def compute_check_digit(vin: str) -> str:
    pipeline = compose(
                normalize,
                transliterate,
                apply_weights,
                sum,
                apply_modulus,
                vin_chk_digit_to_str,
                partial(log_return, vin)
              )
    return pipeline(vin)


def is_valid_vin(vin: str) -> bool:
    return (
        is_seventeen_digits(vin) and
        is_legal_string(vin) and
        is_chk_digit_equal(vin, compute_check_digit(vin))
    )


def expected_check_digit(vin: str) -> str:
    if not is_seventeen_digits(vin):
        raise ValueError("VIN must be 17 chars")
    if not is_legal_string(vin):
        raise ValueError("VIN contains illegal chars")

    return compute_check_digit(vin)


def scramble_vin(vin: str) -> str:
    country_code = slice(0,1)
    manufacturer_id = slice(1,3)
    vehicle_descriptor = slice(3,8)
    check_digit = slice(8,9)
    model_year = slice(9,10)
    assembly_plant = slice(10,11)
    production_seq = slice(-6, None)
    new_prod_seq = new_random_digit_str(vin[production_seq])

    first_half = vin[country_code] + vin[manufacturer_id] + vin[vehicle_descriptor]
    second_half = vin[model_year] + vin[assembly_plant] + new_prod_seq
    interm_vin = first_half + '0' + second_half

    return first_half + expected_check_digit(interm_vin) + second_half


# tests #

class Tests(unittest.TestCase):

    valid_vins =   ["1M8GDM9AXKP042788", "1D7HG48N44S594243",
                    "5GAKRDED0CJ396612", "1g1Jc524417418958",
                    "3GTU2VEC9EG503024", "11111111111111111",
                    "5GAKRDED9CJ396611"]
    invalid_vins = ["1V2UR2CA0KC514873", "5GAKRDED0CJ396611",
                    "5GAKRDED0CJ39665",  "5GAKRDED0CJ39661I",
                    "3GTU2VEC9EG503021", "ZZZZZZZZZZZZZZZZZ"]

    def test_valid_vin(self):
        for vin in self.valid_vins:
            self.assertTrue(is_valid_vin(vin))

    def test_invalid_vin(self):
        for vin in self.invalid_vins:
            self.assertFalse(is_valid_vin(vin))

    def test_scramble_vin(self):
        for vin in self.valid_vins:
            self.assertTrue(is_valid_vin(scramble_vin(vin)))


if __name__ == '__main__':
    unittest.main()
