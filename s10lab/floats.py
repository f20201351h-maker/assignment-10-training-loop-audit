"""
Encode a real number into a small binary float format by hand, using exact rational arithmetic.

No float conversion is used in the derivation: the input is fractions.Fraction(1, 10), and every step
(binary expansion, normalisation, exponent bias, mantissa cut, round-to-nearest-even) is done with integers.
The library encodings (struct / torch) are only used afterwards, to check the hand result.
"""
from dataclasses import dataclass
from fractions import Fraction


@dataclass
class Format:
    name: str
    exp_bits: int
    man_bits: int
    bias: int
    note: str = ""


FP32 = Format("fp32", 8, 23, 127, "IEEE 754 binary32")
BF16 = Format("bf16", 8, 7, 127, "bfloat16: fp32's exponent, 7 mantissa bits")
E4M3 = Format("fp8 E4M3 (OCP 'fn')", 4, 3, 7,
              "OCP FP8 E4M3 = torch.float8_e4m3fn: no inf, only S.1111.111 is NaN, max 448")
E5M2 = Format("fp8 E5M2", 5, 2, 15, "IEEE-like, has inf; shown only as a contrast")


def binary_fraction_digits(x: Fraction, n: int):
    """First n binary digits after the point of 0 <= x < 1, by repeated doubling."""
    digits, steps = [], []
    for _ in range(n):
        x *= 2
        d = 1 if x >= 1 else 0
        steps.append((x, d))
        digits.append(d)
        x -= d
    return digits, steps


def normalise(x: Fraction):
    """x = m * 2**e with 1 <= m < 2."""
    e = 0
    while x >= 2:
        x /= 2
        e += 1
    while x < 1:
        x *= 2
        e -= 1
    return x, e


def encode(x: Fraction, fmt: Format):
    """Round-to-nearest-even encoding of a positive normal number. Returns a dict with every intermediate."""
    assert x > 0
    m, e = normalise(x)
    frac = m - 1                              # the part after the implicit leading 1
    scaled = frac * (1 << fmt.man_bits)       # mantissa as a real number of ULPs
    kept = scaled.numerator // scaled.denominator
    rem = scaled - kept                       # what was cut off, in ULPs (0 <= rem < 1)
    if rem > Fraction(1, 2) or (rem == Fraction(1, 2) and kept % 2 == 1):
        rounded, decision = kept + 1, "round up (remainder > 1/2 ULP)" if rem > Fraction(1, 2) else "tie -> even, up"
    else:
        rounded, decision = kept, "round down (remainder < 1/2 ULP)" if rem < Fraction(1, 2) else "tie -> even, down"
    if rounded == (1 << fmt.man_bits):        # mantissa overflowed into the exponent
        rounded, e = 0, e + 1
    biased = e + fmt.bias
    assert 1 <= biased <= (1 << fmt.exp_bits) - 1, "not a normal number in this format"
    if fmt.name.startswith("fp8 E4M3"):
        assert not (biased == 15 and rounded == 7), "S.1111.111 is NaN in E4M3fn"
    sign = 0
    exp_str = format(biased, f"0{fmt.exp_bits}b")
    man_str = format(rounded, f"0{fmt.man_bits}b")
    bits = f"{sign}{exp_str}{man_str}"
    value = (1 + Fraction(rounded, 1 << fmt.man_bits)) * (Fraction(2) ** e)
    cut_digits, _ = binary_fraction_digits(rem, 8)
    return {
        "format": fmt.name, "sign": sign, "exp_bits": fmt.exp_bits, "man_bits": fmt.man_bits, "bias": fmt.bias,
        "unbiased_exp": e, "biased_exp": biased, "exp_field": exp_str,
        "mantissa_kept_before_rounding": format(kept, f"0{fmt.man_bits}b"),
        "first_dropped_bits": "".join(map(str, cut_digits)),
        "rounding": decision, "mantissa_field": man_str,
        "bits": bits, "bits_grouped": f"{sign} | {exp_str} | {man_str}",
        "hex": f"0x{int(bits, 2):0{(len(bits) + 3) // 4}X}",
        "value_exact": value, "value": float(value),
        "abs_error": float(abs(value - x)), "rel_error": float(abs(value - x) / x),
        "ulp_at_value": float(Fraction(2) ** (e - fmt.man_bits)),
    }


def decode(bits: str, fmt: Format):
    """Independent decoder: bits string -> exact value (normal numbers only)."""
    s, e, m = int(bits[0]), int(bits[1:1 + fmt.exp_bits], 2), int(bits[1 + fmt.exp_bits:], 2)
    assert 0 < e < (1 << fmt.exp_bits) - 1 or fmt.name.startswith("fp8 E4M3")
    return (-1) ** s * (1 + Fraction(m, 1 << fmt.man_bits)) * Fraction(2) ** (e - fmt.bias)


def format_range(fmt: Format):
    """Largest finite, smallest normal, smallest subnormal and decimal digits of a format."""
    if fmt.name.startswith("fp8 E4M3"):
        max_finite = (1 + Fraction(6, 8)) * Fraction(2) ** (15 - fmt.bias)  # 1.110 x 2^8 = 448
    else:
        max_finite = (2 - Fraction(1, 1 << fmt.man_bits)) * Fraction(2) ** ((1 << fmt.exp_bits) - 2 - fmt.bias)
    min_normal = Fraction(2) ** (1 - fmt.bias)
    min_sub = Fraction(2) ** (1 - fmt.bias - fmt.man_bits)
    import math
    return {"format": fmt.name, "max_finite": float(max_finite), "min_normal": float(min_normal),
            "min_subnormal": float(min_sub), "decimal_digits": round((fmt.man_bits + 1) * math.log10(2), 2)}
