/*
 * Floating-point number parsing implementation for musl C library
 *
 * This module implements high-precision floating-point parsing from strings,
 * supporting decimal and hexadecimal formats. It's used by strtof/strtod/strtold
 * and scanf family functions.
 *
 * Key algorithms:
 * - Uses a "billion-based" (B1B) internal representation where each array
 *   element stores up to 9 decimal digits (up to 999,999,999)
 * - Handles arbitrary precision during parsing, then rounds to target precision
 * - Supports both decimal (1.23e45) and hexadecimal (0x1.23p45) formats
 * - Implements correct rounding according to IEEE 754 standards
 */

#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <limits.h>
#include <errno.h>
#include <ctype.h>

#include "shgetc.h"
#include "floatscan.h"

/*
 * Configuration for different long double representations
 * LD_B1B_DIG: Number of billion-based digits needed for long double precision
 * LD_B1B_MAX: Maximum values for each billion-based digit position
 * KMAX: Maximum number of billion-based digits we can store
 */
#if LDBL_MANT_DIG == 53 && LDBL_MAX_EXP == 1024
/* IEEE 754 double precision (used when long double == double) */
#define LD_B1B_DIG 2
#define LD_B1B_MAX 9007199, 254740991
#define KMAX 128

#elif LDBL_MANT_DIG == 64 && LDBL_MAX_EXP == 16384
/* x87 80-bit extended precision */
#define LD_B1B_DIG 3
#define LD_B1B_MAX 18, 446744073, 709551615
#define KMAX 2048

#elif LDBL_MANT_DIG == 113 && LDBL_MAX_EXP == 16384
/* IEEE 754 quadruple precision */
#define LD_B1B_DIG 4
#define LD_B1B_MAX 10384593, 717069655, 257060992, 658440191
#define KMAX 2048

#else
#error Unsupported long double representation
#endif

/* Circular buffer mask for billion-based digit array */
#define MASK (KMAX-1)

/* Constants for decimal parsing */
#define BILLION 1000000000
#define DIGITS_PER_B1B 9  /* 9 decimal digits per billion-based digit */

/*
 * Parse an exponent from the input stream (e.g., the "123" in "e123" or "p123")
 *
 * Parameters:
 *   f: Input stream
 *   pok: If true, partial parsing is OK (for scanf-style parsing)
 *
 * Returns:
 *   The exponent value, or LLONG_MIN if no valid exponent was found
 */
static long long scan_exponent(FILE *f, int partial_ok)
{
	int c;
	int exponent_int;           /* Exponent value as int (for overflow detection) */
	long long exponent_result;  /* Final exponent value */
	int is_negative = 0;

	c = shgetc(f);

	/* Handle optional sign */
	if (c == '+' || c == '-') {
		is_negative = (c == '-');
		c = shgetc(f);
		/* If partial parsing is OK and we don't find a digit, back up */
		if (c - '0' >= 10U && partial_ok) {
			shunget(f);
		}
	}

	/* Must have at least one digit */
	if (c - '0' >= 10U) {
		shunget(f);
		return LLONG_MIN;  /* Signal: no valid exponent found */
	}

	/* Parse exponent, watching for overflow */
	/* First phase: accumulate in int to detect overflow early */
	for (exponent_int = 0; c - '0' < 10U && exponent_int < INT_MAX/10; c = shgetc(f)) {
		exponent_int = 10 * exponent_int + (c - '0');
	}

	/* Second phase: continue in long long for larger values */
	for (exponent_result = exponent_int; c - '0' < 10U && exponent_result < LLONG_MAX/100; c = shgetc(f)) {
		exponent_result = 10 * exponent_result + (c - '0');
	}

	/* Skip any remaining digits (they would cause overflow anyway) */
	for (; c - '0' < 10U; c = shgetc(f)) {
		/* Just consume the digits */
	}

	shunget(f);  /* Put back the non-digit character */
	return is_negative ? -exponent_result : exponent_result;
}


/*
 * Parse a decimal floating-point number (e.g., "123.456e78")
 *
 * This is the core decimal parsing function. It uses a "billion-based" internal
 * representation where each array element can hold up to 999,999,999 (9 decimal digits).
 * This allows efficient handling of very long decimal numbers.
 *
 * Algorithm overview:
 * 1. Parse digits into billion-based array (each element = up to 9 decimal digits)
 * 2. Handle decimal point and exponent to determine radix position
 * 3. Scale the number to get the right number of significant bits
 * 4. Convert to target floating-point format with proper rounding
 *
 * Variable naming guide:
 * - x[]: billion-based digit array (each element holds up to 9 decimal digits)
 * - j: current position within a billion-based digit (0-8)
 * - k: current index in the x[] array
 * - a: start index of significant digits in x[] (circular buffer)
 * - z: end index of significant digits in x[] (circular buffer)
 * - dc: total decimal digit count
 * - lrp: logical radix position (position of decimal point)
 * - lnz: position of last non-zero digit
 * - rp: radix position in billion-based representation
 * - e2: binary exponent for final result
 * - e10: decimal exponent from 'e' notation
 *
 * Parameters:
 *   f: Input stream
 *   c: First character already read
 *   bits: Target precision in bits (24 for float, 53 for double, etc.)
 *   emin: Minimum exponent for target type
 *   sign: Sign of the number (+1 or -1)
 *   pok: If true, partial parsing is OK (for scanf-style parsing)
 *
 * Returns:
 *   The parsed floating-point value
 */
static long double decfloat(FILE *f, int c, int bits, int emin, int sign, int pok)
{
	/* Billion-based digit array - each element holds up to 9 decimal digits */
	uint32_t x[KMAX];

	/* Threshold values for determining when we have enough precision */
	static const uint32_t th[] = { LD_B1B_MAX };

	/* Loop counters and array indices */
	int i, j, k, a, z;

	/* Position tracking variables */
	long long lrp = 0;    /* Logical radix position (decimal point position) */
	long long dc = 0;     /* Total decimal digit count */
	long long e10 = 0;    /* Decimal exponent from 'e' notation (e.g., "e123") */
	int lnz = 0;          /* Position of last non-zero digit */

	/* Parsing state flags */
	int gotdig = 0;       /* Found at least one digit */
	int gotrad = 0;       /* Found decimal point */

	/* Working variables for final conversion */
	int rp;               /* Radix position in billion-based representation */
	int e2;               /* Binary exponent for final result */
	int emax = -emin - bits + 3;  /* Maximum allowed exponent */
	int denormal = 0;     /* Whether result will be denormal */
	long double y;        /* Final result accumulator */
	long double frac = 0; /* Fractional part for rounding */
	long double bias = 0; /* Rounding bias */

	/* Powers of 10 for quick calculations */
	static const int p10s[] = { 10, 100, 1000, 10000,
		100000, 1000000, 10000000, 100000000 };

	/* Initialize parsing state */
	j = 0;  /* Position within current billion-based digit (0-8) */
	k = 0;  /* Current index in the x[] array */

	/* Skip leading zeros - they don't consume buffer space */
	for (; c == '0'; c = shgetc(f)) {
		gotdig = 1;
	}

	/* Handle leading zeros after decimal point */
	if (c == '.') {
		gotrad = 1;
		for (c = shgetc(f); c == '0'; c = shgetc(f)) {
			gotdig = 1;
			lrp--;  /* Each leading zero moves radix left */
		}
	}

	/* Initialize the billion-based digit array */
	x[0] = 0;

	/* Main digit parsing loop - collect digits into billion-based representation */
	for (; (c - '0' < 10U) || c == '.'; c = shgetc(f)) {
		if (c == '.') {
			/* Handle decimal point */
			if (gotrad) break;  /* Second decimal point - stop parsing */
			gotrad = 1;
			lrp = dc;  /* Record position of decimal point */
		} else if (k < KMAX - 3) {
			/* We have space in our buffer for more digits */
			dc++;  /* Increment total digit count */
			if (c != '0') {
				lnz = dc;  /* Update position of last non-zero digit */
			}

			/* Add digit to current billion-based element */
			if (j) {
				/* Not the first digit in this element - multiply by 10 and add */
				x[k] = x[k] * 10 + (c - '0');
			} else {
				/* First digit in this element */
				x[k] = c - '0';
			}

			/* Move to next billion-based element if current one is full (9 digits) */
			if (++j == 9) {
				k++;
				j = 0;
			}
			gotdig = 1;
		} else {
			/* Buffer is full - just track whether we have more non-zero digits */
			dc++;
			if (c != '0') {
				lnz = (KMAX - 4) * 9;  /* Mark significant digits beyond our buffer */
				x[KMAX - 4] |= 1;     /* Set a flag that we lost precision */
			}
		}
	}

	/* If no decimal point was found, radix is at the end */
	if (!gotrad) {
		lrp = dc;
	}

	/* Parse optional exponent (e.g., "e123" or "E-45") */
	if (gotdig && (c|32) == 'e') {
		e10 = scan_exponent(f, pok);
		if (e10 == LLONG_MIN) {
			/* No valid exponent found */
			if (pok) {
				shunget(f);  /* Back up for partial parsing */
			} else {
				shlim(f, 0);  /* Signal error */
				return 0;
			}
			e10 = 0;
		}
		lrp += e10;  /* Adjust logical radix position by exponent */
	} else if (c >= 0) {
		shunget(f);  /* Put back the non-digit character */
	}

	/* Must have found at least one digit */
	if (!gotdig) {
		errno = EINVAL;
		shlim(f, 0);
		return 0;
	}

	/* Handle zero specially to avoid complex processing */
	if (!x[0]) {
		return sign * 0.0;
	}

	/* Fast path for small integers without exponent */
	if (lrp == dc && dc < 10 && (bits > 30 || x[0] >> bits == 0)) {
		return sign * (long double)x[0];
	}

	/* Check for overflow/underflow */
	if (lrp > -emin/2) {
		errno = ERANGE;
		return sign * LDBL_MAX * LDBL_MAX;
	}
	if (lrp < emin - 2*LDBL_MANT_DIG) {
		errno = ERANGE;
		return sign * LDBL_MIN * LDBL_MIN;
	}

	/* Pad incomplete final billion-based digit with zeros */
	if (j) {
		for (; j < 9; j++) {
			x[k] *= 10;
		}
		k++;
		j = 0;
	}

	/* Initialize working variables for precision conversion */
	a = 0;      /* Start of valid data in circular buffer */
	z = k;      /* End of valid data in circular buffer */
	e2 = 0;     /* Binary exponent for final result */
	rp = lrp;   /* Working copy of radix position */

	/* Fast path for small to mid-size integers (even with exponent notation) */
	if (lnz < 9 && lnz <= rp && rp < 18) {
		if (rp == 9) {
			return sign * (long double)x[0];
		}
		if (rp < 9) {
			return sign * (long double)x[0] / p10s[8 - rp];
		}
		int bitlim = bits - 3 * (int)(rp - 9);
		if (bitlim > 30 || x[0] >> bitlim == 0) {
			return sign * (long double)x[0] * p10s[rp - 10];
		}
	}

	/* Remove trailing zero billion-based digits */
	for (; !x[z - 1]; z--) {
		/* Keep removing until we find a non-zero digit */
	}

	/*
	 * Align radix point to billion-based digit boundary
	 * The billion-based representation works with groups of 9 decimal digits,
	 * so we need to adjust if the radix point isn't aligned to a 9-digit boundary.
	 */
	if (rp % 9) {
		int rpm9 = rp >= 0 ? rp % 9 : rp % 9 + 9;
		int p10 = p10s[8 - rpm9];
		uint32_t carry = 0;

		/* Divide all digits by the appropriate power of 10 */
		for (k = a; k != z; k++) {
			uint32_t tmp = x[k] % p10;
			x[k] = x[k] / p10 + carry;
			carry = 1000000000 / p10 * tmp;

			/* If the leading digit becomes zero, advance the start pointer */
			if (k == a && !x[k]) {
				a = (a + 1) & MASK;
				rp -= 9;
			}
		}

		/* If there's a carry, add a new digit */
		if (carry) {
			x[z++] = carry;
		}
		rp += 9 - rpm9;
	}

	/*
	 * Upscale until desired number of bits are left of radix point
	 *
	 * We need exactly LD_B1B_DIG billion-based digits to the left of the radix point
	 * to have enough precision for the target floating-point format. If we don't have
	 * enough, we multiply by 2^29 to shift digits leftward.
	 *
	 * Why 2^29? This is a key insight of the algorithm:
	 * - 2^29 = 536,870,912 ≈ 0.537 × 10^9 (close to one billion)
	 * - It's the largest power of 2 less than 10^9 = 1,000,000,000
	 * - Being a power of 2, it's very efficient (just a 29-bit left shift)
	 * - 2^29 × 999,999,999 < 2^32, so no overflow in intermediate calculations
	 * - We can track the scaling exactly in binary exponent (e2 -= 29)
	 *
	 * This bridges between decimal input and binary floating-point output:
	 * we get substantial decimal scaling (~0.5 billion factor) while maintaining
	 * exact binary arithmetic throughout the conversion process.
	 *
	 * The condition checks:
	 * 1. rp < 9*LD_B1B_DIG: Not enough digits to the left of radix point
	 * 2. rp == 9*LD_B1B_DIG && x[a]<th[0]: Exactly enough digits, but the leading
	 *    digit is too small (less than the threshold for the target precision)
	 */
	while (rp < 9*LD_B1B_DIG || (rp == 9*LD_B1B_DIG && x[a] < th[0])) {
		uint32_t carry = 0;
		e2 -= 29;  /* Adjust binary exponent for 2^29 scaling */

		/* Multiply all digits by 2^29, working backwards through the array */
		for (k = (z-1) & MASK; ; k = (k-1) & MASK) {
			uint64_t tmp = ((uint64_t)x[k] << 29) + carry;
			if (tmp > 1000000000) {
				carry = tmp / 1000000000;
				x[k] = tmp % 1000000000;
			} else {
				carry = 0;
				x[k] = tmp;
			}
			/* Remove trailing zeros that might have been created */
			if (k == ((z-1) & MASK) && k != a && !x[k]) {
				z = k;
			}
			if (k == a) break;  /* Processed all digits */
		}

		/* If there's a carry, we need to add a new leading digit */
		if (carry) {
			rp += 9;  /* We now have 9 more digits to the left of radix point */
			a = (a-1) & MASK;  /* Move start pointer back to make room */
			if (a == z) {
				/* Buffer overflow - merge the last two digits */
				z = (z-1) & MASK;
				x[(z-1) & MASK] |= x[z];  /* Set a bit to indicate lost precision */
			}
			x[a] = carry;  /* Store the new leading digit */
		}
	}

	/*
	 * Downscale until exactly the right number of bits are left of radix point
	 *
	 * After upscaling, we might have too many significant digits. We need to
	 * scale down by dividing by powers of 2 until we have exactly the right
	 * amount of precision for the target floating-point format.
	 *
	 * The algorithm compares our current leading digits with the threshold
	 * values (th[]) to determine if we have the right magnitude.
	 */
	for (;;) {
		uint32_t carry = 0;
		int sh = 1;  /* Shift amount (power of 2 to divide by) */

		/* Check if we have the right magnitude by comparing with thresholds */
		for (i = 0; i < LD_B1B_DIG; i++) {
			k = (a + i) & MASK;
			if (k == z || x[k] < th[i]) {
				i = LD_B1B_DIG;  /* Signal: magnitude is too small */
				break;
			}
			if (x[(a + i) & MASK] > th[i]) break;  /* Magnitude is too large */
		}

		/* If we have exactly the right magnitude and position, we're done */
		if (i == LD_B1B_DIG && rp == 9 * LD_B1B_DIG) break;

		/* Choose shift amount - larger shifts for very large numbers */
		/* FIXME: find a way to compute optimal sh */
		if (rp > 9 + 9 * LD_B1B_DIG) sh = 9;

		e2 += sh;  /* Adjust binary exponent */

		/* Divide all digits by 2^sh */
		for (k = a; k != z; k = (k + 1) & MASK) {
			uint32_t tmp = x[k] & ((1 << sh) - 1);  /* Bits that will be shifted out */
			x[k] = (x[k] >> sh) + carry;
			carry = (1000000000 >> sh) * tmp;  /* Carry for next digit */

			/* If leading digit becomes zero, advance start pointer */
			if (k == a && !x[k]) {
				a = (a + 1) & MASK;
				i--;
				rp -= 9;
			}
		}

		/* Handle any remaining carry */
		if (carry) {
			if (((z + 1) & MASK) != a) {
				x[z] = carry;
				z = (z + 1) & MASK;
			} else {
				x[(z - 1) & MASK] |= 1;  /* Set sticky bit for lost precision */
			}
		}
	}

	/*
	 * Assemble the final floating-point value
	 *
	 * Convert the billion-based digits back to a floating-point number.
	 * We take exactly LD_B1B_DIG billion-based digits and combine them
	 * into a single long double value.
	 */
	for (y = i = 0; i < LD_B1B_DIG; i++) {
		/* If we run out of digits, pad with zeros */
		if (((a + i) & MASK) == z) {
			x[(z = ((z + 1) & MASK)) - 1] = 0;
		}
		/* Accumulate: y = y * 10^9 + next_digit */
		y = 1000000000.0L * y + x[(a + i) & MASK];
	}

	y *= sign;  /* Apply the sign */

	/*
	 * Handle denormal numbers
	 *
	 * If the exponent is too small, we need to reduce precision to avoid
	 * underflow. Denormal numbers have reduced precision near zero.
	 */
	if (bits > LDBL_MANT_DIG + e2 - emin) {
		bits = LDBL_MANT_DIG + e2 - emin;
		if (bits < 0) bits = 0;
		denormal = 1;
	}

	/*
	 * Implement correct rounding
	 *
	 * If we need less precision than long double provides, we use a bias
	 * technique to force proper rounding. The bias is added and then
	 * subtracted to round the number to the target precision.
	 */
	if (bits < LDBL_MANT_DIG) {
		bias = copysignl(scalbn(1, 2*LDBL_MANT_DIG - bits - 1), y);
		frac = fmodl(y, scalbn(1, LDBL_MANT_DIG - bits));
		y -= frac;
		y += bias;
	}

	/*
	 * Process remaining digits to affect rounding
	 *
	 * We've assembled the main significant digits, but there might be more
	 * digits that should influence rounding. We look at the next billion-based
	 * digit to determine how to round.
	 *
	 * The rounding logic:
	 * - If next digit < 500000000: round down (add 0.25 to bias toward down)
	 * - If next digit > 500000000: round up (add 0.75 to bias toward up)
	 * - If next digit = 500000000: round to even (add 0.5, or 0.75 if more digits)
	 */
	if (((a + i) & MASK) != z) {
		uint32_t t = x[(a + i) & MASK];  /* Next billion-based digit */

		if (t < 500000000 && (t || ((a + i + 1) & MASK) != z)) {
			/* Less than half, or exactly half with more digits following */
			frac += 0.25 * sign;
		} else if (t > 500000000) {
			/* More than half - round up */
			frac += 0.75 * sign;
		} else if (t == 500000000) {
			/* Exactly half */
			if (((a + i + 1) & MASK) == z) {
				/* Exactly half with no more digits - round to even */
				frac += 0.5 * sign;
			} else {
				/* Exactly half with more digits - round up */
				frac += 0.75 * sign;
			}
		}

		/* Additional rounding adjustment for very low precision */
		if (LDBL_MANT_DIG - bits >= 2 && !fmodl(frac, 1)) {
			frac++;
		}
	}

	/* Apply the fractional rounding adjustment */
	y += frac;
	y -= bias;

	/*
	 * Handle overflow and final range checking
	 *
	 * Check if the result is too large for the target format, and handle
	 * the transition between normal and denormal numbers.
	 */
	if ((e2 + LDBL_MANT_DIG & INT_MAX) > emax - 5) {
		if (fabsl(y) >= 2 / LDBL_EPSILON) {
			/* Number is too large - might need to adjust */
			if (denormal && bits == LDBL_MANT_DIG + e2 - emin) {
				denormal = 0;  /* Actually not denormal after all */
			}
			y *= 0.5;  /* Scale down */
			e2++;       /* Adjust exponent */
		}

		/* Check for overflow or denormal underflow */
		if (e2 + LDBL_MANT_DIG > emax || (denormal && frac)) {
			errno = ERANGE;
		}
	}

	/* Combine mantissa and exponent to get final result */
	return scalbnl(y, e2);
}

/*
 * Parse a hexadecimal floating-point number (e.g., "0x1.23p45")
 *
 * Hexadecimal floating-point format: 0x[hex_digits].[hex_digits]p[exponent]
 * The exponent is a power of 2 (not 10), and the mantissa is in base 16.
 *
 * This format is useful because it can represent any binary floating-point
 * number exactly, without rounding errors from decimal conversion.
 *
 * Parameters:
 *   f: Input stream
 *   bits: Target precision in bits
 *   emin: Minimum exponent for target type
 *   sign: Sign of the number (+1 or -1)
 *   pok: If true, partial parsing is OK (for scanf-style parsing)
 *
 * Returns:
 *   The parsed floating-point value
 */
static long double hexfloat(FILE *f, int bits, int emin, int sign, int pok)
{
	uint32_t x = 0;           /* Integer part of mantissa */
	long double y = 0;        /* Fractional part of mantissa */
	long double scale = 1;    /* Scale factor for fractional digits */
	long double bias = 0;     /* Rounding bias */
	int gottail = 0;          /* Found digits beyond precision limit */
	int gotrad = 0;           /* Found decimal point */
	int gotdig = 0;           /* Found at least one hex digit */
	long long rp = 0;         /* Radix point position */
	long long dc = 0;         /* Digit count */
	long long e2 = 0;         /* Binary exponent from 'p' notation */
	int d;                    /* Current hex digit value */
	int c;                    /* Current character */

	c = shgetc(f);

	/* Skip leading zeros - they don't affect the value */
	for (; c == '0'; c = shgetc(f)) {
		gotdig = 1;
	}

	/* Handle leading decimal point */
	if (c == '.') {
		gotrad = 1;
		c = shgetc(f);
		/* Count zeros after decimal point before first significant digit */
		for (rp = 0; c == '0'; c = shgetc(f), rp--) {
			gotdig = 1;
		}
	}

	/* Main hex digit parsing loop */
	for (; (c - '0' < 10U) || ((c|32) - 'a' < 6U) || c == '.'; c = shgetc(f)) {
		if (c == '.') {
			/* Handle decimal point */
			if (gotrad) break;  /* Second decimal point - stop */
			rp = dc;            /* Record position of decimal point */
			gotrad = 1;
		} else {
			/* Process hex digit */
			gotdig = 1;
			if (c > '9') {
				d = (c|32) + 10 - 'a';  /* Convert 'a'-'f' to 10-15 */
			} else {
				d = c - '0';            /* Convert '0'-'9' to 0-9 */
			}

			if (dc < 8) {
				/* First 8 hex digits go into integer part */
				x = x * 16 + d;
			} else if (dc < LDBL_MANT_DIG/4 + 1) {
				/* Additional digits go into fractional part */
				y += d * (scale /= 16);
			} else if (d && !gottail) {
				/* Beyond precision - just set a flag for rounding */
				y += 0.5 * scale;
				gottail = 1;
			}
			dc++;
		}
	}

	/* Must have found at least one hex digit */
	if (!gotdig) {
		shunget(f);
		if (pok) {
			shunget(f);
			if (gotrad) shunget(f);
		} else {
			shlim(f, 0);
		}
		return sign * 0.0;
	}

	/* If no decimal point found, it's at the end */
	if (!gotrad) rp = dc;

	/* Pad integer part to 8 hex digits (32 bits) */
	while (dc < 8) {
		x *= 16;
		dc++;
	}

	/* Parse optional binary exponent (e.g., "p123" or "P-45") */
	if ((c|32) == 'p') {
		e2 = scan_exponent(f, pok);
		if (e2 == LLONG_MIN) {
			if (pok) {
				shunget(f);
			} else {
				shlim(f, 0);
				return 0;
			}
			e2 = 0;
		}
	} else {
		shunget(f);
	}

	/* Adjust exponent: 4 bits per hex digit, minus 32 for normalization */
	e2 += 4 * rp - 32;

	/* Handle zero case */
	if (!x) return sign * 0.0;

	/* Check for overflow/underflow */
	if (e2 > -emin) {
		errno = ERANGE;
		return sign * LDBL_MAX * LDBL_MAX;
	}
	if (e2 < emin - 2*LDBL_MANT_DIG) {
		errno = ERANGE;
		return sign * LDBL_MIN * LDBL_MIN;
	}

	/*
	 * Normalize the mantissa
	 *
	 * We need the most significant bit of x to be in the top bit position
	 * (0x80000000). If it's not, we shift left and adjust the exponent.
	 */
	while (x < 0x80000000) {
		if (y >= 0.5) {
			/* Fractional part >= 0.5, so carry into integer part */
			x += x + 1;  /* x = x*2 + 1 */
			y += y - 1;  /* y = y*2 - 1 */
		} else {
			/* No carry needed */
			x += x;      /* x = x*2 */
			y += y;      /* y = y*2 */
		}
		e2--;  /* Adjust exponent for the left shift */
	}

	/* Handle denormal numbers */
	if (bits > 32 + e2 - emin) {
		bits = 32 + e2 - emin;
		if (bits < 0) bits = 0;
	}

	/* Set up rounding bias for target precision */
	if (bits < LDBL_MANT_DIG) {
		bias = copysignl(scalbn(1, 32 + LDBL_MANT_DIG - bits - 1), sign);
	}

	/* Round to odd if we're truncating and have fractional bits */
	if (bits < 32 && y && !(x & 1)) {
		x++;
		y = 0;
	}

	/* Apply bias, combine parts, then remove bias for proper rounding */
	y = bias + sign * (long double)x + sign * y;
	y -= bias;

	/* Check for underflow to zero */
	if (!y) errno = ERANGE;

	/* Combine mantissa and exponent */
	return scalbnl(y, e2);
}

/*
 * Main floating-point parsing function
 *
 * This is the entry point for floating-point parsing. It handles:
 * - Sign parsing
 * - Special values (infinity, NaN)
 * - Dispatch to decimal or hexadecimal parsing
 *
 * Parameters:
 *   f: Input stream
 *   prec: Precision selector (0=float, 1=double, 2=long double)
 *   pok: If true, partial parsing is OK (for scanf-style parsing)
 *
 * Returns:
 *   The parsed floating-point value
 */
long double __floatscan(FILE *f, int prec, int pok)
{
	int sign = 1;     /* Sign of the number (+1 or -1) */
	size_t i;         /* Loop counter */
	int bits;         /* Target precision in bits */
	int emin;         /* Minimum exponent for target type */
	int c;            /* Current character */

	/* Set precision parameters based on target type */
	switch (prec) {
	case 0:  /* float */
		bits = FLT_MANT_DIG;
		emin = FLT_MIN_EXP - bits;
		break;
	case 1:  /* double */
		bits = DBL_MANT_DIG;
		emin = DBL_MIN_EXP - bits;
		break;
	case 2:  /* long double */
		bits = LDBL_MANT_DIG;
		emin = LDBL_MIN_EXP - bits;
		break;
	default:
		return 0;
	}

	/* Skip leading whitespace */
	while (isspace((c = shgetc(f)))) {
		/* Keep skipping */
	}

	/* Parse optional sign */
	if (c == '+' || c == '-') {
		sign -= 2 * (c == '-');  /* +1 for '+', -1 for '-' */
		c = shgetc(f);
	}

	/*
	 * Check for "infinity" or "inf"
	 *
	 * We match against "infinity" case-insensitively. We accept:
	 * - "inf" (3 characters)
	 * - "infinity" (8 characters)
	 * - Partial matches if pok is true (for scanf)
	 */
	for (i = 0; i < 8 && (c|32) == "infinity"[i]; i++) {
		if (i < 7) c = shgetc(f);
	}
	if (i == 3 || i == 8 || (i > 3 && pok)) {
		/* Found "inf" or "infinity" (or partial match with pok) */
		if (i != 8) {
			shunget(f);  /* Put back the non-matching character */
			if (pok) {
				/* For partial parsing, put back extra characters */
				for (; i > 3; i--) shunget(f);
			}
		}
		return sign * INFINITY;
	}

	/*
	 * Check for "nan"
	 *
	 * NaN can optionally be followed by a parenthesized sequence
	 * like "nan(123)" for implementation-specific NaN payloads.
	 */
	if (!i) {
		/* Only check for "nan" if we didn't match any of "infinity" */
		for (i = 0; i < 3 && (c|32) == "nan"[i]; i++) {
			if (i < 2) c = shgetc(f);
		}
	}
	if (i == 3) {
		/* Found "nan" - check for optional payload */
		if (shgetc(f) != '(') {
			shunget(f);
			return NAN;
		}

		/* Parse NaN payload: alphanumeric characters and underscore */
		for (i = 1; ; i++) {
			c = shgetc(f);
			if ((c - '0' < 10U) || (c - 'A' < 26U) || (c - 'a' < 26U) || c == '_') {
				continue;  /* Valid payload character */
			}
			if (c == ')') return NAN;  /* End of payload */

			/* Invalid character in payload */
			shunget(f);
			if (!pok) {
				errno = EINVAL;
				shlim(f, 0);
				return 0;
			}
			/* For partial parsing, back up and return NaN */
			while (i--) shunget(f);
			return NAN;
		}
		return NAN;
	}

	/* If we partially matched something but not completely, it's an error */
	if (i) {
		shunget(f);
		errno = EINVAL;
		shlim(f, 0);
		return 0;
	}

	/*
	 * Check for hexadecimal format (0x...)
	 *
	 * If we see "0x" or "0X", dispatch to hexadecimal parser.
	 * Otherwise, parse as decimal.
	 */
	if (c == '0') {
		c = shgetc(f);
		if ((c|32) == 'x') {
			return hexfloat(f, bits, emin, sign, pok);
		}
		shunget(f);
		c = '0';  /* Treat the '0' as start of decimal number */
	}

	/* Parse as decimal floating-point number */
	return decfloat(f, c, bits, emin, sign, pok);
}
