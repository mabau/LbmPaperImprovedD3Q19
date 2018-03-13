"""

Module Overview
~~~~~~~~~~~~~~~

This module provides functions to

- generate moments according to certain patterns
- compute moments of discrete probability distribution functions
- create transformation rules into moment space


Moment Representation
~~~~~~~~~~~~~~~~~~~~~

Moments can be represented in two ways:

- by an index :math:`i,j`: defined as :math:`m_{ij} := \sum_{\mathbf{d} \in stencil} <\mathbf{d}, \mathbf{x}> f_i`
- or by a polynomial in the variables x,y and z. For example the polynomial :math:`x^2 y^1 z^3 + x + 1` is
  describing the linear combination of moments: :math:`m_{213} + m_{100} + m_{000}`

The polynomial description is more powerful, since one polynomial can express a linear combination of single moments.
All moment polynomials have to use ``MOMENT_SYMBOLS`` (which is a module variable) as degrees of freedom.

Example ::

    from lbmpy.moments import MOMENT_SYMBOLS
    x, y, z = MOMENT_SYMBOLS
    secondOrderMoment = x*y + y*z

"""
import itertools
import math
from collections import Counter
import sympy as sp


__author__ = "Martin Bauer"
__copyright__ = "Copyright 2018, Martin Bauer"
__license__ = "GPL"
__version__ = "3"
__email__ = "martin.bauer@fau.de"


MOMENT_SYMBOLS = sp.symbols("x y z")


def removeHigherOrderTerms(term, order=3, symbols=None):
    """
    Removes all terms that that contain more than 'order' factors of given 'symbols'

    Example:
        >>> x, y = sp.symbols("x y")
        >>> term = x**2 * y + y**2 * x + y**3 + x + y ** 2
        >>> removeHigherOrderTerms(term, order=2, symbols=[x, y])
        x + y**2
    """
    from sympy.core.power import Pow
    from sympy.core.add import Add, Mul

    result = 0
    term = term.expand()

    if not symbols:
        symbols = sp.symbols(" ".join(["u_%d" % (i,) for i in range(3)]))
        symbols += sp.symbols(" ".join(["u_%d" % (i,) for i in range(3)]), real=True)

    def velocityFactorsInProduct(product):
        uFactorCount = 0
        if type(product) is Mul:
            for factor in product.args:
                if type(factor) == Pow:
                    if factor.args[0] in symbols:
                        uFactorCount += factor.args[1]
                if factor in symbols:
                    uFactorCount += 1
        elif type(product) is Pow:
            if product.args[0] in symbols:
                uFactorCount += product.args[1]
        return uFactorCount

    if type(term) == Mul or type(term) == Pow:
        if velocityFactorsInProduct(term) <= order:
            return term
        else:
            return sp.Rational(0, 1)

    if type(term) != Add:
        return term

    for sumTerm in term.args:
        if velocityFactorsInProduct(sumTerm) <= order:
            result += sumTerm
    return result


def __uniqueList(seq):
    seen = {}
    result = []
    for item in seq:
        if item in seen:
            continue
        seen[item] = 1
        result.append(item)
    return result


def __uniquePermutations(elements):
    if len(elements) == 1:
        yield (elements[0],)
    else:
        unique_elements = set(elements)
        for first_element in unique_elements:
            remaining_elements = list(elements)
            remaining_elements.remove(first_element)
            for sub_permutation in __uniquePermutations(remaining_elements):
                yield (first_element,) + sub_permutation


# ------------------------------ Discrete (Exponent Tuples) ------------------------------------------------------------


def momentMultiplicity(exponentTuple):
    """
    Returns number of permutations of the given moment tuple

    Example:
    >>> momentMultiplicity((2,0,0))
    3
    >>> list(momentPermutations((2,0,0)))
    [(0, 0, 2), (0, 2, 0), (2, 0, 0)]
    """
    c = Counter(exponentTuple)
    result = math.factorial(len(exponentTuple))
    for d in c.values():
        result //= math.factorial(d)
    return result


def momentPermutations(exponentTuple):
    """Returns all (unique) permutations of the given tuple"""
    return __uniquePermutations(exponentTuple)


def momentsUpToComponentOrder(order, dim=3):
    """All tuples of length 'dim' where each entry is smaller or equal to 'order' """
    return tuple(itertools.product(*[range(order + 1)] * dim))


def extendMomentsWithPermutations(exponentTuples):
    """Returns all permutations of the given exponent tuples"""
    allMoments = []
    for i in exponentTuples:
        allMoments += list(momentPermutations(i))
    return __uniqueList(allMoments)


# ------------------------------ Representation Conversions ------------------------------------------------------------


def exponentToPolynomialRepresentation(exponentTuple):
    """
    Converts an exponent tuple to corresponding polynomial representation

    Example:
        >>> exponentToPolynomialRepresentation( (2,1,3) )
        x**2*y*z**3
    """
    poly = 1
    for sym, tupleEntry in zip(MOMENT_SYMBOLS[:len(exponentTuple)], exponentTuple):
        poly *= sym ** tupleEntry
    return poly


def exponentsToPolynomialRepresentations(sequenceOfExponentTuples):
    """Applies :func:`exponentToPolynomialRepresentation` to given sequence"""
    return tuple([exponentToPolynomialRepresentation(t) for t in sequenceOfExponentTuples])


def polynomialToExponentRepresentation(polynomial, dim=3):
    """
    Converts a linear combination of moments in polynomial representation into exponent representation

    :returns list of tuples where the first element is the coefficient and the second element is the exponent tuple

    Example:
        >>> x , y, z = MOMENT_SYMBOLS
        >>> set(polynomialToExponentRepresentation(1 + (42 * x**2 * y**2 * z) )) == {(42, (2, 2, 1)), (1, (0, 0, 0))}
        True
    """
    assert dim <= 3
    x, y, z = MOMENT_SYMBOLS
    polynomial = polynomial.expand()
    coeffExpTupleRepresentation = []

    summands = [polynomial] if polynomial.func != sp.Add else polynomial.args
    for expr in summands:
        if len(expr.atoms(sp.Symbol) - set(MOMENT_SYMBOLS)) > 0:
            raise ValueError("Invalid moment polynomial: " + str(expr))
        c, x_exp, y_exp, z_exp = sp.Wild('c'), sp.Wild('xexp'), sp.Wild('yexp'), sp.Wild('zc')
        matchRes = expr.match(c * x ** x_exp * y ** y_exp * z ** z_exp)
        assert matchRes[x_exp].is_integer and matchRes[y_exp].is_integer and matchRes[z_exp].is_integer
        expTuple = (int(matchRes[x_exp]), int(matchRes[y_exp]), int(matchRes[z_exp]),)
        if dim < 3:
            for i in range(dim, 3):
                assert expTuple[i] == 0, "Used symbols in polynomial are not representable in that dimension"
            expTuple = expTuple[:dim]
        coeffExpTupleRepresentation.append((matchRes[c], expTuple))
    return coeffExpTupleRepresentation


# -------------------- Common Function working with exponent tuples and polynomial moments -----------------------------


def discreteMoment(function, moment, stencil):
    """
    Computes discrete moment of given distribution function

    .. math ::
        \sum_{d \in stencil} p(d) f_i

    where :math:`p(d)` is the moment polynomial where :math:`x, y, z` have been replaced with the components of the
    stencil direction, and :math:`f_i` is the i'th entry in the passed function sequence

    :param function: list of distribution functions for each direction
    :param moment: can either be a exponent tuple, or a sympy polynomial expression
    :param stencil: sequence of directions
    """
    assert len(stencil) == len(function)
    res = 0
    for factor, e in zip(function, stencil):
        if type(moment) is tuple:
            for vel, exponent in zip(e, moment):
                factor *= vel ** exponent
            res += factor
        else:
            weight = moment
            for variable, e_i in zip(MOMENT_SYMBOLS, e):
                weight = weight.subs(variable, e_i)
            res += weight * factor

    return res


def momentMatrix(moments, stencil):
    """
    Returns transformation matrix to moment space

    each row corresponds to a moment, each column to a direction of the stencil
    The entry i,j is the i'th moment polynomial evaluated at direction j
    """

    if type(moments[0]) is tuple:
        def generator(row, column):
            result = sp.Rational(1, 1)
            for exponent, stencilEntry in zip(moments[row], stencil[column]):
                result *= int(stencilEntry ** exponent)
            return result
    else:
        def generator(row, column):
            evaluated = moments[row]
            for var, stencilEntry in zip(MOMENT_SYMBOLS, stencil[column]):
                evaluated = evaluated.subs(var, stencilEntry)
            return evaluated

    return sp.Matrix(len(moments), len(stencil), generator)


