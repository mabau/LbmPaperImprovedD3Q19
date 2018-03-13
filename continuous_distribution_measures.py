import sympy as sp
from moments import polynomialToExponentRepresentation


__author__ = "Martin Bauer"
__copyright__ = "Copyright 2018, Martin Bauer"
__license__ = "GPL"
__version__ = "3"
__email__ = "martin.bauer@fau.de"


def completeTheSquare(expr, symbolToComplete, newVariable):
    """
    Transforms second order polynomial into only squared part i.e.
        a*symbolToComplete**2 + b*symbolToComplete + c
          is transformed into
        newVariable**2 + d

    returns replacedExpr, "a tuple to to replace newVariable such that old expr comes out again"

    if given expr is not a second order polynomial:
        return expr, None
    """
    p = sp.Poly(expr, symbolToComplete)
    coeffs = p.all_coeffs()
    if len(coeffs) != 3:
        return expr, None
    a, b, _ = coeffs
    expr = expr.subs(symbolToComplete, newVariable - b / (2 * a))
    return sp.simplify(expr), (newVariable, symbolToComplete + b / (2 * a))


def makeExponentialFuncArgumentSquares(expr, variablesToCompleteSquares):
    """Completes squares in arguments of exponential which makes them simpler to integrate
    Very useful for integrating Maxwell-Boltzmann and its moment generating function"""
    expr = sp.simplify(expr)
    dim = len(variablesToCompleteSquares)
    dummies = [sp.Dummy() for i in range(dim)]

    def visit(term):
        if term.func == sp.exp:
            expArg = term.args[0]
            for i in range(dim):
                expArg, substitution = completeTheSquare(expArg, variablesToCompleteSquares[i], dummies[i])
            return sp.exp(sp.expand(expArg))
        else:
            paramList = [visit(a) for a in term.args]
            if not paramList:
                return term
            else:
                return term.func(*paramList)

    result = visit(expr)
    for i in range(dim):
        result = result.subs(dummies[i], variablesToCompleteSquares[i])
    return result


def momentGeneratingFunction(function, symbols, symbolsInResult):
    """
    Computes the moment generating function of a probability distribution. It is defined as:

    .. math ::
        F[f(\mathbf{x})](\mathbf{t}) = \int e^{<\mathbf{x}, \mathbf{t}>} f(x)\; dx

    :param function: sympy expression
    :param symbols: a sequence of symbols forming the vector x
    :param symbolsInResult: a sequence forming the vector t
    :return: transformation result F: an expression that depends now on symbolsInResult
             (symbols have been integrated out)

    .. note::
         This function uses sympys symbolic integration mechanism, which may not work or take a large
         amount of time for some functions.
         Therefore this routine does some transformations/simplifications on the function first, which are
         taylored to expressions of the form exp(polynomial) i.e. Maxwellian distributions, so that these kinds
         of functions can be integrated quickly.

    """
    assert len(symbols) == len(symbolsInResult)
    for t_i, v_i in zip(symbolsInResult, symbols):
        function *= sp.exp(t_i * v_i)

    # This is a custom transformation that speeds up the integrating process
    # of a MaxwellBoltzmann distribution
    # without this transformation the symbolic integration is sometimes not possible (e.g. in 2D without assumptions)
    # or is really slow
    # other functions should not be affected by this transformation
    # Without this transformation the following assumptions are required for the u and v variables of Maxwell Boltzmann
    #  2D: real=True ( without assumption it will not work)
    #  3D: no assumption ( with assumptions it will not work )
    function = makeExponentialFuncArgumentSquares(function, symbols)
    function = function.collect(symbols)

    bounds = [(s_i, -sp.oo, sp.oo) for s_i in symbols]
    result = sp.integrate(function, *bounds)

    return sp.simplify(result)


def multiDifferentiation(generatingFunction, index, symbols):
    """
    Computes moment from moment-generating function or cumulant from cumulant-generating function,
    by differentiating the generating function, as specified by index and evaluating the derivative at symbols=0

    :param generatingFunction: function with is differentiated
    :param index: the i'th index specifies how often to differentiate w.r.t. to symbols[i]
    :param symbols: symbol to differentiate
    """
    assert len(index) == len(symbols), "Length of index and length of symbols has to match"

    diffArgs = []
    for order, t_i in zip(index, symbols):
        for i in range(order):
            diffArgs.append(t_i)

    if len(diffArgs) > 0:
        r = sp.diff(generatingFunction, *diffArgs)
    else:
        r = generatingFunction

    for t_i in symbols:
        r = r.subs(t_i, 0)

    return r


def __continuousMomentOrCumulant(function, moment, symbols, generatingFunction):
    if type(moment) is tuple and not symbols:
        symbols = sp.symbols("xvar yvar zvar")

    dim = len(moment) if type(moment) is tuple else len(symbols)

    # not using sp.Dummy here - since it prohibits caching
    t = tuple([sp.Symbol("tmpvar_%d" % i, ) for i in range(dim)])
    symbols = symbols[:dim]
    genFunc = generatingFunction(function, symbols, t)

    if type(moment) is tuple:
        return multiDifferentiation(genFunc, moment, t)
    else:
        assert symbols is not None, "When passing a polynomial as moment, also the moment symbols have to be passed"
        moment = sp.sympify(moment)

        result = 0
        for coefficient, exponents in polynomialToExponentRepresentation(moment, dim=dim):
            result += coefficient * multiDifferentiation(genFunc, exponents, t)

        return result


def continuousMoment(function, moment, symbols=None):
    """
    Computes moment of given function

    :param function: function to compute moments of
    :param moment: tuple or polynomial describing the moment
    :param symbols: if moment is given as polynomial, pass the moment symbols, i.e. the dof of the polynomial
    """
    return __continuousMomentOrCumulant(function, moment, symbols, momentGeneratingFunction)


