# -*- coding: utf-8 -*-
"""
This module contains the continuous Maxwell-Boltzmann equilibrium and its discrete polynomial approximation, often
used to formulate lattice-Boltzmann methods for hydrodynamics.
Additionally functions are provided to compute moments of these distributions.
"""

import sympy as sp
from sympy import Rational as R
from moments import removeHigherOrderTerms, discreteMoment, MOMENT_SYMBOLS
from continuous_distribution_measures import continuousMoment


__author__ = "Martin Bauer"
__copyright__ = "Copyright 2018, Martin Bauer"
__license__ = "GPL"
__version__ = "3"
__email__ = "martin.bauer@fau.de"


def getWeights(stencil):
    Q = len(stencil)

    def weightForDirection(direction):
        absSum = sum([abs(d) for d in direction])
        return getWeights.weights[Q][absSum]
    return [weightForDirection(d) for d in stencil]
getWeights.weights = {
    9: {
        0: R(4, 9),
        1: R(1, 9),
        2: R(1, 36),
    },
    15: {
        0: R(2, 9),
        1: R(1, 9),
        3: R(1, 72),
    },
    19: {
        0: R(1, 3),
        1: R(1, 18),
        2: R(1, 36),
    },
    27: {
        0: R(8, 27),
        1: R(2, 27),
        2: R(1, 54),
        3: R(1, 216),
    }
}


def discreteMaxwellianEquilibrium(stencil, rho=sp.Symbol("rho"), u=tuple(sp.symbols("u_0 u_1 u_2")),
                                  order=2, compressible=True):
    """
    Returns the common discrete LBM equilibrium as a list of sympy expressions

    :param stencil: tuple of directions
    :param rho: sympy symbol for the density
    :param u: symbols for macroscopic velocity, only the first `dim` entries are used
    :param order: highest order of velocity terms (for hydrodynamics order 2 is sufficient)
    :param compressible: compressibility
    """
    weights = getWeights(stencil)
    assert len(stencil) == len(weights)

    dim = len(stencil[0])
    u = u[:dim]
    c_s_sq = sp.Rational(1,3)
    rhoOutside = rho if compressible else sp.Rational(1, 1)
    rhoInside = rho if not compressible else sp.Rational(1, 1)

    res = []
    for w_q, e_q in zip(weights, stencil):
        eTimesU = 0
        for c_q_alpha, u_alpha in zip(e_q, u):
            eTimesU += c_q_alpha * u_alpha

        fq = rhoInside + eTimesU / c_s_sq

        if order <= 1:
            res.append(fq * rhoOutside * w_q)
            continue

        uTimesU = 0
        for u_alpha in u:
            uTimesU += u_alpha * u_alpha
        fq += sp.Rational(1, 2) / c_s_sq**2 * eTimesU ** 2 - sp.Rational(1, 2) / c_s_sq * uTimesU

        if order <= 2:
            res.append(fq * rhoOutside * w_q)
            continue

        fq += sp.Rational(1, 6) / c_s_sq**3 * eTimesU**3 - sp.Rational(1, 2) / c_s_sq**2 * uTimesU * eTimesU

        res.append(sp.expand(fq * rhoOutside * w_q))

    return tuple(res)


def continuousMaxwellianEquilibrium(dim=3, rho=sp.Symbol("rho"),
                                    u=tuple(sp.symbols("u_0 u_1 u_2")),
                                    ζ=tuple(sp.symbols("ζ_0 ζ_1 ζ_2")),
                                    c_s_sq=sp.Symbol("c_s") ** 2):
    """
    Returns sympy expression of Maxwell Boltzmann distribution

    :param dim: number of space dimensions
    :param rho: sympy symbol for the density
    :param u: symbols for macroscopic velocity (expected value for velocity)
    :param ζ: symbols for particle velocity
    :param c_s_sq: symbol for speed of sound squared, defaults to symbol c_s**2
    """
    u = u[:dim]
    v = ζ[:dim]

    velTerm = sum([(v_i - u_i) ** 2 for v_i, u_i in zip(v, u)])
    return rho / (2 * sp.pi * c_s_sq) ** (sp.Rational(dim, 2)) * sp.exp(- velTerm / (2 * c_s_sq))


# -------------------------------- Equilibrium moments/cumulants  ------------------------------------------------------


def getMomentsOfContinuousMaxwellianEquilibrium(moments, dim, rho=sp.Symbol("rho"), u=tuple(sp.symbols("u_0 u_1 u_2")),
                                                c_s_sq=sp.Symbol("c_s") ** 2, order=None):
    """
    Computes moments of the continuous Maxwell Boltzmann equilibrium distribution

    :param moments: moments to compute, either in polynomial or exponent-tuple form
    :param dim: dimension (2 or 3)
    :param rho: symbol or value for the density
    :param u: symbols or values for the macroscopic velocity
    :param c_s_sq: symbol for speed of sound squared, defaults to symbol c_s**2
    :param order: if this parameter is not None, terms that have a higher polynomial order in the macroscopic velocity
                  are removed

    >>> getMomentsOfContinuousMaxwellianEquilibrium( ( (0,0,0), (1,0,0), (0,1,0), (0,0,1), (2,0,0) ), dim=3 )
    [rho, rho*u_0, rho*u_1, rho*u_2, rho*(c_s**2 + u_0**2)]
    """

    # trick to speed up sympy integration (otherwise it takes multiple minutes, or aborts):
    # use a positive, real symbol to represent c_s_sq -> then replace this symbol afterwards with the real c_s_sq
    c_s_sq_helper = sp.Symbol("csqHelper", positive=True, real=True)
    mb = continuousMaxwellianEquilibrium(dim, rho, u, MOMENT_SYMBOLS[:dim], c_s_sq_helper)
    result = [continuousMoment(mb, moment, MOMENT_SYMBOLS[:dim]).subs(c_s_sq_helper, c_s_sq) for moment in moments]
    if order is not None:
        result = [removeHigherOrderTerms(r, order, u) for r in result]

    return result


def getMomentsOfDiscreteMaxwellianEquilibrium(stencil, moments,
                                              rho=sp.Symbol("rho"), u=tuple(sp.symbols("u_0 u_1 u_2")),
                                              order=None, compressible=True):
    """
    Compute moments of discrete maxwellian equilibrium

    :param stencil: stencil is required to compute moments of discrete function
    :param moments: moments in polynomial or exponent-tuple form
    :param rho: symbol or value for the density
    :param u: symbols or values for the macroscopic velocity
    :param order: highest order of u terms
    :param compressible: compressible or incompressible form
    """
    if order is None:
        order = 4
    mb = discreteMaxwellianEquilibrium(stencil, rho, u, order, compressible)
    return tuple([discreteMoment(mb, moment, stencil).expand() for moment in moments])


def compressibleToIncompressibleMomentValue(term, rho=sp.Symbol("rho"), u=sp.symbols("u_:3")):
    term = sp.sympify(term)
    term = term.expand()
    if term.func != sp.Add:
        args = [term, ]
    else:
        args = term.args

    res = 0
    for t in args:
        containedSymbols = t.atoms(sp.Symbol)
        if rho in containedSymbols and len(containedSymbols.intersection(set(u))) > 0:
            res += t / rho
        else:
            res += t
    return res

