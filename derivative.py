"""
Custom derivative object for sympy targeted at Chapman Enskog analysis
"""
import sympy as sp
from sympy.core.cache import cacheit
from functools import reduce
import operator

__author__ = "Martin Bauer"
__copyright__ = "Copyright 2018, Martin Bauer"
__license__ = "GPL"
__version__ = "3"
__email__ = "martin.bauer@fau.de"


def defaultDiffSortKey(d):
    return str(d.superscript), str(d.target)


class Diff(sp.Expr):
    """
    Sympy Node representing a derivative. The difference to sympy's built in differential is:
        - shortened latex representation
        - all simplifications have to be done manually
        - optional marker displayed as superscript
    """
    is_number = False
    is_Rational = False

    def __new__(cls, argument, target=-1, superscript=-1, **kwargs):
        if argument == 0:
            return sp.Rational(0, 1)
        return sp.Expr.__new__(cls, argument.expand(), sp.sympify(target), sp.sympify(superscript), **kwargs)

    @property
    def is_commutative(self):
        anyNonCommutative = any(not s.is_commutative for s in self.atoms(sp.Symbol))
        if anyNonCommutative:
            return False
        else:
            return True

    def getArgRecursive(self):
        """Returns the argument the derivative acts on, for nested derivatives the inner argument is returned"""
        if not isinstance(self.arg, Diff):
            return self.arg
        else:
            return self.arg.getArgRecursive()

    def changeArgRecursive(self, newArg):
        """Returns a Diff node with the given 'newArg' instead of the current argument. For nested derivatives
        a new nested derivative is returned where the inner Diff has the 'newArg'"""
        if not isinstance(self.arg, Diff):
            return Diff(newArg, self.target, self.superscript)
        else:
            return Diff(self.arg.changeArgRecursive(newArg), self.target, self.superscript)

    def splitLinear(self, functions):
        """
        Applies linearity property of Diff: i.e.  'Diff(c*a+b)' is transformed to 'c * Diff(a) + Diff(b)'
        The parameter functions is a list of all symbols that are considered functions, not constants.
        For the example above: functions=[a, b]
        """
        constant, variable = 1, 1

        if self.arg.func != sp.Mul:
            constant, variable = 1, self.arg
        else:
            for factor in normalizeProduct(self.arg):
                if factor in functions or isinstance(factor, Diff):
                    variable *= factor
                else:
                    constant *= factor

        if isinstance(variable, sp.Symbol) and variable not in functions:
            return 0

        if isinstance(variable, int) or variable.is_number:
            return 0
        else:
            return constant * Diff(variable, target=self.target, superscript=self.superscript)

    @property
    def arg(self):
        """Expression the derivative acts on"""
        return self.args[0]

    @property
    def target(self):
        """Subscript, usually the variable the Diff is w.r.t. """
        return self.args[1]

    @property
    def superscript(self):
        """Superscript, used as the Chapman Enskog order index"""
        return self.args[2]

    def _latex(self, printer, *args):
        result = "{\partial"
        if self.superscript >= 0:
            result += "^{(%s)}" % (self.superscript,)
        if self.target != -1:
            result += "_{%s}" % (printer.doprint(self.target),)

        contents = printer.doprint(self.arg)
        if isinstance(self.arg, int) or isinstance(self.arg, sp.Symbol) or self.arg.is_number or self.arg.func == Diff:
            result += " " + contents
        else:
            result += " (" + contents + ") "

        result += "}"
        return result

    def __str__(self):
        return "D(%s)" % self.arg


class DiffOperator(sp.Expr):
    """
    Un-applied differential, i.e. differential operator
    Its args are:
        - target: the differential is w.r.t to this variable.
                 This target is mainly for display purposes (its the subscript) and to distinguish DiffOperators
                 If the target is '-1' no subscript is displayed
        - superscript: optional marker displayed as superscript
                        is not displayed if set to '-1'
    The DiffOperator behaves much like a variable with special name. Its main use is to be applied later, using the
    DiffOperator.apply(expr, arg) which transforms 'DiffOperator's to applied 'Diff's
    """
    is_commutative = True
    is_number = False
    is_Rational = False

    def __new__(cls, target=-1, superscript=-1, **kwargs):
        return sp.Expr.__new__(cls, sp.sympify(target), sp.sympify(superscript), **kwargs)

    @property
    def target(self):
        return self.args[0]

    @property
    def superscript(self):
        return self.args[1]

    def _latex(self, printer, *args):
        result = "{\partial"
        if self.superscript >= 0:
            result += "^{(%s)}" % (self.superscript,)
        if self.target != -1:
            result += "_{%s}" % (self.target,)
        result += "}"
        return result

    @staticmethod
    def apply(expr, argument):
        """
        Returns a new expression where each 'DiffOperator' is replaced by a 'Diff' node.
        Multiplications of 'DiffOperator's are interpreted as nested application of differentiation:
        i.e. DiffOperator('x')*DiffOperator('x') is a second derivative replaced by Diff(Diff(arg, x), t)
        """
        def handleMul(mul):
            args = normalizeProduct(mul)
            diffs = [a for a in args if isinstance(a, DiffOperator)]
            if len(diffs) == 0:
                return mul * argument
            rest = [a for a in args if not isinstance(a, DiffOperator)]
            diffs.sort(key=defaultDiffSortKey)
            result = argument
            for d in reversed(diffs):
                result = Diff(result, target=d.target, superscript=d.superscript)
            return prod(rest) * result

        expr = expr.expand()
        if expr.func == sp.Mul or expr.func == sp.Pow:
            return handleMul(expr)
        elif expr.func == sp.Add:
            return expr.func(*[handleMul(a) for a in expr.args])
        else:
            return expr * argument


class CeMoment(sp.Symbol):
    def __new__(cls, name, *args, **kwds):
        obj = CeMoment.__xnew_cached_(cls, name, *args, **kwds)
        return obj

    def __new_stage2__(cls, name, momentTuple, superscript=-1):
        obj = super(CeMoment, cls).__xnew__(cls, name)
        obj.momentTuple = momentTuple
        while len(obj.momentTuple) < 3:
            obj.momentTuple = obj.momentTuple + (0,)
        obj.superscript = superscript
        return obj

    __xnew__ = staticmethod(__new_stage2__)
    __xnew_cached_ = staticmethod(cacheit(__new_stage2__))

    def _hashable_content(self):
        superClassContents = list(super(CeMoment, self)._hashable_content())
        return tuple(superClassContents + [hash(repr(self.momentTuple)), hash(repr(self.superscript))])

    def __getnewargs__(self):
        return self.name, self.momentTuple, self.superscript

    def _latex(self, printer, *args):
        coordStr = []
        for i, comp in enumerate(self.momentTuple):
            coordStr += [str(i)] * comp
        coordStr = "".join(coordStr)
        result = "{%s_{%s}" % (self.name, coordStr)
        if self.superscript >= 0:
            result += "^{(%d)}}" % (self.superscript,)
        else:
            result += "}"
        return result

    def __repr__(self):
        return "%s_(%d)_%s" % (self.name, self.superscript, self.momentTuple)

    def __str__(self):
        return "%s_(%d)_%s" % (self.name, self.superscript, self.momentTuple)


# ----------------------------------------------------------------------------------------------------------------------


def d(arg, *args):
    """Shortcut to create nested derivatives"""
    args = sorted(args, reverse=True)
    res = arg
    for i in args:
        res = Diff(res, i)
    return res


def expandUsingLinearity(expr, functions=None, constants=None):
    """
    Expands all derivative nodes by applying Diff.splitLinear
    :param expr: expression containing derivatives
    :param functions: sequence of symbols that are considered functions and can not be pulled before the derivative.
                      if None, all symbols are viewed as functions
    :param constants: sequence of symbols which are considered constants and can be pulled before the derivative
    """
    if functions is None:
        functions = expr.atoms(sp.Symbol)
        if constants is not None:
            functions.difference_update(constants)

    if isinstance(expr, Diff):
        arg = expandUsingLinearity(expr.arg, functions)
        if hasattr(arg, 'func') and arg.func == sp.Add:
            result = 0
            for a in arg.args:
                result += Diff(a, target=expr.target, superscript=expr.superscript).splitLinear(functions)
            return result
        else:
            diff = Diff(arg, target=expr.target, superscript=expr.superscript)
            if diff == 0:
                return 0
            else:
                return diff.splitLinear(functions)
    else:
        newArgs = [expandUsingLinearity(e, functions) for e in expr.args]
        result = sp.expand(expr.func(*newArgs) if newArgs else expr)
        return result


def normalizeDiffOrder(expression, functions=None, constants=None, sortKey=defaultDiffSortKey):
    """
    Assumes order of differentiation can be exchanged. Changes the order of nested Diffs to a standard order defined
    by the sorting key 'sortKey' such that the derivative terms can be further simplified 
    """
    def visit(expr):
        if isinstance(expr, Diff):
            nodes = [expr]
            while isinstance(nodes[-1].arg, Diff):
                nodes.append(nodes[-1].arg)

            processedArg = visit(nodes[-1].arg)
            nodes.sort(key=sortKey)

            result = processedArg
            for d in reversed(nodes):
                result = Diff(result, target=d.target, superscript=d.superscript)
            return result
        else:
            newArgs = [visit(e) for e in expr.args]
            return expr.func(*newArgs) if newArgs else expr

    expression = expandUsingLinearity(expression.expand(), functions, constants).expand()
    return visit(expression)


def zeroDiffs(expr, label):
    """Replaces all differentials with the given target by 0"""
    def visit(e):
        if isinstance(e, Diff):
            if e.target == label:
                return 0
        newArgs = [visit(arg) for arg in e.args]
        return e.func(*newArgs) if newArgs else e
    return visit(expr)


def prod(seq):
    """Takes a sequence and returns the product of all elements"""
    return reduce(operator.mul, seq, 1)


def normalizeProduct(product):
    """
    Expects a sympy expression that can be interpreted as a product and
    - for a Mul node returns its factors ('args')
    - for a Pow node with positive integer exponent returns a list of factors
    - for other node types [product] is returned
    """
    def handlePow(power):
        if power.exp.is_integer and power.exp.is_number and power.exp > 0:
            return [power.base] * power.exp
        else:
            return [power]

    if product.func == sp.Pow:
        return handlePow(product)
    elif product.func == sp.Mul:
        result = []
        for a in product.args:
            if a.func == sp.Pow:
                result += handlePow(a)
            else:
                result.append(a)
        return result
    else:
        return [product]