from inspect import signature
from types import MethodType, FunctionType

class MethodSignatureError(Exception):
    def __init__(self, expected_sig=None, actual_sig=None):

        message = f'Expected signature {expected_sig}, instead got signature {actual_sig}'
        super().__init__(message)


def ensure_method_signature(expected_method=None, actual_method=None):
    if not isinstance(expected_method, MethodType):
        ValueError(f'{expected_method} is not a method.')
    else:
        expected_params = [*map(str, signature(expected_method).parameters.values())]
        actual_params = [*map(str, signature(actual_method).parameters.values())]

        expected_params = ["self"] + expected_params
        expected_sig = f'({", ".join(expected_params)})'

        actual_sig = f'({", ".join(actual_params)})'

        if actual_sig != expected_sig:
            raise MethodSignatureError(expected_sig=expected_sig, actual_sig=actual_sig)


def update_instance_method(instance, old_method_str, new_method):
    '''
    Updates an instance's method to a new one, but first checks
    if the signature of the new method matches that of the old one.

    instance: object
        An instance of any object

    old_method_str: str
        The string which uniquely identifies the method you seek
        to update. Strictly speaking, this is how you get the method:
            getattr(instance, old_method_str)

    new_method: FunctionType
        Any function whose argument must begin with 'self'.
        If the function's signature matches that of the instance's
        old method, the instance's method is updated.

    '''
    
    # Make sure it's safe to do so first
    old_method = getattr(instance, old_method_str)
    ensure_method_signature(expected_method=old_method, actual_method=new_method)
    
    # Update the method
    setattr(instance, old_method_str, MethodType(new_method, instance))


if __name__ == '__main__':

    class A:
        def f(self, x, *args, **kwargs):
            return 5

    def g(self, x, *args, **kwargs):
        return 6

    L = lambda self, x, *args, **kwargs: 7

    a = A()

    update_instance_method(a, 'f', g)
    update_instance_method(a, 'f', L)
    print(a.f(None))