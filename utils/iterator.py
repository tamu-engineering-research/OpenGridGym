import warnings

def episode_iterator(t_prev=-1, max_steps=100, num_iter=None, iterator=range, warn=True):
    '''
    Iterate which yields time, starting at t_prev
    and ending after num_iter iterations.

    t_prev: int
        Time stamp already visited (i.e. "previous").
        Iterator will begin one time step after this.

    max_steps: int
        Maxmimum number of steps (in case num_iter is too large).

    num_iter: int
        Number of time steps to iterate over.
        If left as None, (max_steps - t_prev) is assumed.

    iterator: Iterator[int]
        An iterator which yields current time step, and either
        shows progress or not, depending on whether uses chooses
        range or trange (tqdm) for this value, respectively.
        More generally, you can specify any iterator provided
        that it yields the current time (int) and takes two
        arguments: (t_prev, t_end+1).

    warn: bool
        If True, it will warn the user (not raise error) if num_iter
        is large enough that it exceeds maximum remaining steps.

    Example:
        
        >>> for t in episode_iterator(50, num_iter=3): # progress for 3 new steps
        ...     print(t)
        ...
        51
        52
        53

    '''
    # End time based on number of iterations
    if num_iter is None:
        t_end = max_steps - 1
        num_iter = t_end - t_prev
    elif num_iter < 0:
        raise ValueError('Negative time traversal not supported.')
    else:
        t_end = t_prev + num_iter
        if t_end > max_steps - 1:
            t_end = max_steps - 1
            if warn:
                warnings.warn(f'\n\nnum_iter truncated to {t_end - t_prev} '
                              f'to avoid exceeding maximum end time.\n')

    # Progress and yield time
    for t in iterator(t_prev+1, t_end+1):

        yield t

if __name__ == '__main__':

    for t in episode_iterator(50, num_iter=5, max_steps=100):
        print(t)

    for t in episode_iterator(50, num_iter=5, max_steps=54):
        print(t)