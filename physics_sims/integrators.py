def velocity_verlet(dt, t, x, v, a, calc_acceleration):
    r'''Velocity Verlet integration
    https://en.wikipedia.org/wiki/Verlet_integration

    Args:
        dt: Amount of time to integrate over
        t: Current time
        x: Current position coordinate
        v: Current velocity
        a: Current acceleration
        calc_acceleration: Callable of the form ``(t, x, v) -> a``

    Returns:
        (t, x, v, a)
    '''
    x_next = x + v * dt + 0.5 * a * dt**2
    t_next = t + dt
    a_next = calc_acceleration(t_next, x_next, v)
    v_next = v + 0.5 * (a + a_next) * dt

    return t_next, x_next, v_next, a_next

def runge_kutta_4th_order(dt, t, x, v, calc_acceleration):
    r'''Classic Fourth-order Runge-Kutta integration method
    https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods

    Args:
        dt: Amount of time to integrate over
        t: Current time
        x: Current position coordinate
        v: Current velocity
        calc_acceleration: Callable of the form ``(t, x, v) -> a``

    Returns:
        (t, x, v)
    '''
    k1v = dt * calc_acceleration(t, x, v)
    k1x = dt * v

    k2v = dt * calc_acceleration(
        t + 0.5 * dt,
        x + 0.5 * k1x,
        v + 0.5 * k1v)
    k2x = dt * (v + 0.5 * k1v)

    k3v = dt * calc_acceleration(
        t + 0.5 * dt,
        x + 0.5 * k2x,
        v + 0.5 * k2v)
    k3x = dt * (v + 0.5 * k2v)

    k4v = dt * calc_acceleration(
        t + dt,
        x + k3x,
        v + k3v)
    k4x = dt * (v + k3v)

    t_next = t + dt
    x_next = x + (k1x + 2.0 * (k2x + k3x) + k4x) / 6.0
    v_next = v + (k1v + 2.0 * (k2v + k3v) + k4v) / 6.0

    return t_next, x_next, v_next