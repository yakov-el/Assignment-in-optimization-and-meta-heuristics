import numpy as np


def minimize(f, g, x0):
    eps = 0.000001
    mu = 0.01
    max_iter = 10000000000
    a, b = [-10] * len(x0), [10] * len(x0)

    # Axis
    def Axis(a, b, f, point, eps, max_iter):
       for j in range(10):  # running the code for ten times
        for i in range(len(a)): # iterating on each axis
            point[i] = golden_section(a[i], b[i], point, f, i, max_iter, eps) # updating point at the i axis
       return point

    # Golden section
    def golden_section(ai, bi, point, f, i, max_iter, eps):
        phi = (1 + (5 ** 0.5)) / 2 # golden ratio
        left, right = point.copy(), point.copy() # a and b
        left[i], right[i] = ai, bi  # updating those points to be given a and given b in the given axis
        la = bi - (1 / phi) * (bi - ai) # calculating lambda
        mu = ai + (1 / phi) * (bi - ai) # calculating mu
        left[i] = la  # updating those points to be calc la and calc mu in the given axis
        right[i] = mu
        val_la = f(left) # checking the value of the points with  mu and lambda as f is not really unidimensional
        val_mu = f(right)
        for k in range(max_iter):
            if abs(val_la - val_mu) < eps: # breaking condition
                break
            if val_la > val_mu:
                ai = la
                val_la = val_mu
                la = mu
                mu = ai + (1 / phi) * (bi - ai) # calculating new mu
                right[i] = mu  # updating
                val_mu = f(right)  # recalculating the value of the right border
            else:
                bi = mu
                val_mu = val_la
                mu = la
                la = bi - (1 / phi) * (bi - ai) # calculating new lam
                left[i] = la
                val_la = f(left) # recalculating the value of the left border
        return (ai + bi) / 2 # the average between the borders of the interval

    def alpha(g_funcs, point): # penalty function
        penalty = 0
        g_equal = g_funcs(point) # the equal constraints
        for g in g_equal:
            penalty += max(0, g) # if the constraint is broken add to penalty else zero
        return penalty

    def theta(point, mu):
        return (f(point) + mu * alpha(g, point)) # the value of the theta according to the algorithm

    def penalty(f, point, eps, N, a, b):
        mu = 0.1
        def T(point):
            return theta(point, mu)
        curr_point = np.round(Axis(a,b,f, point, eps, N),3)
        print("The optimal point of f without constraints is:", curr_point)
        print("The optimal value of f without constraints is:", f(curr_point))
        for k in range(100):
            if (alpha(g, curr_point)) < eps and k > 0: # breaking condition
                break
            curr_point = Axis(a,b,T, point, eps, N) # updating current point using the axis func
            print("|k: ", k+1 ,"|Mu: ", mu, "|optimal point: ", np.round(curr_point,3), "|f(x): ", round(f(curr_point),5),
                  "|Alpha: ", alpha(g, point), "|mu* alpha: ", mu * alpha(g, point))
            mu = mu * 10 # updating mu
        return curr_point, f(curr_point)
    point, value = penalty(f,x0,eps,max_iter,a,b)
    return point , value






