import numpy as np
from numpy.linalg import eig, inv

import b2ac.preprocess
import b2ac.fit
import b2ac.conversion


'''
A set of functions designed to fit a set of given points to an ellipse. These functions were outright stolen from
Nicky van Foreest at http://nicky.vanforeest.com/misc/fitEllipse/fitEllipse.html

Thank you, Nicky. 
'''

def fitEllipse(x,y):
    '''Returns the parameters for the best fitting ellipse of points x,y'''
    x = x[:,np.newaxis]
    y = y[:,np.newaxis]
    D = np.hstack((x*x, x*y, y*y, x, y, np.ones_like(x)))
    S = np.dot(D.T,D)
    C = np.zeros([6,6])
    C[0,2] = C[2,0] = 2; C[1,1] = -1
    E, V =  eig(np.dot(inv(S), C))
    n = np.argmax(E)
    a = V[:,n]
    return a

def fitEllipse_Halir_Flusser(x,y):
    '''Correction to the above ellipse fitting by Halir and Flusser'''
    x = x[:,np.newaxis]
    y = y[:,np.newaxis]
    D1 = np.hstack((x*x, x*y, y*y))
    D2 = np.hstack((x, y, np.ones_like(x)))
    S1 = np.dot(D1.T,D1)
    S2 = np.dot(D1.T,D2)
    S3 = np.dot(D2.T,D2)
    T = - inv(S3) * S2.T
    M = S1 + S2 * T
    M = np.hstack((M[3,:] / 2., -1 * M[2,:], M[1,:] / 2.))
    evec, eval = eig(M)
    cond = 4 * evec[1,:] * evec[3,:] - evec[2,:] ** 2
    a1 = evec[:,np.where(cond > 0)]
    return [a1, T*a1]

def fitEllipse_b2ac(x,y):
    points = np.column_stack((x,y))
    conic_double = b2ac.fit.fit_improved_B2AC_double(points)
    general_form_double = b2ac.conversion.conic_to_general_1(conic_double)
    return general_form_double

def ellipse_center(a):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    num = b*b-a*c
    x0=(c*d-b*f)/num
    y0=(a*f-b*d)/num
    return np.array([x0,y0])


def ellipse_angle_of_rotation2( a ):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    if b == 0:
        if a > c:
            return 0
        else:
            return np.pi/2
    else:
        if a > c:
            return np.arctan(2*b/(a-c))/2
        else:
            return np.pi/2 + np.arctan(2*b/(a-c))/2


def ellipse_axis_length( a ):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    up = 2*(a*f*f+c*d*d+g*b*b-2*b*d*f-a*c*g)
    down1=(b*b-a*c)*( (c-a)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    down2=(b*b-a*c)*( (a-c)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    res1=np.sqrt(up/down1)
    res2=np.sqrt(up/down2)
    return np.array([res1, res2])