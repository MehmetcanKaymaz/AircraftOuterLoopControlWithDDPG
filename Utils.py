import math


class Utils:
    def __init__(self):
        self.info="info"

    def feetTometer(self,x):
        return 0.3048 * x

    def lbsTonewton(self,x):
        return 4.4482 * x

    def slugoverft2Tokgoverm2(self,x):
        return 157.087464 * x

    def ft2Tom2(self,x):
        return 0.3048 * 0.3048 * x

    def slugTokg(self,x):
        return 14.5939029 * x

    def slugoverft3Tokgoverm3(self,x):
        return 515.378818 * x;

    def degTorad(self,x):
        return x * math.pi / 180

        # Model function

    def qCalc(self,rho, V):
        return 0.5 * rho * pow(V, 2)

    def CLcal(self,CLo, CLalpha, alpha):
        return CLo + CLalpha * alpha

    def CDcal(self,CDo, CDalpha, alpha):
        return CDo + CDalpha * alpha

    def Cycal(self,Cybeta, beta):
        return Cybeta * beta

    def radTodeg(self,x):
        return x * 180 / math.pi

    def sin(self,x):
        return math.sin(x)

    def cos(self,x):
        return math.cos(x)

    def tan(self,x):
        return self.sin(x) / self.cos(x)

    def sec(self,x):
        return 1 / self.cos(x)