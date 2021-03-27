from Utils import Utils

u=Utils()

class Controller:
    def __init__(self):
        self.info="info"

    def BacksteppingController(self,states, trej):
        # states
        Vt = states[0]
        alpha = states[1]
        beta = states[2]
        phi = states[3]
        theta = states[4]
        psi = states[5]
        p = states[6]
        q = states[7]
        r = states[8]
        Ft = states[9]

        phiref = trej[0]
        thetaref = trej[1]
        psiref = trej[2]
        Vtref = trej[3]

        # ModelParameters
        S = u.ft2Tom2(184)
        m = u.slugTokg(85.4)
        rho = u.slugoverft3Tokgoverm3(0.002378)
        g = 9.81

        # Aerodynamic coefficients
        CDo = 0.05
        CDalpha = 0.33

        q2 = u.qCalc(rho, Vt)
        D = q2 * S * u.CDcal(CDo, CDalpha, alpha)

        g1 = g * (-u.cos(alpha) * u.cos(beta) * u.sin(theta) + u.sin(beta) * u.cos(theta) * u.sin(phi) + u.sin(alpha) * u.cos(
            beta) * u.cos(
            theta) * u.cos(phi))

        # Roll angle control
        k1 = 5
        k2 = 8
        f = q * u.sin(phiref) * u.tan(theta) + r * u.cos(phiref) * u.tan(theta)
        u1 = -k2 * (p + k1 * (phi - phiref) + f)

        # Pitch angle control
        k1 = 5
        k2 = 8
        f = q * u.sin(phi) - r * u.sin(phi)
        u2 = -k2 * (q + k1 * (theta - thetaref) + f)

        # Yaw angle control
        k1 = 6
        k2 = 6
        f = q * u.sin(phi) / u.cos(theta) + r * u.cos(phi) / u.cos(theta)
        u3 = -k2 * (r + k1 * (psi - psiref) + f)

        # Velocity Control
        k1 = 5
        k2 = 4000
        f = (-D + Ft * u.cos(alpha) * u.cos(beta) + m * g1) / m
        ut = -k2 * (k1 * (Vt - Vtref) + f)

        u1 = max(min(u1, 1), -1)
        u2 = max(min(u2, 1), -1)
        u3 = max(min(u3, 1), -1)
        return [u1, u2, u3, ut]

    def FeedbackLinerizationController(self,states, trej):
        # states
        Vt = states[0]
        alpha = states[1]
        beta = states[2]
        phi = states[3]
        theta = states[4]
        psi = states[5]
        p = states[6]
        q = states[7]
        r = states[8]
        Ft = states[9]

        phiref = trej[0]
        thetaref = trej[1]
        psiref = trej[2]
        Vtref = trej[3]

        # ModelParameters
        S = u.ft2Tom2(184)
        m = u.slugTokg(85.4)
        rho = u.slugoverft3Tokgoverm3(0.002378)
        g = 9.81

        # Aerodynamic coefficients
        CDo = 0.05
        CDalpha = 0.33

        q2 = u.qCalc(rho, Vt)
        D = q2 * S * u.CDcal(CDo, CDalpha, alpha)

        g1 = g * (-u.cos(alpha) * u.cos(beta) * u.sin(theta) + u.sin(beta) * u.cos(theta) * u.sin(phi) + u.sin(alpha) * u.cos(
            beta) * u.cos(
            theta) * u.cos(phi))

        # Roll angle control
        k1 = 4
        k2 = 6
        f = q * u.sin(phi) * u.tan(theta) + r * u.cos(phi) * u.tan(theta)
        pref = -k1 * (phi - phiref) - f
        u1 = -k2 * (p - pref)

        # Pitch angle control
        k1 = 4
        k2 = 6
        f = q * u.sin(phi) - r * u.sin(phi)
        qref = -k1 * (theta - thetaref) - f
        u2 = -k2 * (q - qref)

        # Yaw angle control
        k1 = 4
        k2 = 5
        f = q * u.sin(phi) * u.sec(theta) + r * u.cos(phi) * u.sec(theta)
        rref = -k1 * (psi - psiref) - f
        u3 = -k2 * (r - rref)

        # Velocity Control
        k1 = 5000
        k2 = 50
        f = (-D + Ft * u.cos(alpha) * u.cos(beta) + m * g1) / m
        Ftref = -k1 * (Vt - Vtref) - f
        ut = -k2 * (Ft - Ftref)

        u1 = max(min(u1, 1), -1)
        u2 = max(min(u2, 1), -1)
        u3 = max(min(u3, 1), -1)
        return [u1, u2, u3, ut]

    def PIDAngleController(self,states, trej):
        # states
        Vt = states[0]
        alpha = states[1]
        beta = states[2]
        phi = states[3]
        theta = states[4]
        psi = states[5]
        p = states[6]
        q = states[7]
        r = states[8]
        Ft = states[9]

        phiref = trej[0]
        thetaref = trej[1]
        psiref = trej[2]
        Vtref = trej[3]

        k1 = 2
        u1 = -k1 * (phi - phiref)

        k1 = 2
        u2 = -k1 * (theta - thetaref)

        k1 = 2
        u3 = -k1 * (psi - psiref)

        k1 = 40
        ut = Ft + k1 * (Vtref - Vt)

        return [u1, u2, u3, ut]

    def PIDController(self,states, trej, ErrSum, LastErr, dtau):
        # states
        Vt = states[0]
        alpha = states[1]
        beta = states[2]
        phi = states[3]
        theta = states[4]
        psi = states[5]
        p = states[6]
        q = states[7]
        r = states[8]
        Ft = states[9]

        trejref = self.PIDAngleController(states, trej)

        # trejectory
        pref = trejref[0]
        qref = trejref[1]
        rref = trejref[2]
        Ftref = trejref[3]

        Err = [pref - p, qref - q, rref - r, Ftref - Ft]

        ErrDiff = [0, 0, 0, 0]

        for i in range(4):
            ErrSum[i] = ErrSum[i] + Err[i] * dtau
            ErrDiff[i] = (Err[i] - LastErr[i]) / dtau

        # roll control
        Kp = 5
        Ki = 0.7
        Kd = 0.06
        u1 = Kp * Err[0] + Ki * ErrSum[0] + Kd * ErrDiff[0]

        # pitch control
        Kp = 4.5
        Ki = 0.5
        Kd = 0.06
        u2 = Kp * Err[1] + Ki * ErrSum[1] + Kd * ErrDiff[1]

        # yaw control
        Kp = 4
        Ki = 0.6
        Kd = 0.05
        u3 = Kp * Err[2] + Ki * ErrSum[2] + Kd * ErrDiff[2]

        # velocity control
        Kp = 15
        Ki = 1
        Kd = 100
        ut = Kp * Err[3] + Ki * ErrSum[3] + Kd * ErrDiff[3]

        u1 = max(min(u1, 1), -1)
        u2 = max(min(u2, 1), -1)
        u3 = max(min(u3, 1), -1)
        return [[u1, u2, u3, ut], ErrSum, Err]