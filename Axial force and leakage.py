import numpy as np


class CentrifugalMultistagePump:
    def __init__(self, h, q, n, i, rho, nu, g, d2, d2_1, dvt, dvt_1, d0, d0_1, eta, eta_1, delta, delta_1):
        self.H = h
        self.Q = q / 3600
        self.n = n / 60
        self.i = i
        self.rho = rho
        self.nu = nu
        self.H1 = h / i
        self.D2 = d2 / 1000
        self.D2_1 = d2_1 / 1000
        self.Dvt = dvt / 1000
        self.Dvt_1 = dvt_1 / 1000
        self.D0 = d0 / 1000
        self.D0_1 = d0_1 / 1000
        self.R2 = self.D2 / 2
        self.R2_1 = self.D2_1 / 2
        self.Rvt = self.Dvt / 2
        self.Rvt_1 = self.Dvt_1 / 2
        self.R0 = self.D0 / 2
        self.R0_1 = self.D0_1 / 2
        self.eta = eta / 100
        self.delta = delta / 1000
        self.eta_1 = eta_1 / 100
        self.delta_1 = delta_1 / 1000
        self.g = g
        self.Ht = self.H1 / self.eta
        self.Ht_1 = self.H1 / self.eta_1
        self.u2 = np.pi * self.D2 * self.n
        self.u2_1 = np.pi * self.D2_1 * self.n
        self.Hp = (1 - self.g * self.Ht / 2 / self.u2 ** 2) * self.Ht
        self.Hp_1 = (1 - self.g * self.Ht_1 / 2 / self.u2_1 ** 2) * self.Ht_1
        self.Di = self.D0 + 2 * self.delta
        self.Di_1 = self.D0_1 + 2 * self.delta_1
        self.Ri = self.Di / 2
        self.Ri_1 = self.Di_1 / 2
        self.v0_Q = (self.Q / np.pi) / (self.R0 ** 2 - self.Rvt ** 2)
        self.v0_Q_1 = (self.Q / np.pi) / (self.R0_1 ** 2 - self.Rvt_1 ** 2)
        self.Fz_External = -self.rho * self.g * np.pi * (self.Ri ** 2 - self.Rvt ** 2) * (
                self.Hp - self.u2 ** 2 / 8 / self.g * (1 - (self.Ri ** 2 + self.Rvt ** 2) / 2 / self.R2 ** 2))
        self.Fz_External_1 = -self.rho * self.g * np.pi * (self.Ri_1 ** 2 - self.Rvt_1 ** 2) * (
                self.Hp_1 - self.u2_1 ** 2 / 8 / self.g * (1 - (self.Ri_1 ** 2 + self.Rvt_1 ** 2) / 2 / self.R2_1 ** 2))
        self.Fz_Internal = self.rho * self.Q * self.v0_Q
        self.Fz_Internal_1 = self.rho * self.Q * self.v0_Q_1
        self.Fz = self.Fz_Internal + self.Fz_External
        self.Fz_1 = self.Fz_Internal_1 + self.Fz_External_1
        self.rotor_Fz = self.Fz * (self.i - 1) + self.Fz_1
        self.P_out = self.rho * self.g * (self.H1 * (self.i - 1) + self.Hp)
        self.H_st = self.P_out / self.rho / self.g

    def piston_axial_force(self, dvt, d):
        fz_piston = self.P_out * 0.25 * np.pi * (d ** 2 - dvt ** 2)
        return fz_piston

    def piston_diameter(self, dvt):
        d = np.sqrt(abs(self.rotor_Fz) / (0.25 * np.pi * self.P_out) + dvt ** 2)
        print(f"Рекомнедуемый диаметр поршня = {d * 1000} мм")
        if input("Хотите ли вы уточнить диаметр поршня?(да/нет): ").lower() == "да":
            d = float(input("Введите уточненный диаметр поршня, мм: ")) / 1000
            print(f"Остаточная осевая сила = {pump.piston_axial_force(dvt, d) + self.rotor_Fz}, Н")
            return d
        else:
            return d

    def pressure_drop_on_gap(self, d, p, stage1):
        if stage1:
            hpd_piston = p - self.u2_1 ** 2 / 8 / self.g * (1 - (d / self.D2_1) ** 2)
        else:
            hpd_piston = p - self.u2 ** 2 / 8 / self.g * (1 - (d / self.D2) ** 2)
        return hpd_piston

    def lambda_rough(self, clearance, rough):
        _lambda = 1 / (1.74 + 2 * np.log10(clearance / rough)) ** 2
        return _lambda

    def lambda_smooth(self, re):
        _lambda = 0.3164 / re ** (1 / 4)
        return _lambda

    def flow_rate_ratio(self, l, clearance, ratio=1, _lambda=0.04):
        mu = 1 / np.sqrt((_lambda * ratio * l / 2 / clearance) + 1.5)
        return mu

    def clearance_velocity_ratio(self, q, d, clearance):
        velocity_q = q / np.pi / d / clearance
        velocity_radial = self.n * np.pi * d / 2
        velocity = np.sqrt(velocity_q ** 2 + velocity_radial ** 2)
        ratio = velocity / velocity_q
        re = 2 * clearance * velocity / self.nu
        _lambda = pump.lambda_smooth(re)
        laminar = 11.6 * self.nu / velocity * np.sqrt(8 / _lambda)
        return ratio, laminar, re

    def leakage_calculation(self, l, clearance, d, rough, p, stage1): #stage1 - это первая ступень (T/F)
        _lambda = pump.lambda_rough(clearance, rough)
        q, laminar, re = pump.leakage_correction(l, clearance, d, _lambda, p, stage1)
        if laminar < rough:
            return q
        else:
            _lambda = pump.lambda_smooth(re)
            q, laminar, re = pump.leakage_correction(l, clearance, d, _lambda, p, stage1)
            return q

    def leakage_correction(self, l, clearance, d, _lambda, p, stage1):
        mu = pump.flow_rate_ratio(l, clearance, _lambda=_lambda)
        hpd = pump.pressure_drop_on_gap(d, p, stage1)
        q = pump.leakage(mu, d, clearance, hpd)
        k, laminar, re = pump.clearance_velocity_ratio(q, d, clearance)
        delta = (k - 1) / k * 100
        while delta > 0.001:
            ki = k
            mu = pump.flow_rate_ratio(l, clearance, k, _lambda)
            q = pump.leakage(mu, d, clearance, hpd)
            k, laminar, re = pump.clearance_velocity_ratio(q, d, clearance)
            delta = (k - ki) / k * 100
        return q, laminar, re

    def leakage(self, mu, d, clearance, hpd):
        leakage = mu * np.pi * d * clearance * np.sqrt(2 * self.g * hpd)
        return leakage


if __name__ == "__main__":
    pump = CentrifugalMultistagePump(157, 320, 1480, 3, 938.8, 1.827 / 10 ** 7, 9.81, 416, 417, 152, 100, 213, 240,
                                     92.7, 91.7, 5, 5)  # h, q, n, i, rho, nu, g, d2,d2_1 dvt, dvt_1, d0, d0_1, eta,
                                                        # eta_1, delta, delta_1
    # --Параметры поршня------------------------------------------------------------------------------------------------
    d_piston_shaft = 152  # , мм
    l_piston = 100  # , мм
    clearance_piston = 0.3  # , мм
    rough_piston = 6.3  # Rz, мкм
    # --Параметры переднего уплотнения рядовой ступени------------------------------------------------------------------
    l_gap = 20  # , мм
    clearance_gap = 0.3  # , мм
    rough_gap = 6.3  # Rz, мкм
    # --Параметры переднего уплотнения первой ступени------------------------------------------------------------------
    l_gap_1 = 40  # , мм
    clearance_gap_1 = 0.3  # , мм
    rough_gap_1 = 6.3  # Rz, мкм
    # --Расчет поршня---------------------------------------------------------------------------------------------------
    print(pump.Fz, pump.Fz_1, pump.rotor_Fz)
    d_piston = pump.piston_diameter(d_piston_shaft/1000)
    # --Расчет утечек в поршне------------------------------------------------------------------------------------------
    q_piston = pump.leakage_calculation(l_piston/1000, clearance_piston/1000, d_piston, rough_piston/1000000,
                                        pump.H_st, False)
    print(f"Расход утечек через поршень: {q_piston * 3600} м3/ч")
    # --Расчет утечек в уплотнениях-------------------------------------------------------------------------------------
    q_gap = pump.leakage_calculation(l_gap/1000, clearance_gap/1000, pump.Di, rough_gap/1000000, pump.Hp, False)
    print(f"Расход утечек через переднее уплотнение рядовой ступени: {q_gap * 3600} м3/ч")
    q_gap_1 = pump.leakage_calculation(l_gap_1/1000, clearance_gap_1/1000, pump.Di_1, rough_gap_1/1000000,
                                       pump.Hp_1, True)
    print(f"Расход утечек через переднее уплотнение первой ступени: {q_gap_1 * 3600} м3/ч")