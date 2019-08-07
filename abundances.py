class AbundanceSet(object):
    def __init__(self, solar_Z, solar_logOH12):
        self.solar_Z = solar_Z
        self.solar_logOH12 = solar_logOH12

    def logZ_to_logOH12(self, logZ):
        return self.solar_logOH12 + logZ

    def Z_to_logOH12(self, Z):
        return np.log10(Z / self.solar_Z) + self.solar_logOH12

    def logOH12_to_Z(self, logOH12):
        return self.solar_Z * 10.**(logOH12 - self.solar_logOH12)

    def __repr__(self):
        return f'{self.__class__} (Zsol = {self.solar_Z}, logOH12sol = {self.solar_logOH12})'

default = AbundanceSet(solar_Z=.0142, solar_logOH12=8.69)

pp04 = AbundanceSet(solar_Z=.0126, solar_logOH12=8.66) 
dopita16 = AbundanceSet(solar_Z=.014, solar_logOH12=8.77)
kd04 = AbundanceSet(solar_Z=.02, solar_logOH12=8.9)  # Anders & Grevesse 1989
maiolino08 = AbundanceSet(solar_Z=.0142, solar_logOH12=8.69)  # Allende Prieto et al. 2001