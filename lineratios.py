import numpy as np

class LineRatio(object):
    '''emission-line flux ratio
    '''
    def __init__(self, name, ratio_function, ratio_argnames, reference):
        self.name = name
        self.ratio_function = ratio_function
        self.ratio_argnames = ratio_argnames

        self.reference = reference

    def get_ratio(self, linefluxes):
        return self.ratio_function(*tuple(linefluxes[n] for n in self.ratio_argnames))

    def __repr__(self):
        return f'{self.__class__} ({self.name})'

o3n2 = LineRatio(
    name='O3N2',
    ratio_function=lambda oiii5007, nii6584, ha, hb: \
                   np.log10((oiii5007 / hb) / (nii6584 / ha)),
    ratio_argnames=['[OIII]-5007', '[NII]-6584', 'H-alpha', 'H-beta'],
    reference='http://www.ucolick.org/~xavier/AY230/ay230_HIIdiag.pdf')

class LineRatioSet(object):
    def __init__(self, *ratios):
        self.ratios = ratios

    def __call__(self, linefluxes):
        return {r.name: r(linefluxes) for r in self.ratios}


class RatioCalibration(object):
    '''calibration from a emission-line flux ratio to a metallicity
    '''
    def __init__(self, name, ratio, ratio_to_logoh12, reference):
        self.name = name
        self.ratio = ratio
        self.ratio_to_logoh12 = ratio_to_logoh12
        self.reference = reference

    def get_logOH12(self, ratio):
        return self.ratio_to_logoh12(ratio)

    def __repr__(self):
        return f'{self.__class__} ({self.name}) <- ({self.ratio.name})'

    def __call__(self, linefluxes):
        r = self.ratio.get_ratio(linefluxes)
        logoh12 = self.get_logOH12(r)
        return logoh12

pp04 = RatioCalibration(
    name='PP04', ratio=o3n2,
    ratio_to_logoh12=lambda x: 8.73 - 0.32 * x,
    reference='Pettini & Pagel (2004)')

