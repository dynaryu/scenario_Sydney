from scipy import stats
import numpy as np
import pandas as pd

def compute_vulnerability(mmi, bldg_class):

    def inv_logit(x):
        return  np.exp(x)/(1.0 + np.exp(x)) 

    def compute_mu(mmi, **kwargs):

        coef = {
            "t0":-8.56, 
            "t1": 0.92, 
            "t2": -4.82, 
            "t3": 2.74, 
            "t4": 0.49, 
            "t5": -0.31}

        flag_timber = kwargs['flag_timber']
        flag_pre = kwargs['flag_pre']           

        mu = coef["t0"] +\
            coef["t1"]*mmi +\
            coef["t2"]*flag_timber +\
            coef["t3"]*flag_pre +\
            coef["t4"]*flag_timber*mmi +\
            coef["t5"]*flag_pre*mmi
        return mu

    flag_timber = 'Timber' in bldg_class
    flag_pre = 'Pre' in bldg_class

    # correction of vulnerability suggested by Mark
    if mmi < 5.5:
        prob55 = inv_logit(compute_mu(5.5, 
            flag_timber=flag_timber, 
            flag_pre=flag_pre))
        return(np.interp(mmi, [4.0, 5.5], [0.0, prob55], left=0.0))
    else:
        mu = compute_mu(mmi, 
            flag_timber=flag_timber, 
            flag_pre=flag_pre)
        return(inv_logit(mu))

def compute_gamma(bldg_class, cov=1.0, nsample=1000, percent=[16, 84]):

	"""
	The probability density function for `gamma` is::

	    gamma.pdf(x, a) = lambda**a * x**(a-1) * exp(-lambda*x) / gamma(a)

	for ``x >= 0``, ``a > 0``. Here ``gamma(a)`` refers to the gamma function.

	The scale parameter is equal to ``scale = 1.0 / lambda``.

	`gamma` has a shape parameter `a` which needs to be set explicitly. For
	instance:

	    >>> from scipy.stats import gamma
	    >>> rv = gamma(3., loc = 0., scale = 2.)

	shape: a 
	scale: b
	mean = a*b
	var = a*b*b
	cov = 1/sqrt(a) = 1/sqrt(shape)
	shape = (1/cov)^2
	"""

# gamma has cov of sqrt(shape)

	# for URM Pre1945
	#cov = 1.0
	#str_ = 'URM_Pre1945'

	shape_ = np.power(1/cov, 2)
	mmi_range = np.arange(4.0, 7.5, 0.2)
	#nsample = 100

	vul = np.zeros((len(mmi_range)))
	temp = np.zeros((nsample, len(mmi_range)))
	for i, mmi in enumerate(mmi_range):
	    vul[i] = compute_vulnerability(mmi, bldg_class)
	    try:
	    	scale_ = vul[i]/shape_
	    	temp[:, i] = stats.gamma.rvs(shape_, loc = 0., scale = scale_, size=nsample)
	    except ZeroDivisionError:
	    	pass

	temp[temp > 1] = 1.0
	med = np.percentile(temp, q=50.0, axis=0)
	mean_ = np.mean(temp, axis=0)
	p1 = np.percentile(temp, q=percent[0], axis=0)
	p2 = np.percentile(temp, q=percent[1], axis=0)

	df = pd.DataFrame({'med': med, 'mean': mean_, 'percent1': p1, 
		'percent2': p2, 'vuln': vul}, index=mmi_range)

	df2 = pd.DataFrame(temp, columns=mmi_range)

	return (df, df2)