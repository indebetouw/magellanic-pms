On May 17, 2021, at 12:28 PM, GALLIANO Frederic <frederic.galliano@cea.fr> wrote:

Salut Remy,

I attach the Av maps. I have actually two versions: one at the SPIRE 500 resolution; and one at 14" (gaussian beam) resolution (up to PACS 160 wavelength). For each version I give you the FITS file of the natural logarithm of Av (thus Av=exp(map)). The file _mean is the actual Av value and _sigma is the 1-sigma uncertainty of ln(Av). Uncertainties are indeed usually more symmetric in log-space.

This modelling was done using HerBIE (Galliano, 2018), with the THEMIS grain properties (Jones et al., 2017). I attach a plot comparing the estimates at both resolutions (I don't remember why the x-axis says "least-squares", but the SPIRE500 map I send you is a Bayesian map). The lesser resolution is probably more conservative.

Fred
