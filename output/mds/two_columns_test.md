# Extracted Content from two_columns_test.pdf

## Why (and How) LGADs Work: Ionization, Space Charge, and Gain Saturation

N. Cartigliaa,∗, A. R. Altamurad,a, R. Arcidiaconoa,b, M. Durandod, S. Gallettod, M. Ferreroa, L. Lanterid,a, A.
Losanad,a, L. Massaccesid,a, L. Menzioc, F. Sivieroa, V. Solad,a, R. Whited,a

aINFN sezione di Torino, Torino, Italy
bUniversit`a del Piemonte Orientale, Novara, Italy
cCERN, Geneva, Switzerland
dUniversit`a degli Studi di Torino, Torino, Italy

## Abstract

The temporal resolution of Low-Gain Avalanche Detectors (LGADs), also known as Ultra-Fast Silicon Detectors
(UFSDs), is governed by two contributions: jitter, arising from electronic noise and signal slew rate, and the Landau
noise term, arising from the non-uniform energy deposition of minimum ionizing particles (MIPs). We show that a
correct simulation of the initial ionization alone significantly overestimates the measured Landau noise. Two addi-
tional physical mechanisms are necessary to reproduce the data: space charge effects during electron/hole drift, which
smooth the granularity of the initial charge distribution, and gain saturation during multiplication, which preferentially
suppresses large-amplitude fluctuations. All steps of the model have been implemented in the fast simulation program
Weightfield2 (WF2). The model is validated against several independent experimental observations: the evolution
of the measured charge distribution with gain, the temporal resolution of events in the Landau tail, and the thickness
dependence of timing performance. We also discuss a data-driven gain measurement method based on gain saturation,
and implications for gain layer design.

Keywords: LGAD; UFSD; temporal resolution; Landau noise; gain saturation; space charge; Weightfield2; silicon
detectors

## 1. Introduction

## 1. Introduction

program
[7],
a
fast,
publicly
available
sim-
ulation
tool
for
silicon
sensors
available
at
https://www.to.infn.it/~cartigli/Weightfield2/.

Low-Gain Avalanche Detectors (LGADs) [1, 2] are
silicon sensors that exploit a moderate internal gain
(∼10–40) to achieve excellent single-layer temporal
resolution, routinely below 50 ps [4].
Their timing
performance is central to several detector upgrades at
the High-Luminosity LHC, including the CMS Endcap
Timing Layer (ETL) and the ATLAS High-Granularity
Timing Detector (HGTD) [5, 6].

The paper is organized as follows.
Section 2 de-
scribes the MIP ionization model, including the seed
distribution and long-range correlations from delta rays.
Section 3 introduces the two components of temporal
resolution and provides an analytical derivation of the
Landau noise term. Section 4 presents the comparison
between WF2 predictions and data, identifying the dis-
crepancy that motivates two additional physical mech-
anisms. Section 5 describes the two smoothing mecha-
nisms — space charge effects during drift and gain sat-
uration during multiplication — and introduces the phe-
nomenological parameter α. Section 6 presents experi-
mental evidence used to determine α. Section 7 shows
that, with α fixed, WF2 predictions agree with mea-
sured temporal resolution across all sensor thicknesses.
Section 8 summarises why LGADs achieve good tim-
ing through the interplay of these three mechanisms.
Section 9 introduces a novel data-driven gain measure-
ment method.
Section 10 discusses implications for

Despite their widespread use, a complete quantitative
understanding of why LGADs achieve such good timing
is still being developed. This work addresses that ques-
tion from first principles, tracing the signal formation
chain from the initial ionization through drift and multi-
plication, and identifying the physical mechanisms that
govern the intrinsic temporal resolution.

All
simulation
results
presented
in
this
pa-
per were obtained with the Weightfield2 (WF2)

March 12, 2026


--- End of Page 1 ---

gain layer design. Section 11 summarizes the conclu-
sions.

![Image associated with caption: Figure 1: Measured (red) and WF2-simulated Landau energy distribu-
tions (blue). The simulation correctly reproduces both the MPV and
the FWHM as a function of sensor thickness. (1) (2) (3) (4)](..\images\two_columns_test\page_2_figure_1.jpg)

*Figure 1: Measured (red) and WF2-simulated Landau energy distribu-
tions (blue). The simulation correctly reproduces both the MPV and
the FWHM as a function of sensor thickness. (1) (2) (3) (4)*

## 2. MIP Ionization in Silicon Sensors

## 2. MIP Ionization in Silicon Sensors

## 2.1. The Seed Distribution Model

The energy deposition of a MIP in a silicon sensor
of arbitrary thickness is modeled in WF2 as a sum of
local deposits drawn independently from a seed distri-
bution — the energy deposited in a 1-µm thick silicon
layer. This seed distribution has two physically distinct
components:

• A soft scattering component, corresponding to
small, frequent energy transfers.
Because the
cross-section diverges at low T, the vast majority
of interactions transfer very little energy. In WF2,
these are modeled as a Gaussian contribution with
mean µsoft(d) and standard deviation σsoft per mi-
cron slice.
• A hard scattering component, corresponding to
more energetic collisions sampled from the 1/T 2
distribution over [Tmin, Tmax], that produce δ-rays
— knock-on electrons carrying significant kinetic
energy away from the primary track. The value of
Tmin sets the width of this energy range and hence
the normalisation of the distribution. A δ-ray with
T ∼30–100 keV can travel tens of microns from
the primary track. These events populate the high-
energy Landau tail.

• A soft scattering component, corresponding to
small, frequent energy transfers.
Because the
cross-section diverges at low T, the vast majority
of interactions transfer very little energy. In WF2,
these are modeled as a Gaussian contribution with
mean µsoft(d) and standard deviation σsoft per mi-
cron slice.

This thickness dependence reflects the fact that the rel-
ative weight of the hard and soft components in the en-
ergy deposition changes systematically with thickness.
The scaling functions were determined by fitting the
measured MPV and FWHM of the Landau distribution
across sensor thicknesses from 5 to 300 µm. The to-
tal energy deposition in a sensor of thickness d is ob-
tained by drawing a sequence of d/1 µm random sam-
ples from the seed distribution and summing them. Us-
ing this approach, WF2 correctly reproduces the mea-
sured MPV [8] and FWHM of the Landau distribution
as a function of sensor thickness (Figure 1), confirming
the validity of the ionization model.

• A hard scattering component, corresponding to
more energetic collisions sampled from the 1/T 2
distribution over [Tmin, Tmax], that produce δ-rays
— knock-on electrons carrying significant kinetic
energy away from the primary track. The value of
Tmin sets the width of this energy range and hence
the normalisation of the distribution. A δ-ray with
T ∼30–100 keV can travel tens of microns from
the primary track. These events populate the high-
energy Landau tail.

## 2.2. Long-Range Correlations from Delta Rays

In WF2, each δ-ray of energy T is assigned a range
using the empirical CSDA relation for electrons in sili-
con [10]:

Concretely, in each 1-µm slice the number of hard
collisions N is drawn from a Poisson distribution with
mean ν · ∆x, where ν is the mean collision rate per mi-
cron. Each collision energy is sampled from the trun-
cated 1/T 2 distribution using the exact inverse-CDF
method:

![isolate_formula](..\images\two_columns_test\page_2_isolate_formula_14.jpg)

δ-rays with R < 2 µm deposit their energy locally at
the point of creation.
For R ≥2 µm, the energy is
distributed along the track using a power-law stopping
profile derived by inverting Eq. (3) (R ∝T 1.75 implies
T ∝R1/1.75):

![isolate_formula](..\images\two_columns_test\page_2_isolate_formula_17.jpg)

with Tmin = 26 eV and Tmax = 600 keV. The results are
insensitive to Tmax for values above ∼200 keV, as the
1/T 2 spectrum strongly suppresses large energy trans-
fers. The soft component is added as an independent
Gaussian draw (µsoft, σsoft = 10 eV) per micron. The
total slice energy (hard + soft) is converted to electron-
hole pairs using the mean ionisation energy in silicon,
ϵ = 3.6 eV/pair. Both ν and µsoft depend on sensor thick-
ness following empirical scaling relations:

![isolate_formula](..\images\two_columns_test\page_2_isolate_formula_20.jpg)

so that the energy deposited in each 1-µm slice is the dif-
ference in remaining energy between its start and end.
This reflects the physical behaviour of electrons, whose
energy loss per unit length increases as they slow down,
so that the deposited energy per slice rises toward the
end of the range. If the range of a δ-ray exceeds the re-
maining distance to the sensor edge, energy deposition

![isolate_formula](..\images\two_columns_test\page_2_isolate_formula_23.jpg)


--- End of Page 2 ---

is truncated at the boundary and the remaining energy
escapes the sensor undetected.

![Image associated with caption: Figure 2: Measured values of the Landau noise as a function of sensor
thickness and a line that shows the ∝
√
d behaviour. (8) (7) (6) (5)](..\images\two_columns_test\page_3_figure_1.jpg)

*Figure 2: Measured values of the Landau noise as a function of sensor
thickness and a line that shows the ∝
√
d behaviour. (8) (7) (6) (5)*

## 3. Components of the Temporal Resolution

## 3.1. Jitter and Landau Noise

The temporal resolution σt of an LGAD is decom-
posed into two independent contributions [3]:

![isolate_formula](..\images\two_columns_test\page_3_isolate_formula_6.jpg)

where n is the linear density of electron-hole pairs (pairs
per unit length). For the MIP case, the charge depo-
sition is non-uniform and the analytical treatment be-
comes significantly more complex; the full derivation is
given in [9].

Jitter arises from electronic noise σV and the signal
slew rate dV/dt at threshold crossing:

![isolate_formula](..\images\two_columns_test\page_3_isolate_formula_10.jpg)

Equation (8) reveals the fundamental structure of the
Landau noise: it grows with the square root of the thick-
ness and decreases with drift velocity.
Operating at
fields sufficient to saturate the drift velocity v, and re-
ducing sensor thickness, are the most effective ways to
reduce this term. Specifically, for the MIP case, the N
charge carriers are not uniformly distributed but follow
the non-uniform Landau deposition. The resulting Lan-
dau noise is larger than for a laser, and its precise value
depends on the specific ionization pattern in each event
[9]. Figure 2 shows that the theoretical
√
d scaling well
matches the measured values of Landau noise [11].

It dominates at low gain, where the signal amplitude is
small relative to the noise.

Landau noise, also called the non-uniform ionization
term, arises from the event-to-event variability of the
charge deposition. Each MIP deposits charge in a dif-
ferent spatial pattern along the track, producing a differ-
ent signal shape and hence a different trigger time. This
term dominates at high gain, where jitter is negligible.

Throughout this work, the trigger time is defined by
a Constant Fraction Discriminator (CFD) at 30% of the
signal amplitude, both in the experimental data and in
the WF2 simulation.

## 3.2. Analytical Derivation of the Landau Noise Term

## 4. The Discrepancy Between Simulation and Data

Following Riegler [9], we derive the general features
of the Landau noise term. Consider a sensor of thick-
ness d in which a single electron-hole pair is created at
a random, uniformly distributed position x ∈[0, d]. The
electron drifts to the collecting electrode with velocity
v, arriving at time t = x/v. Since x is uniform on [0, d],
the standard deviation of the arrival time is:

Figure 3 compares the measured temporal resolution
(jitter subtracted, Landau noise contribution only) with
the WF2 predictions obtained using the Landau ion-
ization model described above. Two key observations
emerge:

1. The simulated Landau noise is considerably larger
than the measured one across all sensor thick-
nesses.
2. The discrepancy grows with increasing sensor
thickness.

1. The simulated Landau noise is considerably larger
than the measured one across all sensor thick-
nesses.

![isolate_formula](..\images\two_columns_test\page_3_isolate_formula_22.jpg)

This is the fundamental upper bound on the Landau
noise: it corresponds to total ignorance of the charge
creation position. For a laser that deposits N electron-
hole pairs uniformly along the track, the uncertainty on
the trigger time improves as 1/
√
N (by analogy with the
mean of N uniform random variables).

2. The discrepancy grows with increasing sensor
thickness.

This discrepancy implies that the actual LGAD sig-
nal shape is smoother than the ionization model im-
plemented in WF2. Two physical mechanisms, not in-
cluded in the simulation, are responsible for this dis-
crepancy: space charge effects during drift and gain sat-
uration during multiplication.

![isolate_formula](..\images\two_columns_test\page_3_isolate_formula_27.jpg)


--- End of Page 3 ---

