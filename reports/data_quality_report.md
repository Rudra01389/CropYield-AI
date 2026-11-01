# Data quality report

## Record counts

- Original file: 345,407 rows (all India, all crops)
- Maharashtra: 17,922 -> Rice: 874 -> Kharif: 646
- Excluded: 4; retained crop records: 642
- Final merged (crop + weather): **642** rows, **34** districts

## Missing values, duplicates, validity (Kharif rice, Maharashtra, before exclusions)

| index | missing |
|---|---|
| district | 0 |
| year | 0 |
| area_ha | 0 |
| production_t | 3 |
| yield_t_ha | 0 |

- Exact duplicate rows: 0; rows sharing district+year: 0
- Non-positive area or negative values: 0
- Units: [{'Area Units': 'Hectare', 'Production Units': 'Tonnes'}]
- Yield vs Production/Area: 0 mismatches in 643 rows with production

## Excluded records

| district | year | area_ha | production_t | reported_yield | reason |
|---|---|---|---|---|---|
| SOLAPUR | 2009-10 | 100.000 |  | 0.000 | Production missing; reported Yield 0.0 is a placeholder, true yield unknown |
| PARBHANI | 2014-15 | 300.000 |  | 0.000 | Production missing; reported Yield 0.0 is a placeholder, true yield unknown |
| YAVATMAL | 2013-14 | 100.000 |  | 0.000 | Production missing; reported Yield 0.0 is a placeholder, true yield unknown |
| KOLHAPUR | 1997-98 | 1100.000 | 246100.000 | 223.727 | Yield > 10.0 t/ha; area inconsistent with the district's other years (see data_quality_report). Not corrected because the true area is unknown |

### Evidence for the implausible-yield exclusion

Reported yield equals Production/Area, so the yield is internally consistent; the problem is the *area* (1,100 ha) which is ~1% of the same district's area in every other year. The true area is not recoverable from the file, so the row is excluded rather than corrected.

| district | Year | area_ha | production_t | yield_t_ha |
|---|---|---|---|---|
| KOLHAPUR | 1997-98 | 1100.000 | 246100.000 | 223.727 |
| KOLHAPUR | 1998-99 | 106000.000 | 289100.000 | 2.727 |
| KOLHAPUR | 1999-00 | 103200.000 | 261900.000 | 2.538 |
| KOLHAPUR | 2000-01 | 106300.000 | 232800.000 | 2.190 |
| KOLHAPUR | 2001-02 | 105900.000 | 289400.000 | 2.733 |
| KOLHAPUR | 2002-03 | 105600.000 | 233200.000 | 2.208 |
| KOLHAPUR | 2003-04 | 106200.000 | 246100.000 | 2.317 |
| KOLHAPUR | 2004-05 | 110500.000 | 279100.000 | 2.526 |
| KOLHAPUR | 2005-06 | 108300.000 | 246100.000 | 2.272 |
| KOLHAPUR | 2006-07 | 109800.000 | 284800.000 | 2.594 |
| KOLHAPUR | 2007-08 | 111900.000 | 336200.000 | 3.004 |
| KOLHAPUR | 2008-09 | 113800.000 | 285200.000 | 2.506 |
| KOLHAPUR | 2009-10 | 112700.000 | 291800.000 | 2.589 |
| KOLHAPUR | 2010-11 | 111500.000 | 321600.000 | 2.884 |
| KOLHAPUR | 2011-12 | 104300.000 | 278900.000 | 2.674 |
| KOLHAPUR | 2012-13 | 107800.000 | 282200.000 | 2.618 |
| KOLHAPUR | 2013-14 | 108800.000 | 339600.000 | 3.121 |
| KOLHAPUR | 2014-15 | 110900.000 | 335200.000 | 3.023 |
| KOLHAPUR | 2015-16 | 111500.000 | 273300.000 | 2.451 |
| KOLHAPUR | 2016-17 | 113800.000 | 396000.000 | 3.480 |
| KOLHAPUR | 2017-18 | 109200.000 | 333300.000 | 3.052 |
| KOLHAPUR | 2018-19 | 106512.000 | 327615.000 | 3.076 |
| KOLHAPUR | 2019-20 | 111359.000 | 270993.000 | 2.434 |

## Flagged but retained

- 97 rows have area < 1000 ha (3 with production 0). Production is rounded to the nearest 100 tonnes in these rows, so yield is coarse (e.g. 0.25 t/ha). They are kept and flagged by `area_lt_1000ha`; decide in Phase 3 whether to down-weight or drop them.
- 14 rows exceed |robust z| > 3.5 within their district (listed below, all retained):

| district | Year | area_ha | yield_t_ha | mad_z |
|---|---|---|---|---|
| NANDED | 2001-02 | 23900.000 | 1.021 | 4.000 |
| SOLAPUR | 2001-02 | 600.000 | 1.333 | 4.370 |
| THANE | 2002-03 | 138000.000 | 1.114 | -3.750 |
| AMRAVATI | 2009-10 | 4300.000 | 1.442 | 4.150 |
| NASHIK | 2009-10 | 48100.000 | 0.391 | -5.740 |
| NASHIK | 2010-11 | 59100.000 | 0.406 | -5.640 |
| NANDED | 2014-15 | 800.000 | 0.125 | -4.040 |
| NANDED | 2017-18 | 700.000 | 1.429 | 7.660 |
| NASHIK | 2016-17 | 71900.000 | 2.028 | 5.360 |
| NASHIK | 2017-18 | 72500.000 | 2.119 | 5.980 |
| SATARA | 1998-99 | 41500.000 | 2.414 | 3.750 |
| SOLAPUR | 1998-99 | 2900.000 | 1.138 | 3.570 |
| SOLAPUR | 1999-00 | 3200.000 | 1.312 | 4.280 |
| SOLAPUR | 2000-01 | 3000.000 | 1.133 | 3.550 |

## District-wise record counts (clean crop data)

| district | records |
|---|---|
| AHMEDNAGAR | 23 |
| AKOLA | 2 |
| AMRAVATI | 23 |
| AURANGABAD | 12 |
| BEED | 23 |
| BHANDARA | 23 |
| BULDHANA | 7 |
| CHANDRAPUR | 23 |
| DHULE | 23 |
| GADCHIROLI | 23 |
| GONDIA | 21 |
| HINGOLI | 13 |
| JALGAON | 13 |
| JALNA | 9 |
| KOLHAPUR | 22 |
| LATUR | 23 |
| NAGPUR | 23 |
| NANDED | 23 |
| NANDURBAR | 21 |
| NASHIK | 23 |
| OSMANABAD | 22 |
| PALGHAR | 6 |
| PARBHANI | 20 |
| PUNE | 23 |
| RAIGAD | 23 |
| RATNAGIRI | 23 |
| SANGLI | 23 |
| SATARA | 23 |
| SINDHUDURG | 23 |
| SOLAPUR | 22 |
| THANE | 23 |
| WARDHA | 11 |
| WASHIM | 12 |
| YAVATMAL | 15 |

## Year-wise record counts

| year | records |
|---|---|
| 1997 | 28 |
| 1998 | 29 |
| 1999 | 31 |
| 2000 | 32 |
| 2001 | 32 |
| 2002 | 32 |
| 2003 | 31 |
| 2004 | 31 |
| 2005 | 30 |
| 2006 | 30 |
| 2007 | 29 |
| 2008 | 27 |
| 2009 | 27 |
| 2010 | 25 |
| 2011 | 28 |
| 2012 | 26 |
| 2013 | 24 |
| 2014 | 24 |
| 2015 | 24 |
| 2016 | 27 |
| 2017 | 24 |
| 2018 | 24 |
| 2019 | 27 |

## Weather coverage

- Daily rows: 298,044; districts with weather: 34
- Missing daily values: {'PRECTOTCORR': 0, 'T2M': 0, 'RH2M': 0}
- Distinct weather series: 30 for 34 districts. Districts sharing one NASA grid cell (identical weather): [['HINGOLI', 'PARBHANI'], ['AKOLA', 'AMRAVATI'], ['NAGPUR', 'WARDHA'], ['BHANDARA', 'GONDIA']]
- Crop rows without a weather row: 0; with incomplete weather: 0

## Final dataset summary

| index | count | mean | std | min | 25% | 50% | 75% | max |
|---|---|---|---|---|---|---|---|---|
| yield_t_ha | 642.000 | 1.265 | 0.840 | 0.000 | 0.525 | 1.114 | 1.844 | 3.480 |
| area_ha | 642.000 | 52979.802 | 58357.229 | 5.000 | 4550.000 | 21750.000 | 79200.000 | 311400.000 |
| rain_kharif_mm | 642.000 | 1007.002 | 574.218 | 265.000 | 623.850 | 833.350 | 1156.950 | 3405.600 |
| temp_kharif_c | 642.000 | 26.387 | 1.316 | 23.460 | 25.740 | 26.610 | 27.310 | 29.080 |
| rh_kharif_pct | 642.000 | 79.870 | 5.424 | 66.870 | 75.850 | 78.985 | 84.335 | 91.610 |

## Assumptions and limitations

- **Year mapping**: crop year `YYYY-YY` is matched to calendar year YYYY (Kharif is sown in June-July and harvested Oct-Dec of that year). The CSV does not state this; it is the standard Indian agricultural-year convention.
- **Weather window**: June-September only. Rice harvested in Oct-Nov is partly influenced by October weather, which is not included.
- **Weather source**: NASA POWER / MERRA-2 reanalysis (0.5 x 0.625 deg), not rain-gauge data. One representative point per district (centre of the OpenStreetMap district boundary) cannot capture within-district variation, e.g. the Konkan coast vs the Sahyadri ranges.
- **Coordinate source**: OpenStreetMap Nominatim (ODbL); see `data/raw/district_coordinates.csv`.
- **District changes**: Palghar (2014) was split from Thane, Gondia from Bhandara (1999), etc. Early-year records of parent districts cover the old boundaries while the coordinates are for the current boundaries.
- **No soil, irrigation, fertiliser or variety data**; area is the only management feature.
- Rice here is the aggregate of all Kharif rice; yield differences between districts include rainfed vs irrigated mix.
- Plot-level statistics (EDA figures) use all rows for description only; no scaling or splitting was done. Phase 3 must split chronologically and fit scalers/encoders on training data only.
