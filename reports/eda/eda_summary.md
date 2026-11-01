# EDA summary

Rows: 642, districts: 34, years 1997-2019

## Descriptive statistics

```
                count       mean        std      min       25%        50%        75%        max
year            642.0   2007.469      6.591  1997.00  2002.000   2007.000   2013.000    2019.00
area_ha         642.0  52979.802  58357.229     5.00  4550.000  21750.000  79200.000  311400.00
rain_kharif_mm  642.0   1007.002    574.218   265.00   623.850    833.350   1156.950    3405.60
temp_kharif_c   642.0     26.387      1.316    23.46    25.740     26.610     27.310      29.08
rh_kharif_pct   642.0     79.870      5.424    66.87    75.850     78.985     84.335      91.61
yield_t_ha      642.0      1.265      0.840     0.00     0.525      1.114      1.844       3.48
```

## Correlation with yield (all rows)
```
year              0.161
area_ha           0.514
rain_kharif_mm    0.670
temp_kharif_c    -0.407
rh_kharif_pct     0.794
```

## Correlation with yield after removing district means
```
year              0.180
area_ha           0.067
rain_kharif_mm    0.177
temp_kharif_c    -0.077
rh_kharif_pct     0.150
```
