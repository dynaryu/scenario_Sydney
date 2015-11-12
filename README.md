# EQ scenario impact assessment to the Greater Sydney

This excercise is carried out for DRR (Disaster Risk Reduction) day at GA.

## Building exposure data
### From NSW to the Greater Sydney
Original data was provided by NEXIS team, which contains residential buildings by state. In order to extract buildings in the Greater Sydney area, I've extracted buildings by SA1 (Statistical Area Level 1 defined by ABS) using the shapefile ('data/GIS/aust_polygon/GCCSA_2011_AUST.shp'). Initially I was trying to use shapely and pyshp, which turned out to be a very tedious job, so I switched to ArcGIS and extracted the SA1 within the boundary.


### Building type distribution
The building type distribution from the NEXIS seems a little strage especially for older suburbs, so decided to apply the building distribution extracted from the Alexandria Canal survey using the python script.

```python
analysis_Alexandria_data.py
```
The older suburbs to which we apply the extracted building distribution are selected manually.

###


## EQRM run

### input file
Two input files are created: sydney_par_site.csv and sydney_soil_par_site.csv.
The sydney_par_site.csv is a grid covering the Greater Sydney area, which was created using the script 
