'''
1. compute loss ratio by suburb
2. compute no. of bldgs in each damage states by bldg type and by suburb
3. estimate fatality based on HAZUS methodology

assumptions 
There are four levels of injury severity: (Table 13.1)
1. Population distribution (from pop. by suburb, or floor area proportional)
2. Indoor casualty only
3. map OZ bldg type to HAZUS bldg type
W1MEAN, W1BVTILE, W1BVMETAL, W1TIMBERTILE, W1TIMBERMETAL => W1
C1LMEAN, C1LSOFT, C1LNOSOFT => C1L
C1MMEAN, C1MSOFT, C1MNOSOFT => C1M
C1HMEAN, C1HSOFT, C1HNOSOFT => C1H
URMLMEAN, URMLTILE, URMLMETAL => URML
URMMMEAN, URMMTILE, URMMMETAL => URMM

usage: python summary_eqrm.py site_tag ncpu input_dir output_dir csv_dir

site_tag = 'Karratha'
ncpus = 4
input_dir = '/nas/gemd/e3p/sandpits/hryu/WA_FESA/Karratha/input'
output_dir = '/nas/gemd/e3p/sandpits/hryu/WA_FESA/Karratha/nci/scen_risk_mw5.09'
csv_dir = '/nas/users/u65242/unix/scratch/WA_FESA/exposure'

'''

import sys
sys.path.append('/nas/users/u65242/unix/scratch/xlrd-0.9.0')
from os.path import join
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.stats as stats
import csv
import xlrd
import subprocess
from scipy.interpolate import interp1d

site_tag = sys.argv[1]
input_dir = sys.argv[2]
output_dir = sys.argv[3]
if len(sys.argv) < 5:
	csv_dir = '/nas/users/u65242/unix/scratch/WA_FESA/exposure'
else:
	csv_dir = sys.argv[4]

def read_loss_txt(dump):
	x = []
	for i in range(2,len(dump)): # skip the first two lines
		temp = [float(val) for val in dump[i].strip('\n').split(' ')]
		x.append(temp)
	x = np.array(x).T # nbldgs, nevents
	return x

def read_indoor_casualty_table(csv_file,ds_str):
	dump = open(csv_file, 'rb').readlines()[1:]
	for line in dump:
		temp = line.strip('\n').split(',')
		fat_rate.setdefault(temp[1],{})[ds_str] = [0.01*float(x) for x in temp[2:]]
	return fat_rate

def compute_loss_ratio(bldg_loss_x, total_bldg_loss_x, total_repl_cost_x):
	# compute ratio
	sum_total_repl_cost = total_repl_cost_x.sum(axis=0)
	ratio = bldg_loss_x.sum(axis=0)/sum_total_repl_cost[0]
	ratio_incl_cont = total_bldg_loss_x.sum(axis=0)/sum_total_repl_cost[1]
	return (ratio, ratio_incl_cont)

# read inventory
data = open(join(input_dir,'sitedb_' + site_tag + '.csv')).readlines()
hdr = data[0].split(',')

id0 = hdr.index('CONTENTS_COST_DENSITY')
id1 = hdr.index('BUILDING_COST_DENSITY')
id2 = hdr.index('FLOOR_AREA')
id3 = hdr.index('SURVEY_FACTOR')
id4 = hdr.index('STRUCTURE_CLASSIFICATION')
id5 = hdr.index('HAZUS_USAGE')
id6 = hdr.index('SUBURB')

# read ci 
a = subprocess.check_output(['grep', '-H', '-r', 'cost', output_dir + '/eqrm_flags.py'])
idx = a.index('=')
ci = float(a[idx+1:])

# read Mw
a = subprocess.check_output(['grep', '-H', '-r', 'magnitude', output_dir + '/eqrm_flags.py'])
idx = a.index('=')
mw = float(a[idx+1:])

# total replacement cost = ci*survey_factor*building_cost_density*floor_area
#+ ci*survey_factor*contents_cost_density*floor_area
# bval.txt == total_replacement_cost including contents

nbldgs = len(data)-1
total_repl_cost = np.zeros((nbldgs,2))
bldg_type, occ_type, suburb = [], [], []
floor_area = []
for i in range(nbldgs):
	temp = data[i+1].split(',')
	bldg_type.append(temp[id4].strip(' '))
	occ_type.append(temp[id5].strip(' '))
	suburb.append(temp[id6])
	bldg_cost_dens = float(temp[id1])
	cont_cost_dens = float(temp[id0])
	floor = float(temp[id2])*float(temp[id3])
	floor_area.append(floor)
	bldg_only = ci*bldg_cost_dens*floor
	cont_only = ci*cont_cost_dens*floor
	total_repl_cost[i,0] = bldg_only
	total_repl_cost[i,1] = bldg_only + cont_only

bldg_type = np.array(bldg_type)
suburb = np.array(suburb)
floor_area = np.array(floor_area)
unq_bldg = np.unique(bldg_type)
unq_sub = np.unique(suburb)
del data

# building loss
bldg_loss_dump = open(join(output_dir,site_tag + '_building_loss.txt')).readlines()
bldg_loss = read_loss_txt(bldg_loss_dump) # nbldgs, nevents
del bldg_loss_dump

# content loss
cont_loss_dump = open(join(output_dir,site_tag + '_contents_loss.txt')).readlines()
cont_loss = read_loss_txt(cont_loss_dump) # nbldgs, nevents
del cont_loss_dump
#cont_loss + bldg_loss = total_bldg_loss (verified)

# total building loss
total_bldg_loss_dump = open(join(output_dir,site_tag + '_total_building_loss.txt')).readlines()
total_bldg_loss = read_loss_txt(total_bldg_loss_dump) # nbldgs, nevents
del total_bldg_loss_dump

# compute loss ratio by suburb
fid = open(join(output_dir,'summary_loss_ratio_by_suburb_' + site_tag + '_Mw'+ str(mw) + '.csv'),'wb')
file_writer = csv.writer(fid)
file_writer.writerow(['suburb', 'loss ratio', 'loss ratio incl. contents'])

for sub in unq_sub:
	tf = (suburb == sub)
	(ratio, ratio_incl_cont) = compute_loss_ratio(bldg_loss[tf,:], total_bldg_loss[tf,:], total_repl_cost[tf,:])
	tmp = []
	tmp.append(sub)
	tmp.append(ratio.mean())
	tmp.append(ratio_incl_cont.mean())
	file_writer.writerow(tmp)

# all
(ratio, ratio_incl_cont) = compute_loss_ratio(bldg_loss, total_bldg_loss, total_repl_cost)
tmp=[]
tmp.append('all')
tmp.append(ratio.mean())
tmp.append(ratio_incl_cont.mean())
file_writer.writerow(tmp)
fid.close()

# plot histogram of loss ratio
plt.figure(); plt.hist(ratio_incl_cont)
plt.xlabel('Loss ratio inclding contents')
plt.ylabel('Frequency')
plt.title(site_tag+', Mw'+str(mw))
plt.grid(1)
plt.savefig(join(output_dir,site_tag + '_Mw'+ str(mw) + '_incl.png'), format='png')

plt.figure(); plt.hist(ratio)
plt.xlabel('Loss ratio excluding contents')
plt.ylabel('Frequency')
plt.title(site_tag+', Mw'+str(mw))
plt.grid(1)
plt.savefig(join(output_dir,site_tag + '_Mw'+ str(mw) + '_excl.png'), format='png')

# write bldg type distribution by suburb
fid = open(join(output_dir,'summary_bldg_type_by_suburb_' + site_tag + '_Mw'+ str(mw) + '.csv'),'wb')
file_writer = csv.writer(fid)
file_writer.writerow(['suburb', 'bldg type', 'frequency'])
for sub in unq_sub:
	for bldg in unq_bldg:
		tf = ((suburb == sub) & (bldg_type == bldg))
		if tf.sum() > 0:
			tmp = []
			tmp.append(sub)
			tmp.append(bldg)
			tmp.append(tf.sum())
			file_writer.writerow(tmp)

###############################################################################
# compute no. of bldgs in each damage state by bldg. type (by suburb)
###############################################################################

# read damage probability
pb = np.load(join(output_dir,'pb_str.npy'))
(nbldgs, nevents, nds) = pb.shape

# convert to pe
pe = np.zeros(pb.shape)
for i in range(nds):
	pe[:,:,i] = pb[:,:,i:].sum(axis=2)

# determine damage state of bldg.
rv = np.dot(np.random.rand(nbldgs,nevents,1),np.ones((1,nds))) # nbldgs, nevents, nds
ds_bldgs_rnd = (rv < pe).sum(axis=2)
ds_bldgs_rnd = np.array(ds_bldgs_rnd) # nbldgs, nevents

# no. of ds by bldg type
no_ds_bldgs_by_bldg = {}
for bldg in unq_bldg:
	tf = (bldg_type == bldg)
	ds_bldgs = ds_bldgs_rnd[tf,:] # nbldgs, nevents
	no_ds_bldgs = []
	# ds <=> (0,1,2,3,4)
	for i in range(5):
		no_ds_bldgs.append((ds_bldgs == i).sum(axis=0))
	no_ds_bldgs = np.array(no_ds_bldgs) # 5, nevents
	no_ds_bldgs_by_bldg[bldg] = no_ds_bldgs.mean(axis=1)

# output file
fid = open(join(output_dir,'summary_no_bldgs_ds_by_bldg_type_' + site_tag + '_Mw'+ str(mw) + '.csv'),'wb')
file_writer = csv.writer(fid)
file_writer.writerow(['Bldg type', 'No', 'DS0', 'DS1', 'DS2', 'DS3', 'DS4'])
for bldg in unq_bldg:
	tmp = []
	tmp.append(bldg)
	tmp.append(sum(bldg_type == bldg))
	[tmp.append(str(x)) for x in no_ds_bldgs_by_bldg[bldg]]
	file_writer.writerow(tmp)

for bldg in unq_bldg:
	tmp = []
	tmp.append(bldg)
	no_bldgs = sum(bldg_type == bldg)
	tmp.append(no_bldgs)
	[tmp.append(str(x/no_bldgs*100.0)) for x in no_ds_bldgs_by_bldg[bldg]]
	file_writer.writerow(tmp)
fid.close()

# percentage of bldgs in each damage state by suburb
no_ds_bldgs_by_sub = {}
for sub in unq_sub:
	tf = (suburb == sub)
	ds_bldgs = ds_bldgs_rnd[tf,:] # nbldgs, nevents
	no_ds_bldgs = []
	# ds <=> (0,1,2,3,4)
	for i in range(5):
		no_ds_bldgs.append((ds_bldgs == i).sum(axis=0))
	no_ds_bldgs = np.array(no_ds_bldgs) # 5, nevents
	no_ds_bldgs_by_sub[sub] = no_ds_bldgs.mean(axis=1)

# output file
fid = open(join(output_dir,'summary_no_bldgs_ds_by_suburb_' + site_tag + '_Mw'+ str(mw) + '.csv'),'wb')
file_writer = csv.writer(fid)
file_writer.writerow(['Suburb', 'No', 'DS0(%)', 'DS1(%)', 'DS2(%)', 'DS3(%)', 'DS4(%)'])
for sub in unq_sub:
	tmp = []
	tmp.append(sub)
	no_bldgs = sum(suburb == sub)
	tmp.append(no_bldgs)
	[tmp.append(str(x/no_bldgs*100.0)) for x in no_ds_bldgs_by_sub[sub]]
	file_writer.writerow(tmp)
fid.close()
	
#print no_ds_bldgs_med
#print 'one event: ', no_ds_bldgs_rnd[0,:]
#print 'average: ', no_ds_bldgs_rnd.mean(axis=0)
#print 'average pct: ', no_ds_bldgs_rnd.mean(axis=0)/float(nbldgs)*100.0
#print no_ds_bldgs_by_bldg


###############################################################################
# Fatality estimation
# 1. HAZUS methodology
###############################################################################

# read indoor casualty (table13.3 through 13.7)
fat_rate={}
list_ds = ['slight','moderate','extensive','complete','collapse']
for ids, ds in enumerate(list_ds):
	fname = join(csv_dir,'table13.' + str(ids+3) + '.csv')
	read_indoor_casualty_table(fname,ds)
nlevels = 4 # no. of severity levels

# read collapse rate (table 13.8)
dump = open(join(csv_dir,'table13.8.csv'), 'rb').readlines()[1:]
for line in dump:
	temp = line.strip('\n').split(',')
	fat_rate.setdefault(temp[1],{})['rate'] = 0.01*float(temp[-1])

# estimate population (proportional to floor area)
wb = xlrd.open_workbook(csv_dir + '/population_by_suburb.xls')
sh = wb.sheet_by_name(site_tag)

pop = {}
for i in range(1,sh.nrows): # skipping head row
	tmp = sh.row_values(i) # suburb, pop
	pop[tmp[0]] = float(tmp[1])

# distribute population proportional to floor area
pop_est = np.zeros(bldg_type.shape)
for sub in unq_sub:
	tf = (suburb == sub)
	pop_density = pop[sub]/floor_area[tf].sum()
	pop_est[tf] = floor_area[tf]*pop_density

pb_x = np.zeros((nbldgs, nevents, nds+1)) # include collapse
for bldg in unq_bldg:
	tf = (bldg_type == bldg)
	pb_sel = pb[tf,:,:]
	pb_x[tf,:,:-1] = pb_sel[:]
	pc = fat_rate[bldg]['rate']
	a = (1.0-pc)*pb_sel[:,:,-1] # complete without collapse
	b = pc*pb_sel[:,:,-1] # complete with collapse
	pb_x[tf,:,-2] = a[:]
	pb_x[tf,:,-1] = b[:]

# estimate no. of fatality by suburb
# Nfat = (Rate|DS) * Nocc
# HAZUS methodology

no_fat = np.zeros((nbldgs,nevents,nlevels)) # nbldgs, nevents, nlevels
for bldg in unq_bldg:
	tf = (bldg_type == bldg)
	pb_x_sel = pb_x[tf,:,:]
	pop_est_sel = pop_est[tf]
	for inj in range(nlevels):
		x_inj = np.zeros((sum(tf),nevents))
		for ids, ds in enumerate(list_ds):
			x_inj += fat_rate[bldg][ds][inj]*pb_x_sel[:,:,ids]*pop_est_sel[:,np.newaxis]
		no_fat[tf,:,inj]=x_inj

# fatality for portfolio
fid = open(join(output_dir,'summary_casualty_' + site_tag + '_Mw'+ str(mw) + '.csv'),'wb')
file_writer = csv.writer(fid)
file_writer.writerow(['suburb', 'population', 'severity1', 'severity2','severity3','severity4'])

# compute fatality by suburb
for sub in unq_sub:
	tf = (suburb == sub)
	no_fat_sub = no_fat[tf,:,:].sum(axis=0).mean(axis=0)
	tmp = []
	tmp.append(sub)
	tmp.append(int(round(sum(pop_est[tf]))))
	[tmp.append(int(round(x))) for x in no_fat_sub]
	file_writer.writerow(tmp)
fid.close()

# histogram
plt.figure()
for i in range(nlevels):
	plt.subplot(2,2,i+1)
	plt.hist(no_fat[:,:,i].sum(axis=0))
	plt.title('Injury level:' + str(i+1))
	plt.grid(1)
plt.savefig(join(output_dir,'histogram_fatality_' + site_tag + '_Mw' + str(mw) + '.png'),format='png')

###############################################################################
# 2. MC simulation
###############################################################################
# convert to pe
pe_x = np.zeros(pb_x.shape)
for i in range(nds+1):
	pe_x[:,:,i] = pb_x[:,:,i:].sum(axis=2)

# determine damage state of bldg.
rv = np.dot(np.random.rand(nbldgs,nevents,1),np.ones((1,nds+1))) # nbldgs, nevents, (nds+1)
ds_bldgs_rnd = (rv < pe_x).sum(axis=2)
ds_bldgs_rnd = np.array(ds_bldgs_rnd) # nbldgs, nevents

no_fat = np.zeros((nbldgs,nevents,nlevels)) # nbldgs, nevents, nlevels
for i in range(nbldgs):
	bldg = bldg_type[i]
	for j in range(nevents):
		if ds_bldgs_rnd[i,j] > 0:
			ds = list_ds[ds_bldgs_rnd[i,j]-1]
			for inj in range(nlevels):
				no_fat[i,j,inj] = fat_rate[bldg][ds][inj]*pop_est[i]

# fatality for portfolio
fid = open(join(output_dir,'summary_casualty_' + site_tag + '_Mw'+ str(mw) + '_mc.csv'),'wb')
file_writer = csv.writer(fid)
file_writer.writerow(['suburb', 'population', 'severity1', 'severity2','severity3','severity4'])

# compute fatality by suburb
for sub in unq_sub:
	tf = (suburb == sub)
	no_fat_sub = no_fat[tf,:,:].sum(axis=0).mean(axis=0)
	tmp = []
	tmp.append(sub)
	tmp.append(sum(pop_est[tf]))
	[tmp.append(str(x)) for x in no_fat_sub]
	file_writer.writerow(tmp)

# histogram
plt.figure()
for i in range(nlevels):
	plt.subplot(2,2,i+1)
	plt.hist(no_fat[:,:,i].sum(axis=0))
	plt.title('Injury level:' + str(i+1))
	plt.grid(1)
plt.savefig(join(output_dir,'histogram_fatality_' + site_tag + '_Mw' + str(mw) + '_mc.png'),format='png')

# percentage of bldgs in each damage state by suburb
no_ds_bldgs_by_sub = {}
fid = open(join(output_dir,'summary_no_bldgs_ds_by_suburb_' + site_tag + '_Mw'+ str(mw) + '_extended.csv'),'wb')
file_writer = csv.writer(fid)
file_writer.writerow(['Suburb', 'No', 'DS0(%)', 'DS1(%)', 'DS2(%)', 'DS3(%)', 'DS4(%)','DS5(%)'])

for sub in unq_sub:
	tf = (suburb == sub)
	ds_bldgs = ds_bldgs_rnd[tf,:] # nbldgs, nevents
	no_ds_bldgs = []
	# ds <=> (0,1,2,3,4,5)
	for i in range(6):
		no_ds_bldgs.append((ds_bldgs == i).sum(axis=0))
	no_ds_bldgs = np.array(no_ds_bldgs) # 6, nevents
	no_ds_bldgs_by_sub[sub] = no_ds_bldgs.mean(axis=1)
	tmp = []
	tmp.append(sub)
	no_bldgs = sum(suburb == sub)
	tmp.append(no_bldgs)
	[tmp.append(str(x/no_bldgs*100.0)) for x in no_ds_bldgs_by_sub[sub]]
	file_writer.writerow(tmp)
fid.close()

###############################################################################
# 3. ATC Method
###############################################################################

CDF_lookup = np.array([[  0,      0,          0,          0],
                [0.5,    3.0/100000,   1.0/250000,   1.0/1000000],
                [5,      3.0/10000,    1.0/25000,    1.0/100000],
                [20,     3.0/1000,     1.0/2500,     1.0/10000],
                [45,     3.0/100,      1.0/250,      1.0/1000],
                [80,     3.0/10,       1.0/25,       1.0/100],
                [100,    2.0/5,        2.0/5,        1.0/5]]) # 7, 4 injuries
CDF_lookup[:,0] = 1.0e-2*CDF_lookup[:,0]
ninjury = CDF_lookup.shape[1]-1
no_fat = np.zeros((nbldgs,nevents,ninjury)) # nbldgs, nevents, ninjury
# bldg_loss (nbldg,nevents)/total_repl_cost (nbldg,2)
ratio_ind = bldg_loss/np.multiply(total_repl_cost[:,0][:,np.newaxis],np.ones((1,nevents)))
for i in range(ninjury):
	f_i = interp1d(CDF_lookup[:,0],CDF_lookup[:,i+1])
	tmp = f_i(ratio_ind)*np.multiply(pop_est[:,np.newaxis],np.ones((1,nevents)))
	no_fat[:,:,i] = tmp[:]

# fatality for portfolio
fid = open(join(output_dir,'summary_casualty_' + site_tag + '_Mw'+ str(mw) + '_ATC.csv'),'wb')
file_writer = csv.writer(fid)
file_writer.writerow(['suburb', 'population', 'injury1', 'injury2','injury3'])

# compute fatality by suburb
for sub in unq_sub:
	tf = (suburb == sub)
	no_fat_sub = no_fat[tf,:,:].sum(axis=0).mean(axis=0)
	tmp = []
	tmp.append(sub)
	tmp.append(sum(pop_est[tf]))
	[tmp.append(str(x)) for x in no_fat_sub]
	file_writer.writerow(tmp)

# histogram
plt.figure()
for i in range(ninjury):
	plt.subplot(2,2,i+1)
	plt.hist(no_fat[:,:,i].sum(axis=0))
	plt.title('Injury level:' + str(i+1))
	plt.grid(1)
plt.savefig(join(output_dir,'histogram_fatality_' + site_tag + '_Mw' + str(mw) + '_ATC.png'),format='png')

'''
ds = ['None','Slight','Moderate','Extensive','Complete']
for i in range(5):
	plt.figure()
	plt.hist(no_ds_bldgs_rnd[:,i])
	plt.plot(no_ds_bldgs_med[i],0,'r*')
	plt.title('Damage state: '+ ds[i])

# check gms
soil_med = np.load('./scen_risk_mw5.49_med/' + site_tag + '_motion/' + 'soil_SA.npy')
soil_med = soil_med[0,0,0,:,0,:] # nbldgs, nperiods

soil_rnd = np.load('./scen_risk_mw5.49_serial/' + site_tag + '_motion/' + 'soil_SA.npy')
soil_rnd = soil_rnd[0,0,0,:,:,:]


plt.figure()
plt.plot(soil_med[:,3],stats.mstats.gmean(soil_rnd[:,:,3],axis=1),'.')

# check gms
bed_med = np.load('./scen_risk_mw5.49_med/' + site_tag + '_motion/' + 'bedrock_SA.npy')
bed_med = bed_med[0,0,0,:,0,:] # nbldgs, nperiods

bed_rnd = np.load('./scen_risk_mw5.49_serial/' + site_tag + '_motion/' + 'bedrock_SA.npy')
bed_rnd = bed_rnd[0,0,0,:,:,:]

plt.figure()
plt.plot(bed_med[:,3],stats.mstats.gmean(bed_rnd[:,:,3],axis=1),'.')

if __name__ == "__main__":

	import sys
		
	
	site_tag = 'Karratha'
	ncpus = 4
	input_dir = r'r:/WA_FESA/Karratha/input'
	output_dir = r'r:/WA_FESA/Karratha/nci/scen_risk_mw5.09'
	
		
	ext_frg(str_type, occ_type, output_dir, atten_model, flag_fig, flag_save)
'''