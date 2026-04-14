# Makefile: Benchmark commands
# Usage: make <target> | make help

# You can create a list in the followinf form : START:END:STEP

# You hawe the following choices for sets
#for cpsat, maybe use 15 * 50 * 50 = 37500

PY = python
RUNNER = benchmark.run_strategies_match

RUNS = 15
TIMEOUT = 30

GEN = crypto_big

N = 2:50:1
DENSITIES = 0.1:5:0.1

BASE = $(PY) -m $(RUNNER) \
	--runs $(RUNS) --timeout $(TIMEOUT) \
	--n $(N) --densities $(DENSITIES) \
	--gen $(GEN)

# -- Lattice studies
#$(BASE) --suite delta --deltas 0.5,0.75,0.99 --out lll_study_0
lll_study:
	$(BASE) --suite lattice_delta --deltas 0.99,0.995,0.999 --out lattice/lll_study_1

bkz_study:
	$(BASE) --suite lattice_block --blocks 10,20,30 --out lattice/bkz_study

lattice_scaling_study_lll:
	$(BASE) --suite lattice_scaling_lll --out lattice/lll_scaling_study

lattice_scaling_study_bkz:
	$(BASE) --suite lattice_scaling_bkz --out lattice/bkz_scaling_study

lattice_arch:
	$(BASE) --suite lattice_arch --out lattice/arch_comparison

lattice_bkz_adaptative:
	$(BASE) --suite lattice_bkz_adaptative --blocks 10,20,30 --out lattice/bkz_adaptative

all_lattice: lll_study bkz_study lattice_seq_indep lattice_scaling lattice_arch


#-- CPSAT studies

cpsat_min_sat:
	$(BASE) --suite cpsat_min_sat --out cpsat/cpsat_min_sat


#-- Hybrid studies
smart_lattice_cpsat:
	$(BASE) --suite smart_lattice_cpsat --out hybrid/smart_lattice_cpsat


#--gray studies
gray_landscape:
	$(BASE) --suite gray_landscape --out gray/gray_landscape

# arch
# gerer le fallback