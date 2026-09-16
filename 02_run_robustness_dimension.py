from pathlib import Path
import importlib.util, sys
import numpy as np, pandas as pd
P=Path('/mnt/data/nem_v10_preserved/code/01_run_baseline.py')
spec=importlib.util.spec_from_file_location('m',P);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
out=Path('/mnt/data/nem_v10_preserved/output')
dim=sys.argv[1]

d,_=m.b.build_panel();D2ref=float(d[d.year==2024].D2_MWh.mean());d['D1n']=d.D1_MWh/D2ref;d['D2n']=d.D2_MWh/D2ref;d['curshare']=(d.curtail_MWh/d.solar_cap_proxy_MWh).clip(0,.5)
KR0=float(d[d.year==2024].groupby('month').solar_cap_proxy_MWh.median().mean()/D2ref);KS0=2800.0/D2ref;KR,KS=m.b.cap_paths(KR0,KS0)
p24=d[d.year==2024];KF0=float((p24.dispatchable_avail_effective_MWh/D2ref).median());Rbar=max(float(((p24.dispatchable_avail_effective_MWh-p24.D2_MWh)/D2ref).median()),.01)
cal=d[d.year.isin([2024,2025])].copy();krc=[];ksc=[]
for r in cal.itertuples(index=False):i=r.year-2024;mo=r.month-1;krc.append(KR[i,mo]);ksc.append(KS[i,mo])
krc=np.array(krc);ksc=np.array(ksc)
a,g,_=m.calibrate(cal,krc,ksc,KF0,Rbar,'curtailment_50',70.0)
months,dr=m.draw_common(d)
rows=[]
if dim=='reserve':
    for rb in [.10,Rbar,.30]:
        aa,gg,_=m.calibrate(cal,krc,ksc,KF0,float(rb),'curtailment_50',70.0);fx,_=m.fixed_transition(d,months,dr,KR,KS,KF0,float(rb),aa,gg,'curtailment_50',70.0,False,D2ref)
        for yr in [2040,2050]:
            f=fx[fx.year==yr].iloc[0];rows.append({'dimension':'reserve_target','scenario':f'Rbar={rb:.3f}','year':yr,'alpha':aa,'gamma':gg,'E_p2_fixed':f.E_p2,'P_scarcity_fixed':f.P_scarcity,'P_shortfall_fixed':f.P_shortfall})
elif dim=='gamma':
    for gf in [1.0,2.0,g,5.0]:
        if abs(gf-g)<1e-8:aa,gg=a,g
        else:aa,gg,_=m.calibrate(cal,krc,ksc,KF0,Rbar,'curtailment_50',70.0,gamma_fixed=gf)
        fx,_=m.fixed_transition(d,months,dr,KR,KS,KF0,Rbar,aa,gg,'curtailment_50',70.0,False,D2ref)
        for yr in [2040,2050]:
            f=fx[fx.year==yr].iloc[0];rows.append({'dimension':'gamma','scenario':f'gamma={gg:.3g}','year':yr,'alpha':aa,'gamma':gg,'E_p2_fixed':f.E_p2,'P_scarcity_fixed':f.P_scarcity,'P_shortfall_fixed':f.P_shortfall})
elif dim=='cf':
    for cf in [50.,70.,100.,150.]:
        aa,gg,_=m.calibrate(cal,krc,ksc,KF0,Rbar,'curtailment_50',cf);fx,_=m.fixed_transition(d,months,dr,KR,KS,KF0,Rbar,aa,gg,'curtailment_50',cf,False,D2ref)
        for yr in [2040,2050]:
            f=fx[fx.year==yr].iloc[0];rows.append({'dimension':'c_f','scenario':f'cF={cf:.0f}','year':yr,'alpha':aa,'gamma':gg,'E_p2_fixed':f.E_p2,'P_scarcity_fixed':f.P_scarcity,'P_shortfall_fixed':f.P_shortfall})
elif dim=='demand':
    fx,_=m.fixed_transition(d,months,dr,KR,KS,KF0,Rbar,a,g,'curtailment_50',70.0,True,D2ref)
    for yr in [2030,2040,2050]:
        f=fx[fx.year==yr].iloc[0];rows.append({'dimension':'demand_growth','scenario':'StepChange_demand_only_stress_test','year':yr,'alpha':a,'gamma':g,'E_p2_fixed':f.E_p2,'P_scarcity_fixed':f.P_scarcity,'P_shortfall_fixed':f.P_shortfall,'E_shortfall_MWh_day':f.E_shortfall_MWh_day})
else: raise SystemExit(dim)
pd.DataFrame(rows).to_csv(out/f'v10_robustness_{dim}.csv',index=False)
print(pd.DataFrame(rows).to_string(index=False))
