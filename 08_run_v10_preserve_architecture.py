from pathlib import Path
import importlib.util, json, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares, minimize_scalar

# Import corrected common-time builder and anchored capacity paths.
SRC = Path('/mnt/data/nem_v9/code/07_run_v10_aligned_rooftop_imputed.py')
spec = importlib.util.spec_from_file_location('v10base', SRC)
b = importlib.util.module_from_spec(spec); spec.loader.exec_module(b)

SEED=42; N_SIM=100_000; YEARS=np.arange(2024,2051); ETA=.90; MPC=20300.; H=2.5
STEP_CHANGE_TWH_2024=172.0; STEP_CHANGE_TWH_2050=312.0
OUT=Path('/mnt/data/nem_v10_preserved/output'); OUT.mkdir(parents=True,exist_ok=True)
CODEOUT=Path('/mnt/data/nem_v10_preserved/code'); CODEOUT.mkdir(parents=True,exist_ok=True)

def demand_scale(year):
    return (STEP_CHANGE_TWH_2050/STEP_CHANGE_TWH_2024)**((year-2024)/(2050-2024))

def scarcity_state(D1,D2,rho,cur,KR,KS,KF,Rbar,alpha,gamma,kappa,mode='curtailment_50',c_f=70.0):
    solar=rho*KR
    agg=np.maximum(solar-D1,0.0)
    frac={'aggregate':0.0,'curtailment_50':.5,'curtailment_100':1.0}[mode]
    surplus=np.maximum(agg, frac*np.maximum(cur*KR,0.0))
    x=np.minimum(np.minimum(KS,surplus),D2/ETA)
    y=ETA*x
    residual=np.maximum(D2-y,0.0)
    q1=np.maximum(D1-solar,0.0)
    avail=KF*kappa
    q2=np.minimum(residual,avail)
    short=np.maximum(residual-avail,0.0)
    reserve=np.maximum(avail-residual,0.0)
    z=np.maximum(Rbar-reserve,0.0)
    u=np.clip(z/Rbar,0,1) if Rbar>0 else np.zeros_like(z)
    adder=np.minimum(alpha*u**gamma,MPC-c_f)
    p2=c_f+adder
    p1=np.where(solar>D1,0.0,c_f)
    return dict(solar=solar,surplus=surplus,x=x,y=y,residual=residual,q1=q1,q2=q2,short=short,reserve=reserve,z=z,u=u,adder=adder,p1=p1,p2=p2)

def calibrate(cal, krc, ksc, KF, Rbar, mode='curtailment_50', c_f=70.0, gamma_fixed=None):
    D1=cal.D1n.values; D2=cal.D2n.values; rho=cal.rho.values; kap=cal.kappa.values; cur=cal.curshare.values; obs=cal.obs_p2.values
    om=float(obs.mean()); os=float(obs.std(ddof=1))
    def prices(a,g): return scarcity_state(D1,D2,rho,cur,krc,ksc,KF,Rbar,a,g,kap,mode,c_f)['p2']
    if gamma_fixed is None:
        def residual(theta):
            a=float(np.exp(theta[0])); g=float(np.exp(theta[1])); p=prices(a,g)
            return np.array([(p.mean()-om)/100.0,(p.std(ddof=1)-os)/300.0])
        r=least_squares(residual, x0=np.log([3500.0,4.0]), bounds=(np.log([1.0,1.0]),np.log([10000.0,20.0])), xtol=1e-11,ftol=1e-11,gtol=1e-11,max_nfev=300)
        a,g=np.exp(r.x);a=float(a);g=float(g)
    else:
        g=float(gamma_fixed)
        def obj(loga):
            a=float(np.exp(loga));p=prices(a,g)
            return ((p.mean()-om)/100.0)**2+((p.std(ddof=1)-os)/300.0)**2
        r=minimize_scalar(obj,bounds=(np.log(1.0),np.log(10000.0)),method='bounded',options={'xatol':1e-10,'maxiter':200})
        a=float(np.exp(r.x))
    p=prices(a,g)
    mom={'alpha':a,'gamma':g,'obs_mean':om,'sim_mean':float(p.mean()),'obs_sd':os,'sim_sd':float(p.std(ddof=1))}
    for q in [.75,.9,.95,.99]: mom[f'obs_q{int(q*100)}']=float(np.quantile(obs,q));mom[f'sim_q{int(q*100)}']=float(np.quantile(p,q))
    return a,g,mom

def draw_common(d):
    rng=np.random.default_rng(SEED)
    months=rng.choice(np.arange(1,13),N_SIM)
    cols=['D1n','D2n','rho','kappa','curshare']
    out={c:np.empty(N_SIM) for c in cols}
    for m in range(1,13):
        pos=np.where(months==m)[0];pool=d[d.month==m].reset_index(drop=True);take=rng.integers(0,len(pool),len(pos));s=pool.iloc[take]
        for c in cols: out[c][pos]=s[c].to_numpy()
    return months,out

def fixed_transition(d, months, dr, KR, KS, KF, Rbar, alpha, gamma, mode='curtailment_50', c_f=70.0, demand_growth=False, D2ref=None):
    rows=[]; samples={}
    if D2ref is None: D2ref=float(d[d.year==2024].D2_MWh.mean())
    for i,y in enumerate(YEARS):
        sc=demand_scale(y) if demand_growth else 1.0
        D1=dr['D1n']*sc; D2=dr['D2n']*sc; rho=dr['rho']; kap=dr['kappa']; cur=dr['curshare']
        kr=KR[i,months-1]; ks=KS[i,months-1]
        st=scarcity_state(D1,D2,rho,cur,kr,ks,KF,Rbar,alpha,gamma,kap,mode,c_f)
        xbar=np.minimum(st['surplus'],D2/ETA); constrained=ks < xbar - 1e-12
        V_day=ETA*np.mean(st['p2']*constrained)
        rent=float(np.mean(st['adder']*st['q2'])*D2ref)
        ps=float(np.mean(st['z']>1e-12)); pl=float(np.mean(st['short']>1e-12))
        rows.append({
            'year':int(y),'demand_scale':sc,'KR_year_end':float(KR[i,-1]),'KS_year_end':float(KS[i,-1]),
            'E_p1':float(np.mean(st['p1'])),'E_p2':float(np.mean(st['p2'])),'E_spread':float(np.mean(st['p2']-st['p1'])),
            'E_scarcity_adder':float(np.mean(st['adder'])),'E_storage_discharge':float(np.mean(st['y'])),'E_surplus':float(np.mean(st['surplus'])),
            'P_storage_active':float(np.mean(st['x']>1e-12)),'P_storage_capacity_constrained':float(np.mean(constrained)),
            'E_qD1':float(np.mean(st['q1'])),'E_qD2':float(np.mean(st['q2'])),'E_reserve':float(np.mean(st['reserve'])),
            'P_scarcity':ps,'P_shortfall':pl,'E_shortfall_MWh_day':float(np.mean(st['short'])*D2ref),
            'Phi_D_AUD_day':rent,'Phi_D_AUD_million_day':rent/1e6,'Phi_D_AUD_billion_year':rent*365/1e9,
            'V_S_AUD_per_MWh_day':float(V_day),'V_S_AUD_per_MWh_year':float(V_day*365)
        })
        if int(y) in [2024,2030,2040,2050]: samples[int(y)]=st['p2'].copy()
    return pd.DataFrame(rows),samples

def prestates(dr,months,KR,KS,Rbar,a,g,mode,c_f=70.0,demand_growth=False):
    o={}
    for i,y in enumerate(YEARS):
        sc=demand_scale(y) if demand_growth else 1.0
        D1=dr['D1n']*sc;D2=dr['D2n']*sc;rho=dr['rho'];cur=dr['curshare'];kap=dr['kappa'];kr=KR[i,months-1];ks=KS[i,months-1]
        solar=rho*kr;agg=np.maximum(solar-D1,0);frac={'aggregate':0,'curtailment_50':.5,'curtailment_100':1}[mode];sur=np.maximum(agg,frac*np.maximum(cur*kr,0));dis=ETA*np.minimum(np.minimum(ks,sur),D2/ETA)
        o[int(y)]={'k':kap,'res':np.maximum(D2-dis,0),'R':Rbar,'a':a,'g':g,'c_f':c_f,'dis':dis,'sur':sur,'D2':D2}
    return o

def evalcap(st,KF,D2ref):
    av=KF*st['k'];q=np.minimum(st['res'],av);short=np.maximum(st['res']-av,0);reserve=np.maximum(av-st['res'],0);z=np.maximum(st['R']-reserve,0);u=np.clip(z/st['R'],0,1);adder=np.minimum(st['a']*u**st['g'],MPC-st['c_f']);p=st['c_f']+adder;rent=float(np.mean(adder*q)*D2ref);cap=KF*D2ref/H;rpm=rent*365/cap if cap else np.nan
    return {'cap':cap,'p':float(p.mean()),'ps':float(np.mean(z>1e-12)),'pl':float(np.mean(short>1e-12)),'eue':float(np.mean(short)*D2ref),'rpm':float(rpm),'rent':rent,'q':float(q.mean()),'reserve':float(reserve.mean()),'adder':float(np.mean(adder))}

def solve(st,Kprev,cost,D2ref):
    if evalcap(st,Kprev,D2ref)['rpm']>=cost:return Kprev
    lo=max(1e-6,Kprev*1e-4);hi=Kprev
    if evalcap(st,lo,D2ref)['rpm']<cost:return lo
    for _ in range(45):
        mid=(lo+hi)/2
        if evalcap(st,mid,D2ref)['rpm']>=cost:lo=mid
        else:hi=mid
    return lo

def retirement_path(states,KF0,cost,D2ref):
    rows=[];kp=KF0
    for y in YEARS:
        y=int(y);K=KF0 if y==2024 else solve(states[y],kp,cost,D2ref);e=evalcap(states[y],K,D2ref);f=evalcap(states[y],KF0,D2ref)
        rows.append({'year':y,'KD_norm':K,'KD_fraction_initial':K/KF0,'capacity_MW_equilibrium':e['cap'],'E_p2_equilibrium':e['p'],'P_scarcity_equilibrium':e['ps'],'P_shortfall_equilibrium':e['pl'],'E_shortfall_MWh_day_equilibrium':e['eue'],'rent_per_MW_year_equilibrium':e['rpm'],'Phi_D_AUD_million_day_equilibrium':e['rent']/1e6,'E_p2_fixedfleet':f['p'],'P_scarcity_fixedfleet':f['ps'],'P_shortfall_fixedfleet':f['pl'],'Phi_D_AUD_million_day_fixedfleet':f['rent']/1e6})
        kp=K
    return pd.DataFrame(rows)

def make_figures(fixed,ret,samples,Rbar):
    y=fixed.year.to_numpy()
    # 1 original prices
    plt.figure(figsize=(7,4.5));plt.plot(y,fixed.E_p1,label='Expected daytime price');plt.plot(y,fixed.E_p2,label='Expected evening price');plt.plot(y,fixed.E_scarcity_adder,linestyle='--',label='Expected scarcity premium');plt.xlabel('Year');plt.ylabel('AUD/MWh');plt.title('Expected prices and scarcity premium: fixed-fleet benchmark');plt.legend();plt.tight_layout();plt.savefig(OUT/'v10_fig1_prices_fixed.pdf',bbox_inches='tight');plt.close()
    # 2 original dispatch/reserve
    plt.figure(figsize=(7,4.5));plt.plot(y,fixed.E_storage_discharge,label='Storage discharge');plt.plot(y,fixed.E_qD2,label='Dispatchable peak output');plt.plot(y,fixed.E_reserve,linestyle='--',label='Available reserve');plt.axhline(Rbar,linewidth=.8,linestyle=':',label='Reserve target');plt.xlabel('Year');plt.ylabel('Fraction of 2024 peak-block energy');plt.title('Storage, dispatchable output, and reserves: fixed fleet');plt.legend();plt.tight_layout();plt.savefig(OUT/'v10_fig2_dispatch_reserve_fixed.pdf',bbox_inches='tight');plt.close()
    # 3 original scarcity/capacity
    fig,ax1=plt.subplots(figsize=(7,4.5));ax1.plot(y,fixed.P_scarcity,label='Reserve-scarcity probability');ax1.plot(y,fixed.P_shortfall,linestyle='--',label='Model capacity-shortfall probability');ax1.set_xlabel('Year');ax1.set_ylabel('Probability');ax1.set_ylim(0,1);ax2=ax1.twinx();ax2.plot(y,fixed.KR_year_end,linestyle='-.',label='Renewable capacity');ax2.plot(y,ETA*fixed.KS_year_end,linestyle=':',label='Effective storage discharge capacity');ax2.set_ylabel('Capacity relative to 2024 peak-block energy');l1,la1=ax1.get_legend_handles_labels();l2,la2=ax2.get_legend_handles_labels();ax2.legend(l1+l2,la1+la2,loc='upper right');plt.title('Scarcity, model shortfall, and capacity build-out');fig.tight_layout();plt.savefig(OUT/'v10_fig3_scarcity_capacity_fixed.pdf',bbox_inches='tight');plt.close()
    # 4 rents, fixed vs endogenous
    plt.figure(figsize=(7,4.5));plt.plot(y,fixed.Phi_D_AUD_million_day,label='Fixed fleet');plt.plot(ret.year,ret.Phi_D_AUD_million_day_equilibrium,label='Endogenous retirement');plt.xlabel('Year');plt.ylabel('AUD million per representative day');plt.title('Dispatchable scarcity rents');plt.legend();plt.tight_layout();plt.savefig(OUT/'v10_fig4_dispatchable_rents.pdf',bbox_inches='tight');plt.close()
    # 5 storage value
    plt.figure(figsize=(7,4.5));plt.plot(y,fixed.V_S_AUD_per_MWh_year/1000);plt.xlabel('Year');plt.ylabel('AUD thousand per MWh-year');plt.title('Pre-exit marginal energy-market value of storage');plt.tight_layout();plt.savefig(OUT/'v10_fig5_storage_value.pdf',bbox_inches='tight');plt.close()
    # 6 price feedback
    plt.figure(figsize=(7,4.5));plt.plot(y,ret.E_p2_fixedfleet,label='Fixed fleet');plt.plot(y,ret.E_p2_equilibrium,label='Endogenous retirement');plt.xlabel('Year');plt.ylabel('AUD/MWh');plt.title('Evening-price feedback from endogenous retirement');plt.legend();plt.tight_layout();plt.savefig(OUT/'v10_fig6_prices_retirement.pdf',bbox_inches='tight');plt.close()
    # 7 capacity
    plt.figure(figsize=(7,4.5));plt.plot(y,100*ret.KD_fraction_initial);plt.xlabel('Year');plt.ylabel('Remaining dispatchable capacity (% of 2024)');plt.title('Privately supported dispatchable capacity');plt.tight_layout();plt.savefig(OUT/'v10_fig7_endogenous_capacity.pdf',bbox_inches='tight');plt.close()
    # 8 histograms
    fig=plt.figure(figsize=(10,7.5));allv=np.concatenate(list(samples.values()));right=max(500,float(np.quantile(allv,.995))*1.05);bins=np.linspace(float(np.min(allv)),right,45)
    for i,yr in enumerate([2024,2030,2040,2050],1):
        ax=fig.add_subplot(2,2,i);ax.hist(samples[yr],bins=bins,density=True);ax.set_title(f'Evening-price distribution: {yr}');ax.set_xlabel('AUD/MWh');ax.set_ylabel('Density')
    fig.tight_layout();plt.savefig(OUT/'v10_fig8_price_histograms.pdf',bbox_inches='tight');plt.close()

def main():
    d,_=b.build_panel(); D2ref=float(d[d.year==2024].D2_MWh.mean()); d['D1n']=d.D1_MWh/D2ref;d['D2n']=d.D2_MWh/D2ref;d['curshare']=(d.curtail_MWh/d.solar_cap_proxy_MWh).clip(0,.5)
    KR0=float(d[d.year==2024].groupby('month').solar_cap_proxy_MWh.median().mean()/D2ref);KS0=2800.0/D2ref;KR,KS=b.cap_paths(KR0,KS0)
    p24=d[d.year==2024];KF0=float((p24.dispatchable_avail_effective_MWh/D2ref).median());Rbar=max(float(((p24.dispatchable_avail_effective_MWh-p24.D2_MWh)/D2ref).median()),.01)
    cal=d[d.year.isin([2024,2025])].copy();krc=[];ksc=[]
    for r in cal.itertuples(index=False):i=r.year-2024;m=r.month-1;krc.append(KR[i,m]);ksc.append(KS[i,m])
    krc=np.array(krc);ksc=np.array(ksc)
    a,g,mom=calibrate(cal,krc,ksc,KF0,Rbar,'curtailment_50',70.0)
    pd.DataFrame([mom]).to_csv(OUT/'v10_calibration_moments.csv',index=False)
    months,dr=draw_common(d)
    fixed,samples=fixed_transition(d,months,dr,KR,KS,KF0,Rbar,a,g,'curtailment_50',70.0,False,D2ref)
    fixed.to_csv(OUT/'v10_fixed_fleet_summary.csv',index=False)
    states=prestates(dr,months,KR,KS,Rbar,a,g,'curtailment_50',70.0,False);rent2024=evalcap(states[2024],KF0,D2ref)['rpm'];cost=.5*rent2024
    ret=retirement_path(states,KF0,cost,D2ref);ret.to_csv(OUT/'v10_endogenous_retirement_path.csv',index=False)
    # cost sensitivity
    rows=[]
    for ratio in [.25,.5,.75]:
        pp=retirement_path(states,KF0,ratio*rent2024,D2ref);first=pp.loc[pp.KD_fraction_initial<.999999,'year'];fy=int(first.iloc[0]) if len(first) else None;r=pp[pp.year==2050].iloc[0]
        rows.append({'cost_ratio':ratio,'first_exit_year':fy,'capacity_pct_2050':100*r.KD_fraction_initial,'capacity_MW_2050':r.capacity_MW_equilibrium,'E_p2_2050':r.E_p2_equilibrium,'P_scarcity_2050':r.P_scarcity_equilibrium,'P_shortfall_2050':r.P_shortfall_equilibrium})
    pd.DataFrame(rows).to_csv(OUT/'v10_cost_sensitivity.csv',index=False)
    # charging access sensitivity (same absolute cost, recalibrate)
    rows=[]
    for mode,label in [('aggregate','0% accessible curtailment'),('curtailment_50','50% accessible curtailment'),('curtailment_100','100% accessible curtailment')]:
        aa,gg,_=calibrate(cal,krc,ksc,KF0,Rbar,mode,70.0);fx,_=fixed_transition(d,months,dr,KR,KS,KF0,Rbar,aa,gg,mode,70.0,False,D2ref);ss=prestates(dr,months,KR,KS,Rbar,aa,gg,mode,70.0,False);pp=retirement_path(ss,KF0,cost,D2ref)
        for yr in [2030,2040,2050]:
            f=fx[fx.year==yr].iloc[0];r=pp[pp.year==yr].iloc[0]
            rows.append({'dimension':'charging_access','scenario':label,'year':yr,'alpha':aa,'gamma':gg,'E_p2_fixed':f.E_p2,'P_scarcity_fixed':f.P_scarcity,'P_shortfall_fixed':f.P_shortfall,'P_storage_active':f.P_storage_active,'E_p2_equilibrium':r.E_p2_equilibrium,'capacity_pct_equilibrium':100*r.KD_fraction_initial,'P_scarcity_equilibrium':r.P_scarcity_equilibrium})
    # reserve target robustness, fixed fleet, same original design
    for rb in [.10,Rbar,.30]:
        aa,gg,_=calibrate(cal,krc,ksc,KF0,float(rb),'curtailment_50',70.0);fx,_=fixed_transition(d,months,dr,KR,KS,KF0,float(rb),aa,gg,'curtailment_50',70.0,False,D2ref)
        for yr in [2040,2050]:
            f=fx[fx.year==yr].iloc[0];rows.append({'dimension':'reserve_target','scenario':f'Rbar={rb:.3f}','year':yr,'alpha':aa,'gamma':gg,'E_p2_fixed':f.E_p2,'P_scarcity_fixed':f.P_scarcity,'P_shortfall_fixed':f.P_shortfall,'P_storage_active':f.P_storage_active})
    # gamma robustness; include estimated reference for table
    for gf in [1.0,2.0,g,5.0]:
        if abs(gf-g)<1e-8: aa,gg=a,g
        else: aa,gg,_=calibrate(cal,krc,ksc,KF0,Rbar,'curtailment_50',70.0,gamma_fixed=gf)
        fx,_=fixed_transition(d,months,dr,KR,KS,KF0,Rbar,aa,gg,'curtailment_50',70.0,False,D2ref)
        for yr in [2040,2050]:
            f=fx[fx.year==yr].iloc[0];rows.append({'dimension':'gamma','scenario':f'gamma={gg:.3g}','year':yr,'alpha':aa,'gamma':gg,'E_p2_fixed':f.E_p2,'P_scarcity_fixed':f.P_scarcity,'P_shortfall_fixed':f.P_shortfall,'P_storage_active':f.P_storage_active})
    # cF robustness
    for cf in [50.,70.,100.,150.]:
        aa,gg,_=calibrate(cal,krc,ksc,KF0,Rbar,'curtailment_50',cf);fx,_=fixed_transition(d,months,dr,KR,KS,KF0,Rbar,aa,gg,'curtailment_50',cf,False,D2ref)
        for yr in [2040,2050]:
            f=fx[fx.year==yr].iloc[0];rows.append({'dimension':'c_f','scenario':f'cF={cf:.0f}','year':yr,'alpha':aa,'gamma':gg,'E_p2_fixed':f.E_p2,'P_scarcity_fixed':f.P_scarcity,'P_shortfall_fixed':f.P_shortfall,'P_storage_active':f.P_storage_active})
    # demand growth stress test
    fxg,_=fixed_transition(d,months,dr,KR,KS,KF0,Rbar,a,g,'curtailment_50',70.0,True,D2ref)
    for yr in [2030,2040,2050]:
        f=fxg[fxg.year==yr].iloc[0];rows.append({'dimension':'demand_growth','scenario':'StepChange_demand_only_stress_test','year':yr,'alpha':a,'gamma':g,'E_p2_fixed':f.E_p2,'P_scarcity_fixed':f.P_scarcity,'P_shortfall_fixed':f.P_shortfall,'P_storage_active':f.P_storage_active,'E_shortfall_MWh_day':f.E_shortfall_MWh_day})
    rob=pd.DataFrame(rows);rob.to_csv(OUT/'v10_robustness_summary.csv',index=False)
    make_figures(fixed,ret,samples,Rbar)
    meta={'seed':SEED,'n_sim':N_SIM,'n_days':len(d),'block_hours':H,'D2_ref_MWh':D2ref,'D1_D2_2024':float(d[d.year==2024].D1_MWh.mean()/D2ref),'KR0':KR0,'KS0':KS0,'KD0':KF0,'KD0_MW':KF0*D2ref/H,'Rbar':Rbar,'alpha':a,'gamma':g,'c_f':70.,'rent2024_per_MW_year':rent2024,'reference_avoidable_cost':cost,'reference_charging_access':'50% accessible curtailment','emissions_series_reported':False}
    (OUT/'v10_metadata.json').write_text(json.dumps(meta,indent=2))
    print('META',json.dumps(meta,indent=2));print('\nFIXED SELECTED');print(fixed[fixed.year.isin([2024,2030,2040,2050])].to_string(index=False));print('\nRET SELECTED');print(ret[ret.year.isin([2024,2030,2040,2050])].to_string(index=False));print('\nROB');print(rob.to_string(index=False))

if __name__=='__main__':main()
