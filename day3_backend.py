"""IT6117 Day 3 teaching helpers, reproducible scenarios, self-checks and optional solutions.

All hospital/LLM/RCT/drift examples are synthetic exercises; diabetes teaching data is
not a valid basis for a clinical deployment claim. No API, internet or PHI is used.
"""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path
import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, roc_auc_score, average_precision_score,
                             brier_score_loss, precision_score, recall_score)
from sklearn.model_selection import train_test_split
from scipy.stats import ks_2samp

# Load original teaching files from the data/ folder beside this Python script.
# The notebook and backend can be opened from anywhere; paths are script-relative.
# DAY3/
#   Day3_Student_Task.ipynb
#   day3_backend.py
#   data/
#     day2_diabetes_teaching.csv
#     day2_health_data_catalog.csv
DATA_DIR = Path(__file__).resolve().parent / "data"
DIABETES_CSV = DATA_DIR / "day2_diabetes_teaching.csv"
CATALOG_CSV = DATA_DIR / "day2_health_data_catalog.csv"

def _load_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Required Day 3 dataset not found: {path}\n"
            "Put day2_diabetes_teaching.csv and day2_health_data_catalog.csv "
            "inside the data/ folder next to day3_backend.py. "
            "File names must match exactly."
        )
    return pd.read_csv(path)

diabetes = _load_csv(DIABETES_CSV)
health_data_catalog = _load_csv(CATALOG_CSV)

____________ = None  # an untouched student blank is a valid, non-answer Python expression

TARGET = 'high_progression'
FEATURES = ['age','bmi','blood_pressure','glucose','hdl','log_triglycerides']
THRESHOLD = 0.40  # classroom operating point only; NOT a validated clinical threshold

# The label high_progression is constructed by thresholding continuous progression >=141.
# Region/sex labels are not actual hospitals, prospective subgroups, or external validation.
feature_timeline = pd.DataFrame([
    ('age',-30,'allowed'), ('bmi',-7,'allowed'), ('blood_pressure',-1,'allowed'),
    ('glucose',-1,'allowed'), ('post_score_treatment',2,'leakage'),
    ('discharge_diagnosis',10,'leakage'), ('progression',30,'outcome')
], columns=['variable','day_relative_to_score','role'])

STUDY_STAGES = {
    'same_dataset_random_split':'retrospective_internal',
    'different_hospital_same_model':'external_validation',
    'live_predictions_hidden_from_clinicians':'silent_trial',
    'randomized_ai_vs_usual_care_outcomes':'randomized_trial',
}

REPORT_FIELDS = {'population':'442 teaching records', 'outcome':'constructed high_progression',
                 'prediction_time':'illustrative baseline', 'horizon':None,
                 'independent_hospital':None, 'feature_list':FEATURES,
                 'random_seed':42, 'external_validation':False}

STUDY_GUIDES = pd.DataFrame([
    ('Retrospective prediction model study','TRIPOD+AI','Model reporting'),
    ('Prediction model bias and applicability','PROBAST+AI','Critical appraisal, NOT reporting'),
    ('Observational EHR cohort','STROBE + RECORD','Design and routine data'),
    ('Diagnostic accuracy of AI test','STARD-AI','Index test vs reference'),
    ('Randomized trial of AI decision support','CONSORT 2025 + CONSORT-AI','Patient impact'),
    ('AI randomized trial protocol','SPIRIT 2025 + SPIRIT-AI','Trial protocol'),
    ('Early prospective human–AI pilot','DECIDE-AI','Small-scale live evaluation'),
    ('Healthcare LLM benchmark','TRIPOD-LLM','LLM reporting'),
],columns=['objective','guideline','purpose'])

@lru_cache(maxsize=1)
def prediction_bundle():
    """Frozen internal split: all preprocessing fits only on training rows."""
    idx_train, idx_test = train_test_split(np.arange(len(diabetes)),test_size=0.30,
                              stratify=diabetes[TARGET],random_state=42)
    X_train=diabetes.iloc[idx_train][FEATURES].copy()
    X_test=diabetes.iloc[idx_test][FEATURES].copy()
    y_train=diabetes.iloc[idx_train][TARGET].astype(int).to_numpy()
    y_test=diabetes.iloc[idx_test][TARGET].astype(int).to_numpy()
    model=Pipeline([('impute',SimpleImputer(strategy='median')),
                    ('scale', StandardScaler()),
                    ('logistic',LogisticRegression(max_iter=1000,random_state=42))])
    model.fit(X_train,y_train)
    p_test=model.predict_proba(X_test)[:,1]
    groups=np.where(X_test['age'].to_numpy()>=np.median(diabetes['age']),'older','younger')
    return dict(model=model,X_train=X_train,X_test=X_test,y_train=y_train,y_test=y_test,
                p_test=p_test,train_ids=diabetes.iloc[idx_train]['record_id'].to_numpy(),
                test_ids=diabetes.iloc[idx_test]['record_id'].to_numpy(), age_group=groups,
                train_indices=idx_train,test_indices=idx_test)

def confusion_summary(y, p, threshold=THRESHOLD):
    cm=confusion_matrix(np.asarray(y,dtype=int),np.asarray(p)>=threshold,labels=[0,1])
    tn,fp,fn,tp=cm.ravel(); n=tn+fp+fn+tp
    return {'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp),'n':int(n),
            'sensitivity':tp/(tp+fn) if tp+fn else np.nan,
            'specificity':tn/(tn+fp) if tn+fp else np.nan,
            'ppv':tp/(tp+fp) if tp+fp else np.nan,
            'alerts':int(tp+fp),'alert_rate':(tp+fp)/n}

def calibration_table(y,p,n_bins=5):
    """Equal-width bins, count each; empty bins omitted. Exploratory small-sample summary."""
    df=pd.DataFrame({'y':np.asarray(y,dtype=int),'p':np.asarray(p,dtype=float)})
    df['bin']=pd.cut(df.p,bins=np.linspace(0,1,n_bins+1),include_lowest=True)
    return df.groupby('bin',observed=True).agg(n=('y','size'),mean_predicted=('p','mean'),observed_rate=('y','mean')).reset_index()

def subgroup_table(y,p,groups,threshold=THRESHOLD):
    out=[]
    for group in sorted(np.unique(groups)):
        take=np.asarray(groups)==group
        m=confusion_summary(np.asarray(y)[take], np.asarray(p)[take], threshold)
        out.append({'group':str(group),'n':m['n'],'events':int(np.asarray(y)[take].sum()),
                    'sensitivity':m['sensitivity'],'ppv':m['ppv'],
                    'alerts':m['alerts'],'alert_rate':m['alert_rate']})
    return pd.DataFrame(out)

def net_benefit(y,p,threshold):
    """TP/n - FP/n * threshold/(1-threshold), 0<threshold<1."""
    if not 0 < threshold < 1: raise ValueError('Threshold must be in (0, 1)')
    m=confusion_summary(y,p,threshold)
    return m['tp']/m['n']-m['fp']/m['n']*(threshold/(1-threshold))

def treat_all_net_benefit(y,threshold):
    prevalence=np.mean(np.asarray(y)==1)
    return prevalence-(1-prevalence)*threshold/(1-threshold)

def decision_curve(y,p,thresholds=None):
    if thresholds is None: thresholds=np.arange(.10,.81,.05)
    return pd.DataFrame([{'threshold':float(t),'model':net_benefit(y,p,float(t)),
                          'treat_all':treat_all_net_benefit(y,float(t)), 'treat_none':0.0,
                          'alerts':confusion_summary(y,p,float(t))['alerts']} for t in thresholds])

def rct_counts():
    """Hypothetical RCT counts, not observed clinical effectiveness."""
    return pd.DataFrame({'arm':['usual_care','ai_assisted'],'events':[60,44], 'patients':[500,500]})

def risk_difference(ai_events,ai_n,control_events,control_n):
    """Negative = fewer events under AI in a randomized illustration."""
    return ai_events/ai_n-control_events/control_n

def shift_scenario():
    """Synthetic incoming case-mix/device shift. Label-free drift signal."""
    b=prediction_bundle(); baseline=b['X_train'].copy()
    incoming=b['X_test'].copy()
    incoming['bmi']=incoming['bmi']+6.0
    incoming['glucose']=incoming['glucose']+9
    incoming.loc[incoming.index[:int(len(incoming)*.18)],'glucose']=np.nan
    return baseline,incoming

def drift_report(base,now):
    return pd.DataFrame([{'feature':f,
      'baseline_missing':float(base[f].isna().mean()),
      'incoming_missing':float(now[f].isna().mean()),
      'ks_statistic':float(ks_2samp(base[f].dropna(),now[f].dropna()).statistic)}
        for f in ['age','bmi','glucose']])

def triage_rule(missing_rate, max_missing=0.10):
    """Illustrative conservative *data-quality gate*, not a regulatory stop rule."""
    return 'pause_and_investigate' if missing_rate > max_missing else 'continue_monitored'

def model_comparison():
    """Same ranking, altered (intentionally miscalibrated) risk probabilities."""
    b=prediction_bundle(); calibrated=b['p_test']; distorted=np.sqrt(calibrated)
    return pd.DataFrame({'y':b['y_test'],'model_A':calibrated,'model_B':distorted})

def synthetic_llm_audit():
    """Fictional labeled audit cases; rates are not performance of any LLM vendor."""
    return pd.DataFrame([
        ('C01',1,0,0),('C02',1,1,0),('C03',1,0,0),('C04',1,1,1),
        ('C05',1,0,0),('C06',1,0,0),('C07',0,1,0),('C08',1,0,1),
        ('C09',1,1,0),('C10',1,0,0),('C11',1,0,0),('C12',0,1,1)
    ], columns=['case_id','clinically_correct','unsupported_claim','unsafe_action'])

def plot_overview():
    fig,axes=plt.subplots(1,2,figsize=(10,3.5))
    diabetes[TARGET].value_counts().sort_index().plot.bar(ax=axes[0],color=['#64748b','#2563eb'])
    axes[0].set_title('Constructed high-progression label (n=442)')
    axes[0].set_xlabel('Teaching outcome (0=no, 1=yes)');axes[0].set_ylabel('Records')
    diabetes['bmi'].hist(ax=axes[1],bins=20,color='#2563eb',edgecolor='white')
    axes[1].set_title('BMI in teaching data');axes[1].set_xlabel('BMI');axes[1].set_ylabel('Records')
    plt.tight_layout();plt.show()

def plot_evaluation():
    b=prediction_bundle(); fig,ax=plt.subplots(1,2,figsize=(10,3.6))
    t=calibration_table(b['y_test'],b['p_test'])
    ax[0].plot([0,1],[0,1],'k--',label='Ideal calibration')
    ax[0].scatter(t['mean_predicted'],t['observed_rate'],s=30+t['n']*2,label='Test bins')
    ax[0].set(xlabel='Mean predicted risk',ylabel='Observed fraction',title='Internal test: calibration (5 bins)',xlim=(0,1),ylim=(0,1));ax[0].legend()
    d=decision_curve(b['y_test'],b['p_test'])
    for col,ls in [('model','-'),('treat_all','--'),('treat_none',':')]:ax[1].plot(d.threshold,d[col],ls,label=col)
    ax[1].set(xlabel='Threshold probability',ylabel='Net benefit per person',title='Decision curve (illustrative)');ax[1].legend()
    plt.tight_layout();plt.show()

def plot_threshold_tradeoff():
    b=prediction_bundle(); xx=np.arange(.1,.81,.05)
    dd=pd.DataFrame([{'threshold':x,**confusion_summary(b['y_test'],b['p_test'],x)} for x in xx])
    fig,ax=plt.subplots(1,2,figsize=(10,3.5))
    ax[0].plot(dd.threshold,dd.alerts,'o-');ax[0].axhline(50,color='grey',linestyle='--',label='Illustrative 50-alert capacity')
    ax[0].set(xlabel='Threshold',ylabel='Alerts / 133 records',title='Workload');ax[0].legend()
    ax[1].plot(dd.threshold,dd.sensitivity,label='Sensitivity');ax[1].plot(dd.threshold,dd.ppv,label='PPV')
    ax[1].set(xlabel='Threshold',ylabel='Fraction',title='Missed events vs false alarms',ylim=(0,1));ax[1].legend()
    plt.tight_layout();plt.show()

def plot_subgroups():
    b=prediction_bundle();t=subgroup_table(b['y_test'],b['p_test'],b['age_group'])
    fig,ax=plt.subplots(figsize=(6.5,3.4));x=np.arange(len(t));w=.34
    ax.bar(x-w/2,t.sensitivity,w,label='Sensitivity');ax.bar(x+w/2,t.alert_rate,w,label='Alert rate')
    ax.set(xticks=x,xticklabels=[f'{g} (n={n})' for g,n in zip(t.group,t.n)],ylim=(0,1),ylabel='Fraction',title='Internal test subgroups: descriptive only');ax.legend()
    plt.tight_layout();plt.show()

def plot_drift():
    base,now=shift_scenario();fig,axes=plt.subplots(1,2,figsize=(9.5,3.3))
    axes[0].hist(base.bmi,bins=18,alpha=.55,label='Baseline');axes[0].hist(now.bmi,bins=18,alpha=.55,label='Incoming (synthetic)');axes[0].legend();axes[0].set(xlabel='BMI',ylabel='Records',title='Input distribution shift')
    miss=[base.glucose.isna().mean(),now.glucose.isna().mean()]
    axes[1].bar(['Baseline','Incoming'],miss);axes[1].axhline(.10,color='grey',linestyle='--',label='Demo gate 10%');axes[1].set(title='Glucose missingness (synthetic)',ylabel='Missing fraction',ylim=(0,.25));axes[1].legend();plt.tight_layout();plt.show()

def plot_rct():
    d=rct_counts();fig,ax=plt.subplots(figsize=(6,3));ax.bar(d.arm,100*d.events/d.patients);ax.set(ylabel='Illustrative outcome event rate (%)',title='Synthetic RCT counts (not clinical evidence)',ylim=(0,15));plt.tight_layout();plt.show()

def plot_llm():
    x=synthetic_llm_audit(); rates=x[['clinically_correct','unsupported_claim','unsafe_action']].mean()
    fig,ax=plt.subplots(figsize=(7,3));rates.plot.bar(ax=ax);ax.set(title='Fictional LLM safety audit (n=12)',ylabel='Fraction of cases',ylim=(0,1));plt.xticks(rotation=12,ha='right');plt.tight_layout();plt.show()

# Tutorial examples can be requested by students at any time; never shown on check failure.
TUTORIAL_CODE = {1: 'answer = round(diabetes["high_progression"].mean(), 3)',
 2: 'before_score = feature_timeline.loc[feature_timeline["day_relative_to_score"] <= 0]\n'
    'answer = before_score["variable"].tolist()',
 3: 'train_ids = set(bundle["train_ids"])\n'
    'test_ids = set(bundle["test_ids"])\n'
    'overlap = train_ids.intersection(test_ids)\n'
    'answer = len(overlap)',
 4: 'scenario = "different_hospital_same_model"\nanswer = STUDY_STAGES[scenario]',
 5: 'rows = STUDY_GUIDES.loc[STUDY_GUIDES["objective"] == "Retrospective prediction model '
    'study"]\n'
    'answer = rows["guideline"].iloc[0]',
 6: 'answer = [name for name, value in REPORT_FIELDS.items() if value is None]',
 7: 'counts = confusion_summary(bundle["y_test"], bundle["p_test"], THRESHOLD)\n'
    'answer = counts["tp"]',
 8: 'counts = confusion_summary(bundle["y_test"], bundle["p_test"], THRESHOLD)\n'
    'tp = counts["tp"]\n'
    'fn = counts["fn"]\n'
    'answer = tp / (tp + fn)',
 9: 'counts = confusion_summary(bundle["y_test"], bundle["p_test"], THRESHOLD)\n'
    'tp = counts["tp"]\n'
    'fp = counts["fp"]\n'
    'answer = tp / (tp + fp)',
 10: 'answer = roc_auc_score(bundle["y_test"], bundle["p_test"])',
 11: 'answer = average_precision_score(bundle["y_test"], bundle["p_test"])',
 12: 'answer = brier_score_loss(bundle["y_test"], bundle["p_test"])',
 13: 'bins = calibration_table(bundle["y_test"], bundle["p_test"])\n'
     'first_bin = bins.iloc[0]\n'
     'answer = first_bin["observed_rate"]',
 14: 'groups = subgroup_table(bundle["y_test"], bundle["p_test"], bundle["age_group"])\n'
     'older = groups.loc[groups["group"] == "older", "sensitivity"].iloc[0]\n'
     'younger = groups.loc[groups["group"] == "younger", "sensitivity"].iloc[0]\n'
     'answer = abs(older - younger)',
 15: 'new_threshold = 0.60\n'
     'predicted_alerts = bundle["p_test"] >= new_threshold\n'
     'answer = int(predicted_alerts.sum())',
 16: 'counts = confusion_summary(bundle["y_test"], bundle["p_test"], THRESHOLD)\n'
     'TP, FP, N = counts["tp"], counts["fp"], counts["n"]\n'
     'penalty = THRESHOLD / (1 - THRESHOLD)\n'
     'answer = TP / N - FP / N * penalty',
 17: 'prevalence = bundle["y_test"].mean()\n'
     'answer = prevalence - (1 - prevalence) * THRESHOLD / (1 - THRESHOLD)',
 18: 'ai_risk = 44 / 500\nusual_risk = 60 / 500\nanswer = ai_risk - usual_risk',
 19: 'baseline, incoming = shift_scenario()\n'
     'ks_result = ks_2samp(baseline["bmi"].dropna(), incoming["bmi"].dropna())\n'
     'answer = ks_result.statistic',
 20: 'baseline, incoming = shift_scenario()\n'
     'missing_rate = incoming["glucose"].isna().mean()\n'
     'answer = triage_rule(missing_rate, max_missing=0.10)'}

@lru_cache(maxsize=1)
def _expected():
    b = prediction_bundle()
    baseline, incoming = shift_scenario()
    counts = confusion_summary(b['y_test'], b['p_test'], THRESHOLD)
    groups = subgroup_table(b['y_test'], b['p_test'], b['age_group'])
    return {
        1: round(float(diabetes[TARGET].mean()), 3),
        2: feature_timeline.loc[feature_timeline.day_relative_to_score <= 0, 'variable'].tolist(),
        3: len(set(b['train_ids']) & set(b['test_ids'])),
        4: 'external_validation',
        5: 'TRIPOD+AI',
        6: ['horizon', 'independent_hospital'],
        7: counts['tp'],
        8: counts['sensitivity'],
        9: counts['ppv'],
        10: float(roc_auc_score(b['y_test'], b['p_test'])),
        11: float(average_precision_score(b['y_test'], b['p_test'])),
        12: float(brier_score_loss(b['y_test'], b['p_test'])),
        13: float(calibration_table(b['y_test'], b['p_test'])['observed_rate'].iloc[0]),
        14: float(abs(groups['sensitivity'].iloc[0] - groups['sensitivity'].iloc[1])),
        15: confusion_summary(b['y_test'], b['p_test'], .60)['alerts'],
        16: float(net_benefit(b['y_test'], b['p_test'], THRESHOLD)),
        17: float(treat_all_net_benefit(b['y_test'], THRESHOLD)),
        18: float(risk_difference(44, 500, 60, 500)),
        19: float(ks_2samp(baseline['bmi'].dropna(), incoming['bmi'].dropna()).statistic),
        20: 'pause_and_investigate',
    }

def _equivalent(value, expected):
    if value is None or value is Ellipsis:
        return False
    if isinstance(expected, (float, np.floating)):
        try:
            return bool(np.isclose(float(value), float(expected), rtol=1e-5, atol=1e-6))
        except (ValueError, TypeError):
            return False
    if isinstance(expected, list):
        return isinstance(value, (list, tuple)) and list(value) == expected
    try:
        return bool(value == expected)
    except (ValueError, TypeError):
        return False

def _format_result(v):
    if isinstance(v, (float, np.floating)):
        return f'{v:.4f}'
    return str(v)

def _interpret(n, v):
    b = prediction_bundle()
    c = confusion_summary(b['y_test'], b['p_test'], THRESHOLD)
    if n == 1:
        return f'About {float(v)*100:.1f}% of supplied records have the constructed label; this is not disease prevalence or a prospective clinical event rate.'
    if n == 2:
        return 'These variables precede the fictional prediction time. Future records or outcomes would create temporal leakage. Availability in a real EHR must be verified.'
    if n == 3:
        return 'Zero duplicate record IDs across the internal split is desirable, but does not demonstrate external transportability or rule out all forms of leakage.'
    if n == 4:
        return 'A genuinely new hospital with independently collected outcomes permits external validation of a frozen model; it does not establish that AI-assisted care improves outcomes.'
    if n == 5:
        return 'TRIPOD+AI is a reporting guideline for prediction models. Complete reporting does not guarantee low bias; use PROBAST+AI for appraisal.'
    if n == 6:
        return 'Without an actual outcome horizon and independent-hospital evidence, do not claim a 30-day prognostic alert or readiness for deployment.'
    if n == 7:
        return f'At threshold {THRESHOLD:.2f}, {v} alerts correctly identify constructed events, while {c["fn"]} events are missed; verify both error types before considering an alert policy.'
    if n == 8:
        return f'{float(v)*100:.1f}% of constructed events are flagged; the rest are missed. Sensitivity depends on the threshold and does not quantify alert usefulness.'
    if n == 9:
        return f'{float(v)*100:.1f}% of {c["alerts"]} alerts correspond to constructed events; PPV is sensitive to event frequency and case mix.'
    if n == 10:
        return 'AUROC measures discrimination over thresholds, not probability calibration, clinical net benefit, or patient outcome improvement.'
    if n == 11:
        return f'Average precision summarizes the precision–recall tradeoff. The internal test event fraction is {b["y_test"].mean():.3f}; evaluate AP with event frequency and workload in mind.'
    if n == 12:
        return 'Brier score is the mean squared error of estimated probabilities (lower is better); inspect calibration and applicability in addition to this aggregate.'
    if n == 13:
        bins = calibration_table(b['y_test'], b['p_test'])
        return f'In the first nonempty risk bin, {bins.iloc[0]["n"]} records have mean predicted risk {bins.iloc[0]["mean_predicted"]:.3f}, compared with observed rate {float(v):.3f}. Small bins are noisy.'
    if n == 14:
        return f'The absolute descriptive sensitivity gap is {float(v)*100:.1f} percentage points. This teaching split is small and age groups are arbitrary; conduct a powered subgroup audit before making equity claims.'
    if n == 15:
        high = confusion_summary(b['y_test'], b['p_test'], .60)
        return f'At 0.60 there are {v} alerts, versus {c["alerts"]} at 0.40; sensitivity changes from {c["sensitivity"]:.3f} to {high["sensitivity"]:.3f}. Never select a final threshold on the same final test set.'
    if n == 16:
        all_nb = treat_all_net_benefit(b['y_test'], THRESHOLD)
        return f'Model net benefit is {float(v):.4f} per person; treat-all is {all_nb:.4f}, treat-none is 0. Benefit depends on a clinically defensible harm threshold and feasible workflow.'
    if n == 17:
        mod_nb = net_benefit(b['y_test'], b['p_test'], THRESHOLD)
        return f'Treat-all has net benefit {float(v):.4f} at 0.40, versus model {mod_nb:.4f} and treat-none 0; the comparison is educational, not deployment evidence.'
    if n == 18:
        return f'AI-arm minus control risk difference is {float(v)*100:+.1f} percentage points in the fictional randomized counts. Randomization targets causal care impact; a real trial and uncertainty analysis are still required.'
    if n == 19:
        return 'The KS statistic indicates input-distribution change in this simulated batch without requiring new labels. It is not proof of concept drift or worsening clinical performance.'
    if n == 20:
        missing = shift_scenario()[1]['glucose'].isna().mean()
        return f'Incoming glucose missingness is {missing:.1%}, above the fictional 10% quality gate. Pause affected scoring, diagnose the data pipeline, and revalidate; do not automatically retrain.'
    return ''

def check(n, answer):
    """Green 'Correct', then answer and interpretation ONLY when correct."""
    if n not in TUTORIAL_CODE:
        raise ValueError('Tutorial must be 1 through 20')
    ok = _equivalent(answer, _expected()[n])
    if ok:
        try:
            from IPython.display import HTML, display
            display(HTML('<span style="color:#13883d;font-weight:700">Correct</span>'))
        except ImportError:
            print('\033[92mCorrect\033[0m')
        print('Answer:', _format_result(_expected()[n]))
        print('Interpretation:', _interpret(n, _expected()[n]))
    elif answer is None or answer is Ellipsis:
        print('Complete the ____________ code and rerun this cell.')
    else:
        print('Incorrect. Recheck your code against the task and run the cell again.')
    return ok

def show_solution(n):
    """Print one possible student-visible solution only when explicitly called."""
    if n not in TUTORIAL_CODE:
        raise ValueError('Tutorial numbers are 1 through 20')
    print(f'{TUTORIAL_CODE[n]}')

for _n in range(1, 21):
    globals()[f't{_n}_check'] = (lambda ans, i=_n: check(i, ans))
    globals()[f't{_n}_solution'] = (lambda i=_n: show_solution(i))

def teaching_reproducibility():
    b=prediction_bundle()
    assert len(b['train_ids'])+len(b['test_ids']) == 442
    assert not set(b['train_ids']).intersection(set(b['test_ids']))
    assert np.isfinite(b['p_test']).all()
    assert np.all((b['p_test']>=0)&(b['p_test']<=1))
    assert np.isclose(rct_counts().events.sum(),104)
    return {'rows':len(diabetes),'train':len(b['y_train']),'test':len(b['y_test']),
            'constructed_outcome_prevalence':round(diabetes[TARGET].mean(),3),
            'test_prevalence':round(b['y_test'].mean(),3),'seed':42,
            'data_source':'original Day 2 CSV files in data/ folder','unit':'teaching exercise; no clinical deployment'}

def plot_model_comparison():
    """Graphic for Project 2: same patient ordering, different probability scale."""
    d=model_comparison()
    fig,axes=plt.subplots(1,2,figsize=(10,3.6))
    axes[0].scatter(d.model_A,d.model_B,s=14,alpha=.6)
    axes[0].plot([0,1],[0,1],'k--',label='Same probability')
    axes[0].set(title='Two models: same ranking, different risk',xlabel='Model A estimated risk',ylabel='Model B estimated risk',xlim=(0,1),ylim=(0,1))
    for name,ax in [('model_A',axes[1])]:
        t1=calibration_table(d.y,d.model_A);t2=calibration_table(d.y,d.model_B)
        ax.plot([0,1],[0,1],'k--',label='Ideal')
        ax.plot(t1.mean_predicted,t1.observed_rate,'o-',label='A')
        ax.plot(t2.mean_predicted,t2.observed_rate,'s-',label='B')
    axes[1].set(title='Exploratory calibration: test subset',xlabel='Mean predicted risk',ylabel='Observed fraction',xlim=(0,1),ylim=(0,1));axes[1].legend()
    plt.tight_layout();plt.show()

def plot_vendor_trap():
    """Fictional vendor example: no alert despite 50 positive outcomes."""
    fig,ax=plt.subplots(figsize=(6.5,2.9))
    ax.bar(['Actual events (n=50)','Vendor alerts (n=0)'],[50,0])
    ax.set(title='Vendor scenario: events compared with alerts (fictional)',ylabel='People of 1000 encounters',ylim=(0,60))
    plt.tight_layout();plt.show()

def plot_system_workflow():
    """Minimal observable, reversible human-AI architecture for deployment tasks."""
    fig,ax=plt.subplots(figsize=(11,3.2))
    ax.set_xlim(0,11);ax.set_ylim(0,3);ax.axis('off')
    steps=['EHR / source','Schema + time\nvalidation','Versioned\nmodel API','Clinician\nreview','Action +\nfollow-up']
    centers=np.linspace(1,10,5)
    for i,(x,label) in enumerate(zip(centers,steps)):
        ax.text(x,2.2,label,ha='center',va='center',fontsize=10,bbox={'boxstyle':'round,pad=0.5','fc':'#eff6ff','ec':'#2563eb'})
        if i<4:ax.annotate('',xy=(centers[i+1]-.72,2.2),xytext=(x+.72,2.2),arrowprops={'arrowstyle':'->','lw':1.5})
    ax.text(5.5,.60,'Audit logs · input/score/alert monitoring · named safety owner · pause/rollback route',ha='center',va='center',fontsize=10,bbox={'boxstyle':'round,pad=0.6','fc':'#f1f5f9','ec':'#64748b'})
    ax.annotate('',xy=(5.5,1.0),xytext=(5.5,1.85),arrowprops={'arrowstyle':'<->','lw':1.4,'color':'#475569'})
    ax.set_title('Human-in-the-loop deployment: observed and reversible (conceptual)',fontsize=12)
    plt.tight_layout();plt.show()


# Optional independent project/challenge code samples. These are printed only when requested.
# This section does not change any Tutorial 1–20 code, examples, grading or interpretations.
POSSIBLE_INDEPENDENT_SOLUTIONS = {
    "P1": '''allowed = feature_timeline["day_relative_to_score"] <= 0
report = feature_timeline[["variable", "day_relative_to_score"]].copy()
report["allowed_by_time"] = allowed
missing = [key for key, value in REPORT_FIELDS.items() if value is None]
display(report)
print("Missing report details:", missing)
print("DECISION: HOLD a 30-day claim. The teaching label has no genuine 30-day outcome/horizon;")
print("report prediction studies with TRIPOD+AI; appraise bias using PROBAST+AI.")
print("NEXT: define a real patient cohort, prediction time, 30-day outcome and external validation.")''',

    "P2": '''d = model_comparison()
rows = []
for name in ["model_A", "model_B"]:
    y = d["y"].to_numpy()
    p = d[name].to_numpy()
    rows.append({"model": name,
                 "AUROC": roc_auc_score(y, p),
                 "Brier": brier_score_loss(y, p),
                 "alerts_at_0.40": confusion_summary(y, p, 0.40)["alerts"],
                 "net_benefit_at_0.40": net_benefit(y, p, 0.40)})
result = pd.DataFrame(rows)
display(result.round(3))
print("DECISION: Model A is preferable FOR FURTHER VALIDATION: same AUROC, lower Brier,")
print("fewer alerts and higher net benefit at the example threshold. Neither is deployable evidence.")''',

    "P3": '''d = decision_curve(bundle["y_test"], bundle["p_test"], [.40, .55, .60, .65])
d["within_50_alerts"] = d["alerts"] <= 50
d["better_than_defaults"] = (d["model"] > d["treat_all"]) & (d["model"] > d["treat_none"])
display(d.round(3))
print("DECISION: HOLD deployment at 0.40 (79 alerts > 50).")
print("0.65 fits the cap in the teaching set, but selection on the final test set is optimistic.")
print("NEXT: choose an action threshold clinically, lock it, then externally validate and silent-test.")''',

    "P4": '''tab = subgroup_table(bundle["y_test"], bundle["p_test"], bundle["age_group"])
display(tab.round(3))
older = tab.loc[tab["group"] == "older", "sensitivity"].iloc[0]
younger = tab.loc[tab["group"] == "younger", "sensitivity"].iloc[0]
gap = older - younger
print(f"Older minus younger sensitivity: {gap:+.3f} ({100*gap:+.1f} percentage points)")
print("DECISION: This small arbitrary split suggests a subgroup audit, not proof of fairness")
print("or discrimination. Involve clinical and equity reviewers and validate meaningful groups.")''',

    "P5": '''baseline, incoming = shift_scenario()
display(drift_report(baseline, incoming).round(3))
missing = incoming["glucose"].isna().mean()
print(f"Incoming glucose missing: {100*missing:.1f}%")
print("Quality gate:", triage_rule(missing, max_missing=.10))
print("DECISION: PAUSE affected scoring, inspect upstream EHR/preprocessing with the clinical")
print("safety owner, repair data flow and independently revalidate before restart. Do not")
print("automatically retrain; unlabeled input shift is not proof of model failure.")''',

    "P6": '''trial = rct_counts().set_index("arm")
p_usual = trial.loc["usual_care", "events"] / trial.loc["usual_care", "patients"]
p_ai = trial.loc["ai_assisted", "events"] / trial.loc["ai_assisted", "patients"]
rd = p_ai - p_usual
rr = p_ai / p_usual
se = np.sqrt(p_ai*(1-p_ai)/500 + p_usual*(1-p_usual)/500)
ci_low, ci_high = rd - 1.96*se, rd + 1.96*se
summary = pd.DataFrame({"p_usual": [p_usual], "p_ai": [p_ai],
                        "RD_ai_minus_usual": [rd], "risk_ratio": [rr],
                        "approx_RD_CI_low": [ci_low], "approx_RD_CI_high": [ci_high]})
display(summary.round(3))
print("DECISION: Fictional AI event risk is lower (8.8% versus 12.0%) but the")
print("approximate risk-difference interval includes 0; this does not establish a true benefit.")
print("A real adequately powered randomized impact/safety evaluation is necessary.")''',

    "P7": '''audit = synthetic_llm_audit()
flags = ["clinically_correct", "unsupported_claim", "unsafe_action"]
summary = pd.DataFrame({"flag": flags,
                        "count": [int(audit[f].sum()) for f in flags],
                        "rate": [float(audit[f].mean()) for f in flags]})
display(summary)
print("DECISION: No autonomous clinical actions; restrict to clinician-reviewed drafts.")
print("Use source verification, human approval before action, and auditable action/error logs.")
print("Report evaluation details with TRIPOD-LLM; these 12 cases are only illustrative.")''',

    "C1": '''y_true = np.array([1]*50 + [0]*950)
y_pred = np.zeros(1000, dtype=int)
accuracy = (y_true == y_pred).mean()
sensitivity = ((y_pred == 1) & (y_true == 1)).sum() / 50
specificity = ((y_pred == 0) & (y_true == 0)).sum() / 950
alerts = int(y_pred.sum())
print({"accuracy": accuracy, "sensitivity": sensitivity,
       "specificity": specificity, "alerts": alerts, "PPV": "undefined (0 alerts)"})
fig, ax = plt.subplots(figsize=(5, 3))
ax.bar(["Accuracy", "Sensitivity"], [100*accuracy, 100*sensitivity])
ax.set(ylabel="Percent (%)", ylim=(0,100), title="Accuracy hides missed cases")
plt.show()
print("DECISION: HOLD procurement; a system that misses all 50 events is unusable as")
print("a sepsis detector despite 95% accuracy. Request valid prospective labels/timing,")
print("independent-hospital validation, and sensitivity/false-alert plus subgroup safety.")
print("Reference: https://www.bmj.com/content/385/bmj-2023-078378")''',

    "C2": '''# Run the supplied Challenge 2 evidence cell first to create pilot_data.
summary = pilot_data.copy()
summary["timely_pct"] = 100*summary["reviewed_within_30min"]/summary["alerts"]
summary["weekly_gate_pass"] = summary["timely_pct"] >= 90
display(summary)
overall_timely = 100 * summary["reviewed_within_30min"].sum()/summary["alerts"].sum()
failures = int(summary["software_failures"].sum())
print(f"Overall timely review = {overall_timely:.1f}%; software failures = {failures}")
print("DECISION: HOLD expansion (81.3% < 90% and 1 > 0 failures).")
print("Improve review routing/staff coverage; repair and retest the failure condition;")
print("clinical safety owner must sign off before a re-evaluated pilot expansion.")
print("A supervised pilot does NOT prove improved patient outcomes.")''',

    "C3": '''# Run the supplied Challenge 3 evidence cell first to create incident_data.
summary = incident_data.copy()
summary["missing_glucose_pct"] = 100*summary["missing_glucose"]/summary["records"]
display(summary)
before, after = summary.iloc[0], summary.iloc[1]
delta_pp = after["missing_glucose_pct"] - before["missing_glucose_pct"]
print(f"Missing glucose: {before['missing_glucose_pct']:.1f}% → {after['missing_glucose_pct']:.1f}%")
print(f"Increase = {delta_pp:+.1f} percentage points; lost alerts = {int(after['undelivered_alerts'])}")
print("DECISION: PAUSE affected alerting and use clinician fallback: both the >10%")
print("missingness gate and any-undelivered-alert rule were breached.")
print("Clinical safety lead owns triage; EHR interface team checks input mapping AND")
print("delivery acknowledgments/log reconciliation. Communicate to clinical staff.")
print("RESTART only when glucose missingness is <=10%, there are 0 undelivered")
print("alerts, and the repaired workflow passes clinical sign-off/revalidation.")
print("No new labels means patient outcome performance cannot yet be confirmed.")''',
}

def show_project_solution(project):
    'Print one possible working project solution, only on explicit learner request.'
    key = str(project).upper().strip()
    if key not in {f"P{i}" for i in range(1, 8)}:
        raise ValueError("Use P1 through P7")
    print("One possible solution for " + key + ":\n" + POSSIBLE_INDEPENDENT_SOLUTIONS[key])

def show_challenge_solution(challenge):
    'Print one possible working challenge solution, only on explicit learner request.'
    key = str(challenge).upper().strip()
    if key not in {f"C{i}" for i in range(1, 4)}:
        raise ValueError("Use C1 through C3")
    print("One possible solution for " + key + ":\n" + POSSIBLE_INDEPENDENT_SOLUTIONS[key])
