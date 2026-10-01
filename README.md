# -us-stock-expectancy
[README.md](https://github.com/user-attachments/files/32889821/README.md)
# US Stock Expectancy V3 — Mobile Web

手機優先的 Streamlit 網站版本。

## 本機啟動
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 手機使用
部署後會得到 `https://xxx.streamlit.app` 網址，iPhone/Android 直接用 Safari/Chrome 開啟即可。也可加入手機主畫面。

## 免費部署
Streamlit Community Cloud 可從 GitHub repository 建立 app，部署後會產生可分享的 `streamlit.app` 網址。

## V3
- 手機優先 CSS
- 單股分析
- S1/R1
- MA20/60/120
- Risk/Reward
- 量能
- 自選股並行掃描
- 歷史回測
- 勝率/平均獲利/平均虧損/期望值/最大回撤
- CSV 匯出
[app.py](https://github.com/user-attachments/files/32889828/app.py)
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed

st.set_page_config(page_title='US Stock Expectancy V3', page_icon='📈', layout='wide', initial_sidebar_state='collapsed')

st.markdown('''<style>
.block-container{padding:1rem 1rem 2rem;max-width:1200px}
@media(max-width:700px){.block-container{padding:.7rem .55rem 1.5rem}.stMetric{padding:.35rem}.stButton button{width:100%;min-height:48px;font-size:16px}}
[data-testid="stMetricValue"]{font-size:1.25rem}
</style>''', unsafe_allow_html=True)

@st.cache_data(ttl=900, show_spinner=False)
def data(ticker, period='5y'):
    try:
        d=yf.download(ticker,period=period,interval='1d',auto_adjust=True,progress=False,threads=False)
        if d is None or d.empty:return pd.DataFrame()
        if isinstance(d.columns,pd.MultiIndex):d.columns=d.columns.get_level_values(0)
        d.columns=[str(x).title() for x in d.columns]
        d=d[['Open','High','Low','Close','Volume']].dropna().copy()
        for n in [5,10,20,60,120]:d[f'MA{n}']=d.Close.rolling(n).mean()
        prev=d.Close.shift(1)
        tr=pd.concat([d.High-d.Low,(d.High-prev).abs(),(d.Low-prev).abs()],axis=1).max(axis=1)
        d['ATR14']=tr.rolling(14).mean(); d['VolMA20']=d.Volume.rolling(20).mean()
        return d.dropna()
    except Exception:return pd.DataFrame()

def piv(s,kind):
    a=s.values;o=[]
    for i in range(3,len(a)-3):
        w=a[i-3:i+4]
        if kind=='low' and a[i]==min(w) and a[i]<min(a[i-3:i]):o.append((i,float(a[i])))
        if kind=='high' and a[i]==max(w) and a[i]>max(a[i+1:i+4]):o.append((i,float(a[i])))
    return o

def levels(d,tol=.012):
    p=float(d.Close.iloc[-1]);
    def cluster(points):
        cs=[]
        for i,v in sorted(points,key=lambda x:x[1]):
            hit=next((c for c in cs if abs(v-c[0])/c[0]<=tol),None)
            if hit: hit[1].append(v);hit[0]=float(np.mean(hit[1]))
            else:cs.append([v,[v]])
        return cs
    s=sorted([c[0] for c in cluster(piv(d.Low,'low')) if c[0]<p],reverse=True)
    r=sorted([c[0] for c in cluster(piv(d.High,'high')) if c[0]>p])
    return s,r

def analyze(ticker,near=.03,rrmin=1.5,tol=.012):
    d=data(ticker)
    if len(d)<130:return None
    x=d.iloc[-1];s,r=levels(d,tol)
    s1=s[0] if s else np.nan;r1=r[0] if r else np.nan
    risk=(x.Close-s1)/x.Close if np.isfinite(s1) else np.nan
    reward=(r1-x.Close)/x.Close if np.isfinite(r1) else np.nan
    rr=reward/risk if np.isfinite(risk) and risk>0 and np.isfinite(reward) else np.nan
    trend=bool(x.Close>x.MA20>x.MA60>x.MA120)
    return {'Ticker':ticker,'Price':float(x.Close),'S1':s1,'R1':r1,'Risk':risk,'Reward':reward,'RR':rr,'Trend':trend,'Near':bool(np.isfinite(risk) and risk<=near),'RROK':bool(np.isfinite(rr) and rr>=rrmin),'Vol':float(x.Volume/x.VolMA20),'MA20':float(x.MA20),'MA60':float(x.MA60),'MA120':float(x.MA120)}

def backtest(d,near=.03,rr=1.5,horizon=20):
    rows=[]
    for i in range(130,len(d)-horizon-1):
        h=d.iloc[:i+1];p=float(h.Close.iloc[-1]);s,r=levels(h)
        if not s or not r or not(p>h.MA20.iloc[-1]>h.MA60.iloc[-1]>h.MA120.iloc[-1]):continue
        stop=s[0];
        if (p-stop)/p>near:continue
        ent=float(d.Open.iloc[i+1]);target=ent+rr*(ent-stop);ex=None;out='timeout'
        for j in range(i+1,min(i+1+horizon,len(d))):
            lo,hi=float(d.Low.iloc[j]),float(d.High.iloc[j])
            if lo<=stop:ex=stop;out='loss';break
            if hi>=target:ex=target;out='win';break
        if ex is None:ex=float(d.Close.iloc[min(i+horizon,len(d)-1)])
        rows.append({'Date':d.index[i],'Entry':ent,'Exit':ex,'Return':ex/ent-1,'Outcome':out})
    return pd.DataFrame(rows)

st.title('📈 美股交易期望值 V3')
st.caption('手機優先介面｜單股分析｜條件掃描｜歷史回測｜期望值')

with st.sidebar:
    st.header('設定')
    period=st.selectbox('歷史資料',['2y','5y','10y','max'],1)
    near=st.slider('距離支撐 S1 最大',1,10,3)/100
    rrmin=st.number_input('最低 Risk/Reward',0.5,5.0,1.5,.1)
    tol=st.slider('支撐/壓力聚類',.5,3.,1.2,.1)/100
    horizon=st.slider('最長持有日',5,60,20,5)

single,scan,test=st.tabs(['🔎 單股','🚀 掃描','📊 回測'])
with single:
    t=st.text_input('美股代號','AAPL').upper().strip()
    c1,c2=st.columns(2)
    entry=c1.number_input('計畫進場價',.01,100000.,250.,.01)
    capital=c2.number_input('投入資金',0.,100000000.,10000.,100.)
    if st.button('開始分析',type='primary'):
        a=analyze(t,near,rrmin,tol)
        if not a:st.error('找不到足夠歷史資料。')
        else:
            cols=st.columns(4)
            cols[0].metric('現價',f"${a['Price']:,.2f}");cols[1].metric('S1',f"${a['S1']:,.2f}" if np.isfinite(a['S1']) else '—');cols[2].metric('R1',f"${a['R1']:,.2f}" if np.isfinite(a['R1']) else '—');cols[3].metric('R/R',f"1:{a['RR']:.2f}" if np.isfinite(a['RR']) else '—')
            cols=st.columns(4);cols[0].metric('下行風險',f"{a['Risk']*100:.2f}%" if np.isfinite(a['Risk']) else '—');cols[1].metric('上行潛力',f"{a['Reward']*100:.2f}%" if np.isfinite(a['Reward']) else '—');shares=int(capital/entry);cols[2].metric('預估股數',f'{shares:,}');cols[3].metric('20日量能',f"{a['Vol']:.2f}x")
            st.write(f"均線結構：MA20 ${a['MA20']:.2f}｜MA60 ${a['MA60']:.2f}｜MA120 ${a['MA120']:.2f}")
            st.write(f"條件：趨勢 {'✓' if a['Trend'] else '✗'}｜接近支撐 {'✓' if a['Near'] else '✗'}｜R/R {'✓' if a['RROK'] else '✗'}")
            d=data(t,period);st.line_chart(d[['Close','MA20','MA60','MA120']].tail(250))

with scan:
    st.write('手機也能操作：貼入自選股代號，每行一檔。')
    raw=st.text_area('股票清單','AAPL\nMSFT\nNVDA\nAMZN\nMETA\nGOOGL\nTSLA\nAVGO\nAMD\nNFLX')
    workers=st.slider('並行數',1,10,5)
    if st.button('開始掃描',type='primary'):
        syms=list(dict.fromkeys(x.strip().upper() for x in raw.splitlines() if x.strip()));out=[];bar=st.progress(0)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            fs={ex.submit(analyze,s,near,rrmin,tol):s for s in syms}
            for n,f in enumerate(as_completed(fs),1):
                try:
                    z=f.result()
                    if z:out.append(z)
                except:pass
                bar.progress(n/len(fs))
        df=pd.DataFrame(out)
        if df.empty:st.warning('沒有有效資料。')
        else:
            hit=df[df.Trend&df.Near&df.RROK].sort_values(['RR','Reward'],ascending=False)
            st.metric('符合條件',len(hit));st.dataframe(hit[['Ticker','Price','S1','R1','Risk','Reward','RR','Vol']],use_container_width=True,hide_index=True)
            st.download_button('下載 CSV',hit.to_csv(index=False).encode('utf-8-sig'),'v3_scan.csv','text/csv')

with test:
    t2=st.text_input('回測代號','AAPL',key='bt').upper().strip()
    if st.button('執行回測',type='primary'):
        d=data(t2,period);bt=backtest(d,near,rrmin,horizon)
        if bt.empty:st.warning('沒有符合條件的歷史案例。')
        else:
            w=bt.Return>0;wr=w.mean();aw=bt.loc[w,'Return'].mean() if w.any() else 0;al=-bt.loc[~w,'Return'].mean() if (~w).any() else 0;ex=wr*aw-(1-wr)*al;eq=(1+bt.Return).cumprod();dd=(eq/eq.cummax()-1).min()
            c=st.columns(5);c[0].metric('案例',len(bt));c[1].metric('勝率',f'{wr*100:.2f}%');c[2].metric('平均獲利',f'{aw*100:.2f}%');c[3].metric('平均虧損',f'{al*100:.2f}%');c[4].metric('期望值',f'{ex*100:.2f}%');st.metric('最大回撤',f'{dd*100:.2f}%')
            st.dataframe(bt.tail(100),use_container_width=True,hide_index=True)
            st.download_button('下載回測 CSV',bt.to_csv(index=False).encode('utf-8-sig'),f'{t2}_backtest.csv','text/csv')

st.divider();st.caption('V3 可部署成手機瀏覽器直接開啟的網址。歷史資料來自 Yahoo Finance / yfinance；正式交易前應加入正式行情供應商、滑價與交易成本。')
