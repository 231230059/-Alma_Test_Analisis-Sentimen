from __future__ import annotations  
import io  
import json  
import os  
import re  
import joblib  
import matplotlib.pyplot as plt  
import numpy as np  
import pandas as pd  
import plotly.express as px  
import plotly.graph_objects as go  
import streamlit as st  
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve) 
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import label_binarize  
from wordcloud import WordCloud  

# Folder = tempat file ini berada. 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  

# Sesuaikan nama file CSV Anda di sini
MODEL_FILE = "model_nb_sentimen.joblib"
DATA_FILE = "Alma_Test_Analisis Sentimen.csv" 
STEM_FILE = "stem_cache.json"

SEED_TEST_SIZE = 42, 0.2 
COLORS = {"Positif": "#10B981", "Netral": "#F59E0B", "Negatif": "#EF4444"} 

st.set_page_config(page_title="Sentiment Mining · Ruangguru", page_icon="📊", layout="wide") 

st.markdown(""" 
<style> 
.hero{background:linear-gradient(120deg,#4F46E5 0%,#7C3AED 55%,#EC4899 100%);padding:26px 32px; border-radius:18px;color:white;margin-bottom:16px} 
.hero h1{margin:0;font-size:1.9rem;color:white} 
.hero p{margin:6px 0 0;opacity:.92} 
.kpi{background:var(--secondary-background-color);border-radius:14px;padding:14px 18px;border-left:6px solid #7C3AED}  
.kpi .v{font-size:1.65rem;font-weight:700} 
.kpi .l{font-size:0.78rem;opacity:.7;text-transform:uppercase;letter-spacing:.05em}  
.insight{background:var(--secondary-background-color);border-radius:12px;padding:10px 16px;margin:6px 0;border-left:4px solid #EC4899}  
.stTabs [data-baseweb="tab"]{font-weight:600}  
</style>  

<div class="hero">
<h1>Sentiment Mining Dashboard Ruangguru</h1> 
<p>Hasil analisis notebook dengan <b>Multinomial Naive Bayes</b> . TF-IDF . SMOTE . Influential Features . Analisis Aspek</p>
</div>  
""", unsafe_allow_html=True) 

def kpi(col, label, value, color="#7C3AED"): 
    col.markdown(f'''
    <div class="kpi" style="border-left-color:{color}">
        <div class="l">{label}</div>
        <div class="v">{value}</div>
    </div>
    ''', unsafe_allow_html=True) 

def _find(name): 
    p = os.path.join(BASE_DIR, name) 
    return p if os.path.exists(p) else None 

with st.sidebar: 
    st.header("Hasil Notebook") 
    up_model = st.file_uploader("Model (joblib) - opsional", type="joblib", help="Isi hanya jika file tidak ada di folder aplikasi.") 
    up_data = st.file_uploader("Data bersih (.csv) - opsional", type="csv") 
    model_src = up_model.getvalue() if up_model else _find(MODEL_FILE) 
    data_src = up_data.getvalue() if up_data else _find(DATA_FILE) 
    
    if model_src is None or data_src is None: 
        st.error("File hasil notebook belum ditemukan.") 
        st.markdown(f"""
        Letakkan file berikut **satu folder dengan `app.py`**: 
        - `{MODEL_FILE}` 
        - `{DATA_FILE}` 
        """) 
        st.stop() 

@st.cache_resource(show_spinner="Memuat model...") 
def load_bundle(src): 
    return joblib.load(io.BytesIO(src) if isinstance(src, bytes) else src) 

@st.cache_data(show_spinner="Memuat data...") 
def load_data(src): 
    d = pd.read_csv(io.BytesIO(src) if isinstance(src, bytes) else src) 
    # Pastikan kolom yang dibutuhkan ada
    if "clean_content" not in d.columns and "content" in d.columns:
        d["clean_content"] = d["content"].astype(str) # Fallback jika clean_content tidak ada
    else:
        d["clean_content"] = d["clean_content"].fillna("").astype(str) 
    d["at"] = pd.to_datetime(d.get("at"), errors="coerce") if "at" in d.columns else pd.NaT 
    return d 

bundle = load_bundle(model_src) 
df = load_data(data_src) 

vec, model = bundle["vectorizer"], bundle["model"] 

# --- PENGAMAN 1: Classes ---
if hasattr(model, "classes_"):
    classes = list(model.classes_)
elif "classes" in bundle:
    classes = list(bundle["classes"])
else:
    classes = ["Negatif", "Netral", "Positif"]

bp = bundle.get("best_params", {}) 

# --- PENGAMAN 2: Metrik Evaluasi ---
M = bundle.get("metrics", {})
for key in ["Accuracy", "Precision (macro)", "Recall (macro)", "F1-Score (macro)", "ROC-AUC (macro)"]:
    if key not in M:
        M[key] = 0.0 # Isi 0 jika tidak ada di joblib

ev = bundle.get("evaluation", {
    "n_tr": 0, "n_te": 0, 
    "report": pd.DataFrame(columns=["precision", "recall", "f1-score"]), 
    "cm": np.zeros((len(classes), len(classes))), 
    "roc": {}, "errors": pd.DataFrame(), "X_tr": [], "y_tr": []
})

# Pastikan report memiliki f1-score
if "report" in ev and not ev["report"].empty:
    if "f1-score" not in ev["report"].columns and "f1_score" in ev["report"].columns:
        ev["report"]["f1-score"] = ev["report"]["f1_score"]

fitur = vec.get_feature_names_out()

# --- PENGAMAN 3: Feature Log Prob ---
if hasattr(model, "feature_log_prob_"):
    flp = model.feature_log_prob_
else:
    flp = np.zeros((len(classes), len(fitur)))

cmap = COLORS

# --- Fungsi Bantu Preprocessing ---
STOPWORDS = {  
    "di", "dari", "ke", "pada", "dengan", "untuk", "oleh", "tentang", "kepada", "yang", "dan", "atau", "tetapi", "tapi",  
    "karena", "meskipun", "jika", "agar", "saya", "aku", "kamu", "dia", "kita", "mereka", "ini", "itu", "lah", "tuh", "nya",  
    "dong", "aja", "gitu", "deh", "emang", "ya", "sih", "kok", "lho", "nih", "si", "pun", "juga", "sudah", "akan", "lagi",  
    "pernah", "hanya", "adalah", "dalam", "ada", "para", "sebagai", "saat", "ketika", "kalau", "kalo", "jadi", "sama", "mau",  
}  
NEGATIONS = {"tidak", "bukan", "belum", "jangan", "kurang", "gak", "ga", "nggak", "ngga", "tak", "gk", "engkak"}  
DOMAIN_STOPWORDS = {"ruangguru", "rg", "ruang", "kak"}  
SLANG = {  
    "gak": "tidak", "ga": "tidak", "nggak": "tidak", "ngga": "tidak", "gk": "tidak", "engkak": "tidak", "tdk": "tidak",  
    "tak": "tidak", "bgt": "banget", "bgus": "bagus", "bagu": "bagus", "bagust": "bagus", "mantul": "mantap", "udh": "sudah",  
    "udah": "sudah", "sdh": "sudah", "blm": "belum", "yg": "yang", "dgn": "dengan", "krn": "karena", "karna": "karena",  
    "utk": "untuk", "jd": "jadi", "tp": "tapi", "tpi": "tapi", "klo": "kalau", "kalo": "kalau", "sy": "saya", "sm": "sama",  
    "sma": "sama", "bs": "bisa", "bsa": "bisa", "gabisa": "tidak bisa", "lemot": "lambat", "lag": "lambat",  
    "ngelag": "lambat", "eror": "error", "erorr": "error", "apk": "aplikasi", "app": "aplikasi", "aplikasinya": "aplikasi",  
    "thx": "terima kasih", "makasih": "terima kasih", "terimakasih": "terima kasih", "trims": "terima kasih", "oke": "ok",  
    "okey": "ok", "recommended": "recommended", "rekomen": "recommended", "kreen": "kreen", "jelek": "buruk", "jlek": "buruk",  
    "pake": "pakai", "pke": "pakai", "jig": "juga", "dr": "dari", "dl": "dulu", "bgmn": "bagaimana", "gmn": "bagaimana",  
    "knp": "kenapa", "kyk": "seperti", "bnyk": "banyak", "bebrpa": "beberapa", 
}  

_STEM_PATH = os.path.join(BASE_DIR, STEM_FILE)  
_stem_cache: dict = {}  
if os.path.exists(_STEM_PATH):  
    try:  
        _stem_cache = json.load(open(_STEM_PATH, encoding='utf-8'))  
    except (OSError, ValueError):  
        _stem_cache = {}  

@st.cache_resource(show_spinner="Memuat stemmer Sastrawi...")  
def get_stemmer():  
    try:  
        from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
        return StemmerFactory().create_stemmer()
    except ImportError:  
        return None  

def stem_kata(w):  
    if w not in _stem_cache:  
        s = get_stemmer()  
        _stem_cache[w] = s.stem(w) if s else w  
    return _stem_cache[w]  

def bersihkan_teks(teks):  
    t = str(teks).lower()  
    t = re.sub(r"http\S+", "", t)
    t = re.sub(r"\brg\b", "ruangguru", t)
    t = re.sub(r"[^a-z\s]", "", t)
    tokens = [SLANG.get(w, w) for w in t.split()]
    tokens = [w for w in tokens if (w not in STOPWORDS or w in NEGATIONS) and w not in DOMAIN_STOPWORDS and len(w) > 1]  
    return " ".join(stem_kata(w) for w in tokens)  

def simpan_stem_cache():  
    try:  
        json.dump(_stem_cache, open(_STEM_PATH, "w", encoding="utf-8"), ensure_ascii=False)  
    except OSError:  
        pass  

ASPEK = {  
    "Mentor / Guru": r"guru|mentor|pengajar|tutor|kakak|penjelasan|menjelaskan", 
    "Materi & Soal": r"materi|soal|latihan|rangkuman|pembahasan|kuis|pelajaran|kurikulum|bab"
}

def influential(top_n=15):  
    rows = []  
    for i, c in enumerate(classes):  
        diff = flp[i] - np.delete(flp, i, axis=0).mean(axis=0)  
        for rank, j in enumerate(np.argsort(diff)[::-1][:top_n], 1):  
            rows.append({"Kelas": c, "Rank": rank, "Fitur": fitur[j], "Skor Pengaruh": diff[j],  
                         "Rasio Odds": float(np.exp(diff[j])), "log P(kata|kelas)": flp[i, j]})  
    return pd.DataFrame(rows)  

def prediksi(teks, top_k=8): 
    bersih = bersihkan_teks(teks) 
    if not bersih.strip(): return None 
    X = vec.transform([bersih]) 
    p = model.predict_proba(X)[0] 
    order = np.argsort(p)[::-1] 
    pred = classes[order[0]]
    ref = classes[order[1]] if len(order) > 1 else classes[order[0]]
    
    kontrib = [{"Kata": fitur[j], "TF-IDF": float(X[0, j]),  
                f"Dorongan ke {pred}": float((flp[classes.index(pred), j] - flp[classes.index(ref), j]) * X[0, j])} 
               for j in X.nonzero()[1]] 
    cdf = pd.DataFrame(kontrib) 
    if not cdf.empty: 
        cdf = cdf.sort_values(cdf.columns[-1], ascending=False).head(top_k).reset_index(drop=True) 
    return {"clean": bersih, "pred": pred, "proba": dict(zip(classes, p)), "contrib": cdf}  

def aspect_table(df, total): 
    return pd.DataFrame() # Kosongkan karena fungsi asli tidak ada di dokumen

# ==============================================================================
tabs = st.tabs(["📊 Ringkasan", "📈 Eksplorasi Data", "☁️ Analisis Teks", "🎯 Evaluasi Model", "🌟 Influential Features", "🔍 Aspek & Anomali", "🔮 Prediksi", "📚 Metodologi"])  

# --- RINGKASAN ---
with tabs[0]:
    c = st.columns(5) 
    for col, (k, v), color in zip(c, [(k, M[k]) for k in ["Accuracy", "Precision (macro)", "Recall (macro)", "F1-Score (macro)", "ROC-AUC (macro)"]], ["#4F46E5", "#10B981", "#F59E0B", "#EC4899", "#06B6D4"]): 
        kpi(col, k, f"{v*100:.1f}%" if "AUC" not in k else f"{v:.3f}", color) 
    
    st.write("") 
    c = st.columns(4) 
    if "score" in df.columns:
        kpi(c[0], "Total Ulasan", f"{len(df):,}") 
        kpi(c[1], "Rata-rata Rating", f"{df['score'].mean():.2f} ⭐", "#F59E0B") 
    if "label" in df.columns:
        kpi(c[2], "% Positif", f"{(df['label'] == 'Positif').mean()*100:.1f}%", "#10B981") 
        kpi(c[3], "% Negatif", f"{(df['label'] == 'Negatif').mean()*100:.1f}%", "#EF4444")  

    st.subheader("💡 Temuan Utama") 
    insights = []
    try:
        rep = ev["report"].loc[classes] 
        best_c = rep["f1-score"].idxmax()
        worst_c = rep["f1-score"].idxmin() 
        insights.append(f"Model paling baik mengenali kelas **{best_c}** (F1 = {rep.loc[best_c, 'f1-score']:.2f}) dan paling lemah pada kelas **{worst_c}** (F1 = {rep.loc[worst_c, 'f1-score']:.2f}).")
    except Exception:
        insights.append("Data *Classification Report* tidak ditemukan di dalam file `.joblib`.")

    if df["at"].notna().any():  
        yr = df.dropna(subset=["at"]).groupby(df["at"].dt.year)["score"].agg(["mean", "size"])  
        yr = yr[yr["size"] >= 20]  
        if len(yr):  
            insights.append(f"Rata-rata rating tertinggi tercatat pada tahun **{int(yr['mean'].idxmax())}** ({yr['mean'].max():.2f}) dan terendah pada **{int(yr['mean'].idxmin())}** ({yr['mean'].min():.2f}).")  
    
    if "label" in df.columns:
        imb = df["label"].value_counts(normalize=True)  
        insights.append(f"Data **tidak seimbang**: kelas {imb.idxmax()} mendominasi {imb.max()*100:.0f}%, sehingga **F1-macro ({M.get('F1-Score (macro)', 0):.2f})** lebih jujur daripada akurasi ({M.get('Accuracy', 0):.2f}).")  
    
    for t in insights:  
        st.markdown(f'<div class="insight">{t}</div>', unsafe_allow_html=True)  

    a, b = st.columns(2)  
    if "label" in df.columns:
        lb = df["label"].value_counts().reset_index(); lb.columns = ["Sentimen", "Jumlah"] 
        a.plotly_chart(px.pie(lb, names="Sentimen", values="Jumlah", hole=.55, color="Sentimen", color_discrete_map=cmap, title="Distribusi Sentimen"), use_container_width=True)  
    
    if not ev["report"].empty:
        try:
            rp = rep[["precision", "recall", "f1-score"]].reset_index().melt("index", var_name="Metrik", value_name="Nilai")  
            b.plotly_chart(px.bar(rp, x="index", y="Nilai", color="Metrik", barmode="group", range_y=[0, 1], title="Kinerja per Kelas (data uji)", labels={"index": "Kelas"}), use_container_width=True)
        except Exception:
            pass

# --- EKSPLORASI ---
with tabs[1]:  
    d = df.copy()  
    f1c, f2c = st.columns(2)  
    if df["at"].notna().any():  
        yrs = sorted(df["at"].dropna().dt.year.unique())  
        yr_sel = f1c.select_slider("Rentang tahun", options=yrs, value=(yrs[0], yrs[-1]))  
        d = d[d["at"].dt.year.between(*yr_sel) | d["at"].isna()]  
    
    if "label" in df.columns:
        lab_sel = f2c.multiselect("Sentimen", classes, default=classes)  
        d = d[d["label"].isin(lab_sel)]  
    st.caption(f"{len(d):,} ulasan terpilih")  
    
    a, b = st.columns(2)  
    if "score" in d.columns:
        sc = d["score"].value_counts().sort_index().reset_index() 
        sc.columns = ["Rating", "Jumlah"]  
        a.plotly_chart(px.bar(sc, x="Rating", y="Jumlah", text="Jumlah", color="Rating", color_continuous_scale="Viridis", title="Distribusi Rating"), use_container_width=True)  
    
    if "content" in d.columns:
        d["panjang_kata"] = d["content"].astype(str).str.split().str.len()  
        b.plotly_chart(px.histogram(d, x="panjang_kata", color="label" if "label" in d.columns else None, nbins=30, barmode="overlay", opacity=.7, color_discrete_map=cmap, title="Panjang Ulasan (kata)"), use_container_width=True)  
    
    if d["at"].notna().any() and "label" in d.columns:  
        dd = d.dropna(subset=["at"]).copy()  
        dd["Tahun"] = dd["at"].dt.year  
        a, b = st.columns(2)  
        t = dd.groupby(["Tahun", "label"]).size().reset_index(name="Jumlah") 
        a.plotly_chart(px.bar(t, x="Tahun", y="Jumlah", color="label", color_discrete_map=cmap, title="Jumlah Ulasan per Tahun"), use_container_width=True) 
        
        if "score" in d.columns:
            r = dd.groupby(["Tahun"])["score"].agg(["mean", "size"]).reset_index() 
            r.columns = ["Tahun", "Rating", "Jumlah"]  
            fig = px.line(r, x="Tahun", y="Rating", markers=True, hover_data=["Jumlah"], title="Rata-rata Rating per Tahun")  
            fig.update_traces(line_color="#7C3AED"); fig.update_yaxes(range=[1, 5]) 
            b.plotly_chart(fig, use_container_width=True)  
        
    st.subheader("Tabel Ulasan")  
    q = st.text_input("Cari kata di ulasan")  
    if "content" in d.columns:
        tb = d[d["content"].astype(str).str.contains(q, case=False, regex=False)] if q else d  
        st.dataframe(tb[[c for c in ["at", "score", "label", "content"] if c in tb.columns]].head(300), use_container_width=True)  

# --- TEKS ---
with tabs[2]:  
    pick = st.selectbox("WordCloud untuk", ["Semua"] + classes)  
    if "clean_content" in df.columns and "label" in df.columns:
        txt = "".join(df["clean_content"] if pick == "Semua" else df.loc[df["label"] == pick, "clean_content"])  
        if txt.strip():  
            cm_name = {"Negatif": "Reds", "Netral": "Oranges", "Positif": "Greens"}.get(pick, "viridis")  
            wc = WordCloud(width=1000, height=420, background_color="white", colormap=cm_name, max_words=100, random_state=SEED_TEST_SIZE[0]).generate(txt)  
            fig, ax = plt.subplots(figsize=(11, 4.6)); ax.imshow(wc, interpolation="bilinear"); ax.axis("off")  
            st.pyplot(fig)  
        
        cols = st.columns(len(classes))  
        for col, c in zip(cols, classes):  
            top = pd.Series("".join(df.loc[df["label"] == c, "clean_content"]).split()).value_counts().head(12)[::-1] 
            col.plotly_chart(px.bar(x=top.values, y=top.index, orientation="h", title=f"Kata teratas - {c}", color_discrete_sequence=[cmap[c]], labels={"x": "Frekuensi", "y": ""}), use_container_width=True)  
    
    st.divider()  
    st.subheader("Penelusur Kata")  
    kw = st.text_input("Ketik kata/frasa (mis. error, mahal, guru, video)") 
    if kw and "content" in df.columns:  
        m = df["content"].astype(str).str.contains(kw, case=False, regex=False) 
        st.write(f"**{int(m.sum())}** ulasan menyebut '{kw}'")  
        if m.sum():  
            if "label" in df.columns:
                dist = df.loc[m, "label"].value_counts().reset_index(); dist.columns = ["Sentimen", "Jumlah"]  
                a, b = st.columns([1, 2])  
                a.plotly_chart(px.pie(dist, names="Sentimen", values="Jumlah", hole=.5, color="Sentimen", color_discrete_map=cmap), use_container_width=True) 
                b.dataframe(df.loc[m, ["score", "label", "content"]].head(50), use_container_width=True)  

# --- EVALUASI ---
with tabs[3]:  
    c = st.columns(6)  
    for col, (k, v), color in zip(c, M.items(), ["#4F46E5", "#10B981", "#F59E0B", "#EC4899", "#8B5CF6", "#06B6D4"]):  
        kpi(col, k, f"{v*100:.2f}%" if "AUC" not in k else f"{v:.3f}", color) 
    st.write("")  
    
    if "cm" in ev and ev["cm"] is not None and len(ev["cm"]) > 0:
        a, b = st.columns(2)  
        cm = ev["cm"]; cm_pct = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1) * 100  
        mode = a.radio("Tampilan confusion matrix", ["Jumlah", "% per kelas aktual (recall)"], horizontal=True)  
        z, fmt = (cm, "d") if mode == "Jumlah" else (cm_pct, ".1f")  
        a.plotly_chart(px.imshow(z, x=classes, y=classes, text_auto=fmt, color_continuous_scale="Purples", labels=dict(x="Prediksi", y="Aktual", color=""), title="Confusion Matrix"), use_container_width=True)  
        
        if "roc" in ev:
            fig = go.Figure()  
            for cl, (fpr, tpr, auc) in ev["roc"].items():  
                fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{cl} (AUC={auc:.3f})", line=dict(width=3, color=cmap.get(cl))))  
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="gray"), showlegend=False))  
            fig.update_layout(title="ROC Curve (One-vs-Rest)", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")  
            b.plotly_chart(fig, use_container_width=True)  

    st.subheader("Classification Report")  
    if "report" in ev and not ev["report"].empty:
        try:
            st.dataframe(ev["report"].style.format("{:.3f}").background_gradient(cmap="Purples", subset=["precision", "recall", "f1-score"]), use_container_width=True)
        except:
            st.dataframe(ev["report"])

    st.subheader("Cross-Validation (opsional)") 
    st.caption("Menghitung ulang TF-IDF + SMOTE + Naive Bayes di tiap fold pada data latih dengan parameter terbaik dari notebook.") 
    nf = st.slider("Jumlah fold", 3, 10, 5) 
    if st.button("Jalankan Cross-Validation"): 
        from imblearn.over_sampling import SMOTE, RandomOverSampler 
        from imblearn.pipeline import Pipeline as ImbPipeline 
        from sklearn.feature_extraction.text import TfidfVectorizer 
        from sklearn.model_selection import StratifiedKFold, cross_validate 
        from sklearn.naive_bayes import MultinomialNB

        try:
            if len(ev["X_tr"]) > 0 and len(ev["y_tr"]) > 0:
                mn = int(ev["y_tr"].value_counts().min() * (nf - 1) / nf)  
                smp = SMOTE(k_neighbors=min(5, mn - 1), random_state=SEED_TEST_SIZE[0]) if mn - 1 >= 1 else RandomOverSampler(random_state=SEED_TEST_SIZE[0])  
                pipe = ImbPipeline([
                    ("tfidf", TfidfVectorizer(ngram_range=vec.ngram_range, max_features=vec.max_features, min_df=vec.min_df, sublinear_tf=True)),  
                    ("smote", smp), 
                    ("nb", MultinomialNB(alpha=model.alpha))
                ]) 
                with st.spinner("Menghitung..."):  
                    cv = cross_validate(pipe, ev["X_tr"], ev["y_tr"], cv=StratifiedKFold(nf, shuffle=True, random_state=SEED_TEST_SIZE[0]), scoring=["accuracy", "f1_macro"])  
                    st.session_state["cv"] = pd.DataFrame({"Fold": [f"Fold {i+1}" for i in range(nf)], "Accuracy": cv["test_accuracy"], "F1-macro": cv["test_f1_macro"]})  
            else:
                st.warning("Data latih (X_tr, y_tr) tidak ditemukan di dalam file joblib. Cross-validation tidak bisa dijalankan.")
        except Exception as e:
            st.error(f"Gagal menjalankan Cross-Validation: {e}")
    
    if "cv" in st.session_state:  
        cvd = st.session_state["cv"]  
        st.plotly_chart(px.bar(cvd.melt("Fold", var_name="Metrik", value_name="Nilai"), x="Fold", y="Nilai", color="Metrik", barmode="group", range_y=[0, 1], title=f"CV: Accuracy {cvd['Accuracy'].mean():.3f} ± {cvd['Accuracy'].std():.3f} - F1-macro {cvd['F1-macro'].mean():.3f} ± {cvd['F1-macro'].std():.3f}"), use_container_width=True)  
        
    with st.expander("🔍 Analisis kesalahan prediksi"):  
        if "errors" in ev and not ev["errors"].empty:
            er = ev["errors"]  
            st.write(f"{len(er)} salah prediksi dari {ev.get('n_te', 0)} data uji")  
            pair = er.groupby(["aktual", "prediksi"]).size().reset_index(name="Jumlah").sort_values("Jumlah", ascending=False) 
            st.dataframe(pair, use_container_width=True) 
            st.dataframe(er.head(50).style.format({"keyakinan": "{:.2f}"}), use_container_width=True)  
        else:
            st.info("Data kesalahan prediksi tidak tersedia di file joblib.")

# --- INFLUENTIAL FEATURES ---
with tabs[4]:  
    st.subheader("🌟 Influential Features") 
    st.caption("Skor pengaruh = log P(kata | kelas) - rata-rata log P(kata | kelas lain). Makin besar, makin khas kata itu untuk kelas tersebut. Rasio odds = e^skor.") 
    a, b = st.columns([3, 1]) 
    top_n = a.slider("Jumlah fitur teratas", 5, 30, 15) 
    jenis = b.radio("Jenis fitur", ["Semua", "Kata (unigram)", "Frasa (bigram+)"]) 
    inf = influential(300) 
    
    if jenis == "Kata (unigram)": 
        inf = inf[~inf["Fitur"].str.contains(" ")] 
    elif jenis == "Frasa (bigram+)": 
        inf = inf[inf["Fitur"].str.contains(" ")] 
        
    inf = pd.concat([inf[inf["Kelas"] == c].sort_values("Skor Pengaruh", ascending=False).head(top_n) for c in classes]) 
    cols = st.columns(len(classes)) 
    
    for col, c in zip(cols, classes): 
        dd = inf[inf["Kelas"] == c].sort_values("Skor Pengaruh")
        fig = px.bar(dd, x="Skor Pengaruh", y="Fitur", orientation="h", title=c, color_discrete_sequence=[cmap[c]], hover_data=["Rasio Odds"]) 
        fig.update_layout(height=max(380, top_n * 27), margin=dict(l=0, r=0, t=40, b=0)) 
        col.plotly_chart(fig, use_container_width=True)  

    st.subheader("Heatmap log-probabilitas (kata x kelas)") 
    if len(flp) > 0 and len(fitur) > 0:
        pilih = [] 
        for i in range(len(classes)): 
            pilih += list(np.argsort(flp[i] - np.delete(flp, i, axis=0).mean(axis=0))[::-1][:8]) 
        pilih = list(dict.fromkeys(pilih)) 
        hm = pd.DataFrame(flp[:, pilih].T, index=[fitur[i] for i in pilih], columns=classes) 
        st.plotly_chart(px.imshow(hm, aspect="auto", color_continuous_scale="RdPu", text_auto=".1f", height=max(420, len(hm) * 22)), use_container_width=True)  

    st.subheader("🔍 Cek pengaruh sebuah kata") 
    w = st.text_input("Kata/frasa (hasil stemming, mis. error, mahal, bantu)").strip().lower() 
    if w: 
        idx = [j for j, f in enumerate(fitur) if w in f] 
        if not idx: 
            st.info("Kata tidak ada di kosakata model (mungkin terlalu jarang atau berbeda bentuk stemming).") 
        else: 
            rows = [] 
            for j in idx[:40]: 
                r = {"Fitur": fitur[j]} 
                for i, c in enumerate(classes): 
                    r[f"Skor {c}"] = flp[i, j] - np.delete(flp[:, j], i).mean()
                    r["Cendong ke"] = classes[int(np.argmax([r[f"Skor {c}"] for c in classes]))]
                rows.append(r) 
            st.dataframe(pd.DataFrame(rows).style.format({f"Skor {c}": "{:.2f}" for c in classes}), use_container_width=True) 
            st.download_button("⬇️ Unduh Influential Features (CSV)", influential(30).to_csv(index=False).encode("utf-8"), "influential_features.csv", "text/csv") 

# --- ASPEK & ANOMALI ---
with tabs[5]: 
    st.subheader("🔍 Analisis Aspek") 
    st.info("Fitur Analisis Aspek membutuhkan fungsi `aspect_table` yang belum disertakan dalam file ini. (Silakan tambahkan fungsi tersebut dari notebook Anda).")

    st.divider()  
    st.subheader("⚠️ Ulasan Anomali: Teks Berlawanan dengan Rating")  
    st.caption("Model yakin sentimen teksnya berbeda dari rating (mis. rating 1 tetapi teks memuji). Berguna untuk menemukan salah pilih bintang atau ulasan ambigu.")  
    if "clean_content" in df.columns and "label" in df.columns:
        conf = st.slider("Minimal keyakinan model", 0.5, 0.99, 0.80, 0.01)  
        pa = model.predict_proba(vec.transform(df["clean_content"]))  
        an = df[["score", "label", "content"]].copy()  
        an["prediksi_teks"] = model.classes_[pa.argmax(axis=1)]; an["keyakinan"] = pa.max(axis=1)  
        an = an[(an["prediksi_teks"] != an["label"]) & (an["keyakinan"] >= conf)].sort_values("keyakinan", ascending=False)  
        st.write(f"{len(an)} ulasan anomali")  
        st.dataframe(an.head(100).style.format({"keyakinan": "{:.2f}"}), use_container_width=True)  

# --- PREDIKSI ---
with tabs[6]:  
    st.subheader("🔮 Prediksi Sentimen Real-time")  
    ex = st.selectbox("Contoh cepat", ["ketik sendiri", "Aplikasinya bagus banget, materi lengkap dan gurunya jelas!", "Sering error, tidak bisa login, kecewa sekali", "Lumayan lah tapi masih banyak yang perlu diperbaiki"])  
    text = st.text_area("Masukkan ulasan", "" if ex == "ketik sendiri" else ex, height=110)  
    
    if st.button("Analisis Sentimen", type="primary") and text.strip():  
        with st.spinner("Memproses..."):  
            out = prediksi(text)  
            simpan_stem_cache()  
            if out is None:  
                st.warning("Teks kosong setelah dibersihkan. Coba kalimat lain.") 
            else:  
                emo = {"Positif": "😊", "Netral": "😐", "Negatif": "😠"}.get(out["pred"], "🤔")  
                a, b = st.columns([1, 2])  
                a.markdown(f'''
                <div class="kpi" style="border-left-color:{cmap.get(out["pred"], '#7C3AED')}">
                    <div class="l">Prediksi</div>
                    <div class="v">{emo} {out["pred"]}</div>
                </div>
                ''', unsafe_allow_html=True)  
                a.caption(f"Teks bersih: '{out['clean']}'")  
                
                pr = pd.DataFrame({"Kelas": list(out["proba"]), "Probabilitas": list(out["proba"].values())}) 
                b.plotly_chart(px.bar(pr, x="Probabilitas", y="Kelas", orientation="h", color="Kelas", color_discrete_map=cmap, range_x=[0, 1], text=pr["Probabilitas"].map("{:.1%}".format)), use_container_width=True)  
                
                if not out["contrib"].empty: 
                    st.markdown("**Kata yang paling mendorong keputusan ini:**") 
                    st.dataframe(out["contrib"], use_container_width=True) 

    st.divider() 
    st.subheader("📁 Prediksi Batch (CSV)") 
    up = st.file_uploader("CSV dengan kolom 'content'", type="csv", key="batch") 
    if up is not None: 
        bdf = pd.read_csv(up) 
        if "content" in bdf.columns: 
            with st.spinner("Memproses ulasan..."): 
                cl = bdf["content"].astype(str).apply(bersihkan_teks) 
                simpan_stem_cache() 
                P = model.predict_proba(vec.transform(cl)) 
                bdf["prediksi"] = model.classes_[P.argmax(axis=1)]; bdf["keyakinan"] = P.max(axis=1).round(3) 
                st.dataframe(bdf.head(100), use_container_width=True) 
                st.plotly_chart(px.pie(bdf, names="prediksi", hole=.5, color="prediksi", color_discrete_map=cmap), use_container_width=True) 
                st.download_button("⬇️ Unduh hasil", bdf.to_csv(index=False).encode("utf-8"), "hasil_prediksi.csv", "text/csv") 
        else: 
            st.error("Kolom 'content' tidak ditemukan.") 

# --- METODOLOGI ---
with tabs[7]: 
    st.subheader("📚 Alur Proses") 
    st.markdown("""
    Proses berikut dijalankan di **notebook**; aplikasi ini hanya menampilkan hasilnya.

    1. **Load data** ulasan Ruangguru dari Google Play Store.
    2. **Pelabelan** dari rating: 1–2 Negatif, 3 Netral, 4–5 Positif.
    3. **Pra-pemrosesan**: case folding, hapus URL/simbol, normalisasi slang, stopword (negasi dipertahankan), stemming Sastrawi.
    4. **Split stratified** 80/20 (random_state=42), lalu **TF-IDF** dan **SMOTE** hanya pada data latih.
    5. **Tuning** GridSearchCV (F1-macro) dan pelatihan **Multinomial Naive Bayes**.
    6. **Evaluasi**: Accuracy, Precision, Recall, F1, confusion matrix, ROC-AUC OvR, cross-validation.
    7. **Influential Features** dari `feature_log_prob_` dan **analisis aspek**.
    
    Aplikasi membentuk ulang data uji yang sama dengan notebook (split deterministik) lalu hanya mengambil `predict`, sehingga angkanya identik tanpa melatih ulang.
    
    **Catatan:** label berasal dari rating, jadi model mengukur kemampuan menebak rating dari teks, bukan sentimen yang dianotasi manual.
    """) 
    
    st.download_button("⬇️ Unduh data bersih + label (CSV)", df.to_csv(index=False).encode("utf-8"), "data_bersih.csv", "text/csv")