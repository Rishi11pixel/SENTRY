
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

CLASSES = [
    "ALCOHOL_SANITIZER", "AMBIENT_CLEAN", "EXPLOSIVE_PROXY",
    "NARCOTIC_PROXY", "WEATHER_DRIFT"
]

RF_CM = np.array([
    [1953,139,5,100,203], [7,2160,1,3,229], [0,11,2217,6,166],
    [135,20,8,1992,245], [45,430,76,97,1752]
])
FLOAT_CM = np.array([
    [1960,113,11,130,186], [29,2116,5,5,245], [1,9,2239,2,149],
    [85,11,7,2085,212], [45,323,79,109,1844]
])

RF_ACC, RF_F1 = 83.95, 84.15
FLOAT_ACC, FLOAT_F1 = 85.37, 85.53
INT8_ACC, INT8_F1 = 84.27, 84.05

FEATURES = {
    "VMQ3":36.15, "VSEN0568":25.69, "VMQ135":24.93, "VMQ2":10.13,
    "dVdt_max":1.91, "humidity":1.02, "temperature":0.18
}

def find_cm(x):
    if isinstance(x, dict):
        for k, v in x.items():
            if "confusion" in str(k).lower() and "matrix" in str(k).lower():
                try:
                    a = np.asarray(v)
                    if a.shape == (5,5): return a.astype(int)
                except: pass
            r = find_cm(v)
            if r is not None: return r
    elif isinstance(x, list):
        for v in x:
            r = find_cm(v)
            if r is not None: return r
    return None

def get_int8_cm(root):
    p = root/"results"/"final_int8_evaluation.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}. Run final INT8 evaluation first.")
    with p.open(encoding="utf-8") as f: data=json.load(f)
    cm=find_cm(data)
    if cm is None: raise ValueError("No 5x5 INT8 confusion matrix found in final_int8_evaluation.json.")
    return cm

def save(fig, p):
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", p)

def performance(out):
    names=["Random Forest","TinyML Float32","TinyML INT8"]
    acc=[RF_ACC,FLOAT_ACC,INT8_ACC]; f1=[RF_F1,FLOAT_F1,INT8_F1]
    x=np.arange(3); w=.36
    fig,ax=plt.subplots(figsize=(10,6))
    a=ax.bar(x-w/2,acc,w,label="Accuracy"); b=ax.bar(x+w/2,f1,w,label="Macro F1")
    ax.set_title("SENTRY Final Model Performance"); ax.set_ylabel("Score (%)")
    ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylim(75,90); ax.legend()
    for bars in (a,b):
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+.2,
                    f"{bar.get_height():.2f}%",ha="center",fontsize=9)
    save(fig,out/"01_model_performance.png")

def cm_plot(cm,title,name,out):
    fig,ax=plt.subplots(figsize=(9,8))
    cmap = plt.get_cmap("viridis")
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.set_title(title); ax.set_xlabel("Predicted label"); ax.set_ylabel("True label")
    ax.set_xticks(range(5)); ax.set_yticks(range(5))
    ax.set_xticklabels(CLASSES,rotation=45,ha="right"); ax.set_yticklabels(CLASSES)
    for i in range(5):
        for j in range(5):
            value = cm[i,j]
            cell_color = cmap(value / max(1, cm.max()))
            r,g,b,_ = cell_color
            luminance = 0.299*r + 0.587*g + 0.114*b
            text_color = "white" if luminance < 0.55 else "black"
            ax.text(j,i,f"{value:,}",ha="center",va="center",
                    color=text_color,fontsize=9,fontweight="bold")
    fig.colorbar(im,ax=ax,fraction=.046,pad=.04,label="Samples")
    save(fig,out/name)

def feature_plot(out):
    items=sorted(FEATURES.items(),key=lambda x:x[1])
    fig,ax=plt.subplots(figsize=(10,6))
    bars=ax.barh([x[0] for x in items],[x[1] for x in items])
    ax.set_title("SENTRY Sensor-Fusion Feature Importance")
    ax.set_xlabel("Random Forest importance (%)"); ax.set_xlim(0,40)
    for bar,v in zip(bars,[x[1] for x in items]):
        ax.text(v+.5,bar.get_y()+bar.get_height()/2,f"{v:.2f}%",va="center",fontsize=9)
    save(fig,out/"05_feature_importance.png")

def quant_plot(out):
    metrics=["Accuracy","Macro F1"]; f=[FLOAT_ACC,FLOAT_F1]; q=[INT8_ACC,INT8_F1]
    x=np.arange(2); w=.36
    fig,ax=plt.subplots(figsize=(9,6))
    a=ax.bar(x-w/2,f,w,label="TinyML Float32"); b=ax.bar(x+w/2,q,w,label="TinyML INT8")
    ax.set_title("TinyML Quantization Trade-off"); ax.set_ylabel("Score (%)")
    ax.set_xticks(x); ax.set_xticklabels(metrics); ax.set_ylim(75,90); ax.legend()
    for bars in (a,b):
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+.2,
                    f"{bar.get_height():.2f}%",ha="center",fontsize=9)
    ax.text(.98,.03,"INT8 model size: 3.52 KB\nAccuracy change: −1.10 percentage points",
            transform=ax.transAxes,ha="right",va="bottom",fontsize=9)
    save(fig,out/"06_quantization_tradeoff.png")

def architecture(out):
    fig,ax=plt.subplots(figsize=(13,7)); ax.set_xlim(0,13); ax.set_ylim(0,7); ax.axis("off")
    boxes=[(.5,2.6,2.1,1.8,"7 Sensor/\nEnvironment\nFeatures"),
           (3.5,2.6,2.0,1.8,"StandardScaler"),
           (6.4,2.6,2.1,1.8,"TinyML MLP\n7 → 16 → 8 → 5"),
           (9.4,2.6,1.8,1.8,"INT8\nTFLite"),
           (11.8,2.6,.8,1.8,"5-Class\nOutput")]
    for x,y,w,h,text in boxes:
        p=FancyBboxPatch((x,y),w,h,boxstyle="round,pad=.08,rounding_size=.08",linewidth=1.5)
        ax.add_patch(p); ax.text(x+w/2,y+h/2,text,ha="center",va="center",fontsize=11,fontweight="bold")
    for i in range(len(boxes)-1):
        x,y,w,h,_=boxes[i]; x2,y2,_,h2,_=boxes[i+1]
        ax.add_patch(FancyArrowPatch((x+w+.08,y+h/2),(x2-.08,y2+h2/2),
                                     arrowstyle="->",mutation_scale=18,linewidth=1.5))
    ax.text(6.5,5.6,"SENTRY TinyML Inference Architecture",ha="center",fontsize=18,fontweight="bold")
    ax.text(6.5,1.25,"100 ms sampling → 30-sample / 3-second window → feature extraction → inference",
            ha="center",fontsize=11)
    save(fig,out/"07_tinyml_architecture.png")

def main():
    root=Path(__file__).resolve().parent
    out=root/"results"/"visuals"; out.mkdir(parents=True,exist_ok=True)
    int8_cm=get_int8_cm(root)
    for cm in (RF_CM,FLOAT_CM,int8_cm):
        if cm.sum()!=12000: raise ValueError("Each final confusion matrix must contain 12,000 samples.")
    performance(out)
    cm_plot(RF_CM,"SENTRY Random Forest — Final Blind Test","02_confusion_random_forest.png",out)
    cm_plot(FLOAT_CM,"SENTRY TinyML Float32 — Final Blind Test","03_confusion_tinyml_float32.png",out)
    cm_plot(int8_cm,"SENTRY TinyML INT8 — Final Blind Test","04_confusion_tinyml_int8.png",out)
    feature_plot(out); quant_plot(out); architecture(out)
    print("\nSENTRY ML visual generation complete.")
    print("Output:",out)

if __name__=="__main__": main()
