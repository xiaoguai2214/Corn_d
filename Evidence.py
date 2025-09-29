import os, glob, csv, hashlib
from collections import defaultdict, Counter
import cv2, numpy as np

OUT_ROOT = r"E:\model\yolov11\datasets\yuan_d_new"
IMG_DIR, LBL_DIR = os.path.join(OUT_ROOT,"images"), os.path.join(OUT_ROOT,"labels")
LIST_DIR = os.path.join(OUT_ROOT,"lists"); os.makedirs(LIST_DIR, exist_ok=True)

CLASS_NAMES = ['Healthy','Common_Rust','Blight','Gray_Leaf_Spot','Corn_borer','Army_worm','Aphids']
NUM_CLASSES = len(CLASS_NAMES)
IMG_EXTS = (".jpg",".jpeg",".png",".bmp",".tif",".tiff")
AUG_PREFIXES = ("aug1_","mosaic_")

AHASH_SIDE, AHASH_BUCKET_PREFIX_BITS, AHASH_HAMMING_THR = 8, 16, 5

def to_posix(p): return p.replace("\\","/")
def rel_from_root(p): return to_posix(os.path.relpath(p, OUT_ROOT))
def ensure(d): os.makedirs(d, exist_ok=True)
def find_image_for_stem(split, stem):
    base = os.path.join(IMG_DIR, split)
    for ext in IMG_EXTS:
        cand = os.path.join(base, stem+ext)
        if os.path.exists(cand): return cand
    return None

def read_yolo_boxes(p):
    rows=[];
    if not os.path.exists(p): return rows
    with open(p,"r",encoding="utf-8") as f:
        for ln in f:
            ps=ln.strip().split()
            if len(ps)!=5: continue
            try:
                cid=int(float(ps[0])); cx,cy,w,h=map(float,ps[1:5])
                rows.append([cid,cx,cy,w,h])
            except: pass
    return rows
def gather_image_level_classes(label_path): return set([b[0] for b in read_yolo_boxes(label_path)])

def sha1_of_file(path, chunk=1<<20):
    h=hashlib.sha1()
    with open(path,"rb") as f:
        while True:
            b=f.read(chunk)
            if not b: break
            h.update(b)
    return h.hexdigest()

def ahash_of_image(path, side=AHASH_SIDE):
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        s = sha1_of_file(path); return int(s[:16],16)
    small = cv2.resize(img,(side,side), interpolation=cv2.INTER_AREA)
    m=float(small.mean()); bits=(small>=m).astype(np.uint8).flatten()
    out=0
    for b in bits: out=(out<<1)|int(b)
    return out

def _popcount(x):
    try: return x.bit_count()
    except AttributeError: return bin(x).count("1")
def hamming(a,b): return _popcount(a^b)

def list_label_files(split): return sorted(glob.glob(os.path.join(LBL_DIR,split,"*.txt")))
def list_image_files(split):
    base=os.path.join(IMG_DIR,split); out=[]
    for ext in IMG_EXTS: out+=glob.glob(os.path.join(base,f"*{ext}"))
    return sorted(out)

def write_lists_and_stems():
    print("[1] lists & stems")
    for split in ["train","val","test"]:
        lbls=list_label_files(split); img_list=[]; stem_list=[]
        for lp in lbls:
            stem=os.path.splitext(os.path.basename(lp))[0]
            ip=find_image_for_stem(split, stem)
            if ip is None: continue
            img_list.append(rel_from_root(ip)); stem_list.append(stem)
        with open(os.path.join(LIST_DIR,f"{split}.txt"),"w",encoding="utf-8") as f:
            f.write("\n".join(img_list)+"\n")
        with open(os.path.join(LIST_DIR,f"{split}_stems.txt"),"w",encoding="utf-8") as f:
            f.write("\n".join(stem_list)+"\n")
        print(f"  {split}: {len(img_list)} images")

def leakage_scan():
    print("[2] leakage scan (val/test)")
    lines=[]; flags=0
    for split in ["val","test"]:
        offenders=[]
        for ip in list_image_files(split):
            base=os.path.basename(ip).lower()
            if base.startswith(AUG_PREFIXES): offenders.append(base)
        lines.append(f"{split}: augmented_prefix_count={len(offenders)}")
        if offenders: lines.append("  examples: "+", ".join(offenders[:20]))
        flags += len(offenders)
    outp=os.path.join(LIST_DIR,"leakage_check.txt")
    with open(outp,"w",encoding="utf-8") as f: f.write("\n".join(lines)+"\n")
    print("  ->", rel_from_root(outp), "PASS" if flags==0 else "ALERT")

def duplicate_scan_sha1():
    print("[3A] dup by SHA-1")
    by_hash=defaultdict(list)
    for split in ["train","val","test"]:
        for ip in list_image_files(split):
            by_hash[sha1_of_file(ip)].append((split, rel_from_root(ip)))
    groups=[(k,v) for k,v in by_hash.items() if len(v)>1 and len({s for s,_ in v})>=2]
    out=os.path.join(LIST_DIR,"dup_report_sha1.txt")
    with open(out,"w",encoding="utf-8") as f:
        if not groups: f.write("No cross-split exact duplicates found.\n")
        else:
            f.write("# sha1\tsplit\trel_image_path\n")
            for k,v in groups:
                for sp,relp in v: f.write(f"{k}\t{sp}\t{relp}\n")
                f.write("\n")
    print("  ->", rel_from_root(out), f"(groups={len(groups)})")

def duplicate_scan_ahash():
    print("[3B] dup by aHash (Hamming<=5)")
    entries=[]
    for split in ["train","val","test"]:
        for ip in list_image_files(split):
            h=ahash_of_image(ip); pref=h>>(64-AHASH_BUCKET_PREFIX_BITS)
            entries.append((split, rel_from_root(ip), h, pref))
    buckets=defaultdict(list)
    for e in entries: buckets[e[3]].append(e)
    pairs=[]
    for pref, arr in buckets.items():
        if len(arr)<=1: continue
        by_split=defaultdict(list)
        for sp,relp,h,_ in arr: by_split[sp].append((relp,h))
        splits=list(by_split.keys())
        for i in range(len(splits)):
            for j in range(i+1,len(splits)):
                for rel1,h1 in by_split[splits[i]]:
                    for rel2,h2 in by_split[splits[j]]:
                        if hamming(h1,h2)<=AHASH_HAMMING_THR:
                            pairs.append((splits[i],rel1,splits[j],rel2,hamming(h1,h2)))
    out=os.path.join(LIST_DIR,"dup_report_ahash.txt")
    with open(out,"w",encoding="utf-8") as f:
        if not pairs: f.write("No cross-split near-duplicates (aHash) under current threshold.\n")
        else:
            f.write("# split1\trel_image_1\tsplit2\trel_image_2\tHamming\n")
            for sp1,p1,sp2,p2,d in sorted(pairs,key=lambda x:(x[0],x[2],x[4],x[1],x[3])):
                f.write(f"{sp1}\t{p1}\t{sp2}\t{p2}\t{d}\n")
    print("  ->", rel_from_root(out), f"(pairs={len(pairs)})")

def per_class_image_counts(pairs):
    cnt=Counter()
    for _,lp,_ in pairs:
        for cid in gather_image_level_classes(lp): cnt[cid]+=1
    return cnt

def collect_pairs(split):
    img_dir=os.path.join(IMG_DIR,split); lbl_dir=os.path.join(LBL_DIR,split); out=[]
    for lp in glob.glob(os.path.join(lbl_dir,"*.txt")):
        stem=os.path.splitext(os.path.basename(lp))[0]
        ip=find_image_for_stem(split, stem)
        if ip: out.append((ip,lp,stem))
    return out

def write_split_distribution():
    print("[4] split_distribution.csv")
    rows=[]
    for split in ["train","val","test"]:
        cnt=per_class_image_counts(collect_pairs(split))
        for cid in range(NUM_CLASSES):
            rows.append([split, CLASS_NAMES[cid], cnt.get(cid,0)])
    out_csv=os.path.join(OUT_ROOT,"split_distribution.csv")
    with open(out_csv,"w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["split","class","images_containing_class"]); w.writerows(rows)
    print("  ->", rel_from_root(out_csv))

def partition_train_by_prefix():
    base,aug1,mosaic=[],[],[]
    for ip,lp,stem in collect_pairs("train"):
        b=os.path.basename(ip).lower()
        if b.startswith("mosaic_"): mosaic.append((ip,lp,stem))
        elif b.startswith("aug1_"): aug1.append((ip,lp,stem))
        else: base.append((ip,lp,stem))
    return base,aug1,mosaic

def write_counts_after_steps():
    print("[5] counts_after_step1/2")
    base,aug1,mosaic = partition_train_by_prefix()
    step1, step2 = base+aug1, base+aug1+mosaic
    def dump(pairs, out_csv):
        cnt=per_class_image_counts(pairs)
        with open(out_csv,"w",newline="",encoding="utf-8") as f:
            w=csv.writer(f); w.writerow(["split","class","images_containing_class"])
            for cid in range(NUM_CLASSES): w.writerow(["train", CLASS_NAMES[cid], cnt.get(cid,0)])
    dump(step1, os.path.join(OUT_ROOT,"counts_after_step1.csv"))
    dump(step2, os.path.join(OUT_ROOT,"counts_after_step2.csv"))
    print("  ->", rel_from_root(os.path.join(OUT_ROOT,"counts_after_step1.csv")))
    print("  ->", rel_from_root(os.path.join(OUT_ROOT,"counts_after_step2.csv")))

def sanity_check_labels():
    print("[6] label id sanity")
    bad=[]
    for split in ["train","val","test"]:
        for lp in glob.glob(os.path.join(LBL_DIR,split,"*.txt")):
            with open(lp,"r",encoding="utf-8") as f:
                for i,ln in enumerate(f,1):
                    ps=ln.strip().split()
                    if len(ps)!=5: continue
                    try:
                        cid=int(float(ps[0]))
                        if cid<0 or cid>=NUM_CLASSES: bad.append((rel_from_root(lp),i,ln.strip()))
                    except: pass
    rpt=os.path.join(LIST_DIR,"invalid_cid_report.txt")
    with open(rpt,"w",encoding="utf-8") as f:
        if not bad: f.write(f"PASS: all labels in [0,{NUM_CLASSES-1}].\n")
        else:
            for p,i,ln in bad: f.write(f"{p} : line {i} -> {ln}\n")
    print("  ->", rel_from_root(rpt), "PASS" if not bad else f"ALERT ({len(bad)})")

def main():
    print("== Evidence & Leakcheck ==")
    write_lists_and_stems(); leakage_scan()
    duplicate_scan_sha1(); duplicate_scan_ahash()
    write_split_distribution(); write_counts_after_steps(); sanity_check_labels()
    print("\n[OK] Done. Files in lists/ + CSVs at dataset root.")

if __name__=="__main__": main()
