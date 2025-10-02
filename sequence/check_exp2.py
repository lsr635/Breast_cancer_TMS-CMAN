import os, re, collections

root = '/root/autodl-tmp/dataset_exp_1'
phase = 'train'  # 改成 val/test 查看
cls_dirs = ['B','M']

re_pre = re.compile(r'^(BreaDM-(Ma|Be)-\d+)_VIBRANT_p-(\d+)\.jpg$', re.I)
re_sub = re.compile(r'^(BreaDM-(Ma|Be)-\d+)_SUB(\d+)_p-(\d+)\.jpg$', re.I)

total_files=0
buckets = collections.defaultdict(lambda: {'pre': None, 'sub': {}})

for cls in cls_dirs:
    d = os.path.join(root, phase, cls)
    if not os.path.isdir(d): 
        print('Missing dir:', d); 
        continue
    for f in os.listdir(d):
        if not f.lower().endswith('.jpg'):
            continue
        total_files += 1
        m = re_pre.match(f)
        if m:
            key = (m.group(1), int(m.group(3)))  # ("BreaDM-Be-1801", pidx)
            buckets[key]['pre'] = os.path.join(d, f)
            continue
        m = re_sub.match(f)
        if m:
            key = (m.group(1).replace('_SUB',''), int(m.group(4)))
            sidx = int(m.group(3))
            buckets[key]['sub'][sidx] = os.path.join(d, f)

print('Total jpg:', total_files)
full=0; miss_list=[]
for key, v in buckets.items():
    ok = (v['pre'] is not None) and all(k in v['sub'] for k in range(1,9))
    if ok:
        full += 1
    else:
        miss = [k for k in range(1,9) if k not in v['sub']]
        miss_list.append((key, v['pre'] is not None, miss))
print('Complete samples (pre+SUB1..8):', full)
for i,(key, has_pre, miss) in enumerate(miss_list[:20]):
    print(i, key, 'pre:', has_pre, 'missing SUB:', miss)