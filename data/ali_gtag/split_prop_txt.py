from itertools import groupby
from tqdm import tqdm

source_path = 'gtag_train_0_39999_propsal_list.txt'
output_path = 'traindata_1/gtag_train_35000_39999_propsal_list.txt'
start_idx = 35000
end_idx = 39999

lines = list(open(source_path))
groups = groupby(lines, lambda x: x.startswith('#'))
info_list = [[x.strip() for x in list(g)] for k, g in groups if not k]


f = open(output_path, 'w')
pbar = tqdm(total=len(info_list[start_idx:end_idx]))

for v_idx in range(start_idx,end_idx+1):
    pbar.update(1)
    f.write('#{}'.format(str(v_idx))), f.write('\r\n')
    for line in info_list[v_idx]:
    	f.write(line), f.write('\r\n')
pbar.close()
f.close()
