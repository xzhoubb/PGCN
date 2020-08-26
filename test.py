import json

test_path = 'data/ali_gtag/test_num_frames.json'
train_path = 'data/ali_gtag/train_num_frames.json'

with open(train_path,'r') as f:
    train_num_frame_dict = json.load(f)

with open(test_path,'r') as f:
    test_num_frame_dict = json.load(f)

train_max_id = None 
train_num_id = 0
for id, num in train_num_frame_dict.items():
    if num > train_num_id:
        train_max_id = id 
        train_num_id = num
print ('train_max_id', train_max_id)
print ('train_max_num', train_num_id)

test_max_id = None 
test_num_id = 0
for id, num in test_num_frame_dict.items():
    if num > test_num_id:
        test_max_id = id 
        test_num_id = num
print ('test_max_id', test_max_id)
print ('test_max_num', test_num_id)   
