import json
with open('balanced/train/_annotations.coco.json') as f:
    d = json.load(f)
for ann in d['annotations']:
    if ann['category_id'] != 0:
        print('keys:', list(ann.keys()))
        seg = ann.get('segmentation')
        print('seg type:', type(seg))
        if isinstance(seg, dict):
            print('RLE keys:', list(seg.keys()))
            counts = seg.get('counts')
            print('counts type:', type(counts), '| preview:', str(counts)[:60])
        elif isinstance(seg, list) and seg:
            first = seg[0]
            print('polygon - inner type:', type(first))
            if isinstance(first, str):
                print('string RLE preview:', first[:60])
            else:
                print('polygon coords[:6]:', first[:6])
        break
