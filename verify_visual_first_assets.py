from pathlib import Path
from run_visual_first_validation import VALIDATION, asset_dirs, exact_asset, find_video

root = Path('.').resolve()
audio_dir, lecture_dir, video_dirs, gt_dir = asset_dirs(root)
print('=' * 78)
print('LECTURESENSE — VISUAL-FIRST ASSET CHECK')
print('=' * 78)
print('Project root:', root)
for d in video_dirs:
    print('Video dir:', d, 'FOUND' if d.exists() else 'MISSING')
for item in VALIDATION:
    print('\n', item['id'])
    print('  SRT  :', exact_asset(audio_dir, item['srt']))
    print('  PDF  :', exact_asset(lecture_dir, item['pdf']))
    print('  GT   :', exact_asset(gt_dir, item['gt']))
    video, cands = find_video(video_dirs, item['video_aliases'])
    print('  VIDEO:', video)
    if video is None and cands:
        print('  Candidates:')
        for c in cands:
            print('   ', c)
print('\nNo test lecture is inspected by this checker.')
