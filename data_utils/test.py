import os

def test_video_txt():
    sum_all_desc = '/home/alexandertchk/VSCode/multimodal/VG-GPLMs/dataset/sum_all/desc.tok.txt'
    sum_all_tran = '/home/alexandertchk/VSCode/multimodal/VG-GPLMs/dataset/sum_all/tran.tok.txt'
    video_actions = '/home/alexandertchk/VSCode/multimodal/VG-GPLMs/dataset/video_action_features/'

    video_ids_desc = []
    video_ids_tran = []
    video_ids_actions = []

    for filename in os.listdir(video_actions):
        if filename.endswith(".npy"):
            video_id = os.path.splitext(filename)[0]
            video_ids_actions.append(video_id)

    with open(sum_all_desc, 'r') as desc, open(sum_all_tran, 'r') as tran:
        desc_lines = desc.readlines()
        tran_lines = tran.readlines()

    for desc_line, tran_line in zip(desc_lines, tran_lines):
        desc_id = desc_line.split()[0]
        tran_id = tran_line.split()[0]
        video_ids_desc.append(desc_id)
        video_ids_tran.append(tran_id)



    print(f"video_ids_desc == video_ids_tran: {sorted(video_ids_desc) == sorted(video_ids_tran)}")
    print(f"video_ids_actions == video_ids_tran: {sorted(video_ids_actions) == sorted(video_ids_tran)}")


    print(f"len video_ids_desc: {len(video_ids_desc)}")
    print(f"len video_ids_tran: {len(video_ids_tran)}")
    print(f"len video_ids_actions: {len(video_ids_actions)}")

