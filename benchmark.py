from PIL import Image
import argparse
import ollama
import json
import glob
import os
import csv

def split_image(img_file, prefix):
    img = Image.open(img_file)

    wid, hei = img.size
    is_dual = wid > hei * 2

    if is_dual:
        half_wid = wid // 2

        dims = [
            (0, 0, half_wid, hei),
            (half_wid, 0, half_wid, hei)
        ]
    else:
        dims = [ (0, 0, wid, hei) ]

    sub_dims = []

    for x, y, w, h in dims:
        sh = h // 3
        mh = h - sh

        if w > 1280:
            hw = w // 2

            sub_dims.extend([
                (x, y + sh, hw, mh),
                (x + hw, y + sh, hw, mh)
            ])
        else:
            sub_dims.append((x, y + sh, w, mh))

    clip_files, idx = [], 0

    for x, y, w, h in sub_dims:
        clip = img.crop((x, y, x + w, y + h))
        
        idx += 1
        dst_file = f"{prefix}-{idx}.png"

        clip.save(dst_file)
        clip_files.append(os.path.abspath(dst_file))

    return clip_files

def validate_answer(answer):
    for ch in answer:
        if ch.isdigit():
            r = int(ch)
            if r > 5: r = 0
            return r

    return 0

def check_clip(clip_file, reasoning = False):
    reasoning = False
    
    explain_prompt = "Return JSON like {'choice': <Best choice number>, 'why': <The reason>}."
    only_prompt = "Return JSON like {'choice': <Best choice number>}."
    
    r = ollama.chat(
        model = "llama3.2-vision:11b",
        messages = [{
            "role": "user",
            "content": """
Check the image and return one integer number based on the following rules (1-6).
**1** If image contains sexual activity, return 1.
**2** If image contains kissing couple, return 2.
**3** If image contains bikini-level wearing of woman, return 3.
**4** If image contains naked breast, hip, clit of woman, return 4.
**5** If image contains man's pennis, return 5.
**6** If image contains none of above all, return 0.
{reasoning}
            """.format(
                    reasoning = explain_prompt if reasoning else only_prompt
                ),
            "images": [clip_file]
        }],
        options = dict(temperature = 0),
        format = "json"
    )
    answer = r.get("message", {}).get("content", "{ \"choice\": 0, \"why\": \"Error\" }").strip()
    dur = r.get("eval_duration", 0) / 1e9


    try:
        answer = json.loads(answer)
    except Exception:
        print(f"### error JSON parsing: {answer}")
        answer = {}

    choice = answer.get("choice", 0)
    why = answer.get("why", "")

    print(clip_file, '->', choice)
    if type(choice) != int: choice = validate_answer(choice)

    return choice, why, 0

def check_image(image_file, idx, reasoning = False, keep_clips = False):
    prefix = "./temp/{:05}".format(idx)
    clip_files = split_image(image_file, prefix)
    
    ed, r = 0, 0

    for clip in clip_files:
        r, a, d = check_clip(clip, reasoning)
        ed += d

        if reasoning:
            with open("reason.txt", "a", encoding = "utf-8") as fp:
                fp.write(f"{clip}->{r}->{a}\n")

        if r != 0: break

    if not keep_clips:
        for clip in clip_files:
            os.remove(clip)

    return r, ed

def find_image_files(directory):
    all_files = glob.glob(os.path.join(directory, "**"), recursive = True)    
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
    image_files = []
    
    for f in all_files:
        if os.path.isdir(f): continue    
        if f.lower().endswith(image_extensions): image_files.append(f.replace('\\', '/'))
    
    return image_files

def main():
    parser = argparse.ArgumentParser(description = 'Check images for NSFW content.')
    parser.add_argument("directory", type = str, help = 'Directory to search for image files')
    parser.add_argument("output_csv", type = str, help = 'Output CSV file path')
    parser.add_argument("-k", "--keep", action = "store_true", help = "Keep clip images")
    parser.add_argument("-r", "--reason", action = "store_true", help = "Explain why")
    args = parser.parse_args()
    
    if os.path.exists("reason.txt"): os.remove("reason.txt")
    os.makedirs("./temp", exist_ok = True)

    image_files = find_image_files(args.directory)
    results = []

    for idx, image_file in enumerate(image_files):
        result, dur = check_image(image_file, idx, args.reason, args.keep)
        results.append((image_file, result, dur))

        print(image_file, '---', result, '---', '{:.2f}s'.format(dur))

    with open(args.output_csv, mode = 'w', newline = '') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['file', 'result', 'duration'])
        writer.writerows(results)

    print(f'Results written to {args.output_csv}')

if __name__ == '__main__':    
    main()
