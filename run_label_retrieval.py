import torch
import os
import shutil
from tqdm import tqdm
from PIL import Image
from tqdm import tqdm
from collections import Counter
import json
import matplotlib.pyplot as plt
import argparse

from transformers import AutoProcessor, LlavaForConditionalGeneration

def extract_objects(output):
    # Removing leading numbers and punctuation
    objects = output.replace('1.', '').replace('2.', '').replace('3.', '').replace(',', ' ').split()
    # Splitting and cleaning up objects
    cleaned_objects = [obj.strip().lower() for obj in objects if obj.strip() and obj.strip() != 'and']
    # Select unique objects
    cleaned_objects = list(set(cleaned_objects))
    return cleaned_objects


def main(args):

    model = LlavaForConditionalGeneration.from_pretrained("bczhou/tiny-llava-v1-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True).to(0)
    processor = AutoProcessor.from_pretrained("bczhou/tiny-llava-v1-hf")

    counter = Counter()

    for i, image_path in enumerate(tqdm(os.listdir(args.image_dir))):
        image_path = os.path.join(args.image_dir, image_path)
        image = Image.open(image_path)
        
        prompt = "USER: <image>\nName 3 objects in the image, separated by comma.\nASSISTANT:"

        inputs = processor(prompt, image, return_tensors='pt').to(0, torch.float16)

        output = model.generate(**inputs, max_new_tokens=200, do_sample=False)

        decode_output = processor.decode(output[0][:], skip_special_tokens=True)

        # Take only text after ASSISTANT:
        decode_output = decode_output.split("ASSISTANT:")[1].strip()

        objects = extract_objects(decode_output)
        counter.update(objects)

    print(counter)

    results_dir = os.path.join(args.output_dir, os.path.basename(args.image_dir))

    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)
    os.mkdir(results_dir)

    # Save the counter to a file
    with open(os.path.join(results_dir, "labels_counter.json"), "w") as f:
        json.dump(counter, f)

    top_10_labels = counter.most_common(10)

    # Prepare data for plotting
    objects, counts = zip(*top_10_labels)
    objects = list(objects)
    counts = list(counts)


    # Create a horizontal bar plot
    fig, ax = plt.subplots()
    ax.barh(objects[::-1], counts[::-1], color='skyblue')

    plt.xlabel('Counts', fontsize=16)
    plt.ylabel('Labels', fontsize=16)
    plt.title('Top 10 Most Frequent Labels', fontsize=18)
    plt.yticks(fontsize=16)
    plt.tight_layout()

    # Show the plot
    plt.savefig(os.path.join(results_dir, "top_10_labels.png"))


if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, default="eval_outputs/VOC_15_unknown/predicted_unknown_voc")
    parser.add_argument("--output_dir", type=str, default="eval_outputs")

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    main(args)  

    
    

    

    
