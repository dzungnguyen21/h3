try:
    from pattern.en import singularize
except ImportError:
    from nltk.stem import WordNetLemmatizer
    _lemmatizer = WordNetLemmatizer()
    def singularize(word):
        return _lemmatizer.lemmatize(word)
import os
import sys
import json
import nltk
import torch
import argparse
import pickle
import numpy as np
from tqdm import tqdm
from PIL import Image
from collections import defaultdict
# from pattern.en import singularize
from pycocotools.coco import COCO

# Make sure you have downloaded the NLTK tokenizer and corpora:
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')

# =====================================================================
# 1. ROBUST CHAIR NLP EVALUATOR (Original Rules & Synonyms)
# =====================================================================

synonyms_txt = '''
person, girl, boy, man, woman, kid, child, chef, baker, people, adult, rider, children, baby, worker, passenger, sister, biker, policeman, cop, officer, lady, cowboy, bride, groom, male, female, guy, traveler, mother, father, gentleman, pitcher, player, skier, snowboarder, skater, skateboarder, foreigner, caller, offender, coworker, trespasser, patient, politician, soldier, grandchild, serviceman, walker, drinker, doctor, bicyclist, thief, buyer, teenager, student, camper, driver, solider, hunter, shopper, villager
bicycle, bike, unicycle, minibike, trike
car, automobile, van, minivan, sedan, suv, hatchback, cab, jeep, coupe, taxicab, limo, taxi
motorcycle, scooter, motor bike, motor cycle, motorbike, moped
airplane, jetliner, plane, air plane, monoplane, aircraft, jet, airbus, biplane, seaplane
bus, minibus, trolley
train, locomotive, tramway, caboose
truck, pickup, lorry, hauler, firetruck
boat, ship, liner, sailboat, motorboat, dinghy, powerboat, speedboat, canoe, skiff, yacht, kayak, catamaran, pontoon, houseboat, vessel, rowboat, trawler, ferryboat, watercraft, tugboat, schooner, barge, ferry, sailboard, paddleboat, lifeboat, freighter, steamboat, riverboat, battleship, steamship
traffic light, street light, traffic signal, stop light, streetlight, stoplight
fire hydrant, hydrant
stop sign
parking meter
bench, pew
bird, ostrich, owl, seagull, goose, duck, parakeet, falcon, robin, pelican, waterfowl, heron, hummingbird, mallard, finch, pigeon, sparrow, seabird, osprey, blackbird, fowl, shorebird, woodpecker, egret, chickadee, quail, bluebird, kingfisher, buzzard, willet, gull, swan, bluejay, flamingo, cormorant, parrot, loon, gosling, waterbird, pheasant, rooster, sandpiper, crow, raven, turkey, oriole, cowbird, warbler, magpie, peacock, cockatiel, lorikeet, puffin, vulture, condor, macaw, peafowl, cockatoo, songbird
cat, kitten, feline, tabby
dog, puppy, beagle, pup, chihuahua, schnauzer, dachshund, rottweiler, canine, pitbull, collie, pug, terrier, poodle, labrador, doggie, doberman, mutt, doggy, spaniel, bulldog, sheepdog, weimaraner, corgi, cocker, greyhound, retriever, brindle, hound, whippet, husky
horse, colt, pony, racehorse, stallion, equine, mare, foal, palomino, mustang, clydesdale, bronc, bronco
sheep, lamb, ram, goat, ewe
cow, cattle, oxen, ox, calf, holstein, heifer, buffalo, bull, zebu, bison
elephant
bear, panda
zebra
giraffe
backpack, knapsack
umbrella
handbag, wallet, purse, briefcase
tie, bow, bow tie
suitcase, suit case, luggage
frisbee
skis, ski
snowboard
sports ball, ball
kite
baseball bat
baseball glove
skateboard
surfboard, longboard, skimboard, shortboard, wakeboard
tennis racket, racket
bottle
wine glass
cup
fork
knife, pocketknife, knive
spoon
bowl, container
banana
apple
sandwich, burger, sub, cheeseburger, hamburger
orange
broccoli
carrot
hot dog
pizza
donut, doughnut, bagel
cake, cheesecake, cupcake, shortcake, coffeecake, pancake
chair, seat, stool
couch, sofa, recliner, futon, loveseat, settee, chesterfield
potted plant, houseplant
bed
dining table, table, desk
toilet, urinal, commode, lavatory, potty
tv, monitor, televison, television
laptop, computer, notebook, netbook, lenovo, macbook, laptop computer
mouse
remote
keyboard
cell phone, mobile phone, phone, cellphone, telephone, phon, smartphone, iPhone
microwave
oven, stovetop, stove, stove top oven
toaster
sink
refrigerator, fridge, freezer
book
clock
vase
scissors
teddy bear, teddybear
hair drier, hairdryer
toothbrush
'''



def combine_coco_instances(annotation_path):
    if not os.path.exists('%s/instances_%s2014.json' %(annotation_path, 'val')):
        raise Exception("Please download MSCOCO instance annotations for val set")
    val_instances = json.load(open('%s/instances_%s2014.json' %(annotation_path, 'val')))

    # Normally CHAIR loads train + val, but for pure eval on val, val is often enough.
    # To keep it faithful to original, we attempt to load train if it exists.
    train_path = '%s/instances_%s2014.json' %(annotation_path, 'train')
    if os.path.exists(train_path):
        train_instances = json.load(open(train_path))
        all_instances = {'info': train_instances['info'],
                         'licenses': train_instances['licenses'],
                         'type': train_instances.get('type', ''),
                         'categories': train_instances['categories'],
                         'images': train_instances['images'] + val_instances['images'],
                         'annotations': val_instances['annotations'] + train_instances['annotations']}
    else:
        all_instances = val_instances
    return all_instances

class CHAIR(object):
    def __init__(self, coco_path):
        self.imid_to_objects = defaultdict(list)
        self.coco_path = coco_path

        synonyms = synonyms_txt.strip().splitlines()
        synonyms = [s.strip().split(', ') for s in synonyms]
        self.mscoco_objects = []
        self.inverse_synonym_dict = {}
        for synonym in synonyms:
            self.mscoco_objects.extend(synonym)
            for s in synonym:
                self.inverse_synonym_dict[s] = synonym[0]

        coco_double_words = [
            'motor bike', 'motor cycle', 'air plane', 'traffic light',
            'street light', 'traffic signal', 'stop light', 'fire hydrant',
            'stop sign', 'parking meter', 'suit case', 'sports ball',
            'baseball bat', 'baseball glove', 'tennis racket', 'wine glass',
            'hot dog', 'cell phone', 'mobile phone', 'teddy bear',
            'hair drier', 'potted plant', 'bow tie', 'laptop computer',
            'stove top oven', 'home plate', 'train track'
        ]
        animal_words = ['bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'animal', 'cub']
        vehicle_words = ['jet', 'train']

        self.double_word_dict = {}
        for double_word in coco_double_words:
            self.double_word_dict[double_word] = double_word
        for animal_word in animal_words:
            self.double_word_dict['baby %s' %animal_word] = animal_word
            self.double_word_dict['adult %s' %animal_word] = animal_word
        for vehicle_word in vehicle_words:
            self.double_word_dict['passenger %s' %vehicle_word] = vehicle_word
        self.double_word_dict['bow tie'] = 'tie'
        self.double_word_dict['toilet seat'] = 'toilet'
        self.double_word_dict['wine glas'] = 'wine glass'

        self.get_annotations_from_segments()

    def caption_to_words(self, caption):
        words = nltk.word_tokenize(caption.lower())
        words = [singularize(w) for w in words]

        i = 0
        double_words = []
        idxs = []
        while i < len(words):
            idxs.append(i)
            double_word = ' '.join(words[i:i+2])
            if double_word in self.double_word_dict:
                double_words.append(self.double_word_dict[double_word])
                i += 2
            else:
                double_words.append(words[i])
                i += 1
        words = double_words

        if ('toilet' in words) & ('seat' in words): words = [word for word in words if word != 'seat']

        idxs = [idxs[idx] for idx, word in enumerate(words) if word in set(self.mscoco_objects)]
        words = [word for word in words if word in set(self.mscoco_objects)]
        node_words = []
        for word in words:
            node_words.append(self.inverse_synonym_dict[word])
        return words, node_words, idxs, double_words

    def get_annotations_from_segments(self):
        coco_segments = combine_coco_instances(self.coco_path)
        segment_annotations = coco_segments['annotations']
        id_to_name = {cat['id']: cat['name'] for cat in coco_segments['categories']}

        for i, annotation in enumerate(segment_annotations):
            if i % 10000 == 0:
                sys.stdout.write("\rGetting annotations for %d/%d segmentation masks" %(i, len(segment_annotations)))
            imid = annotation['image_id']
            # Only add categories that exist in our MSCOCO mapping
            cat_name = id_to_name[annotation['category_id']]
            if cat_name in self.inverse_synonym_dict:
                node_word = self.inverse_synonym_dict[cat_name]
                self.imid_to_objects[imid].append(node_word)
        print("\n")

        for imid in self.imid_to_objects:
            self.imid_to_objects[imid] = set(self.imid_to_objects[imid])

    def compute_hallucinations(self, imid, cap):
        words, node_words, idxs, raw_words = self.caption_to_words(cap)
        gt_objects = self.imid_to_objects.get(imid, set())

        cap_dict = {
            'mscoco_hallucinated_words': [],
            'mscoco_gt_words': list(gt_objects),
            'mscoco_generated_words': list(node_words),
            'hallucinated_words': 0,
            'recall_words': [],
        }

        for word, node_word, idx in zip(words, node_words, idxs):
            if node_word not in gt_objects:
                cap_dict['hallucinated_words'] += 1
                cap_dict['mscoco_hallucinated_words'].append(node_word) # Appending just the node word
            else:
                cap_dict['recall_words'].append(node_word) # Appending just the node word

        # Deduplicate to prevent double-counting multiple mentions of the same object in one caption
        cap_dict['mscoco_hallucinated_words'] = list(set(cap_dict['mscoco_hallucinated_words']))
        cap_dict['recall_words'] = list(set(cap_dict['recall_words']))

        return cap_dict

# =====================================================================
# 2. EVALUATION METRICS (CHAIR + Coverage + Precision + F1)
# =====================================================================

def compute_chair_score(annotations: dict) -> dict:
    total      = sum(1 for a in annotations.values() if a["generated_caption"])
    has_hall   = sum(1 for a in annotations.values() if a["hallucinated_words"])

    # FIXED: Convert to sets to prevent double-counting repeated words in a single caption
    total_hall = sum(len(set(a["hallucinated_words"])) for a in annotations.values())
    total_gnd  = sum(len(set(a["grounded_words"]))     for a in annotations.values())

    # ── CHAIR ──────────────────────────────────────────────────
    chair_s = has_hall / total * 100               if total > 0 else 0.0
    chair_i = total_hall / (total_hall + total_gnd + 1e-8) * 100

    # ── Coverage / Precision / F1 — computed per image then averaged ──
    coverages, precisions, f1s = [], [], []

    for a in annotations.values():
        if not a["generated_caption"]:
            continue

        annotated = set(a["objects_in_image"])          # ground truth
        mentioned = set(a["grounded_words"]) | set(a["hallucinated_words"])  # all mentioned
        correct   = set(a["grounded_words"])            # mentioned AND in image

        cov = len(correct) / len(annotated)              if annotated  else 0.0
        pre = len(correct) / len(mentioned)              if mentioned  else 0.0
        f1  = (2 * pre * cov) / (pre + cov)             if (pre + cov) > 0 else 0.0

        coverages.append(cov)
        precisions.append(pre)
        f1s.append(f1)

    coverage  = float(np.mean(coverages))  * 100 if coverages  else 0.0
    precision = float(np.mean(precisions)) * 100 if precisions else 0.0
    f1_score  = float(np.mean(f1s))        * 100 if f1s        else 0.0

    print(f"\nCHAIR + Coverage Summary:")
    print(f"  Images processed         : {total}")
    print(f"  Images with hallucination: {has_hall} ({chair_s:.1f}%)")
    print(f"  Total unique hallucinated words : {total_hall}")
    print(f"  Total unique grounded words     : {total_gnd}")
    print(f"  CHAIR_S                  : {chair_s:.1f}%")
    print(f"  CHAIR_I                  : {chair_i:.1f}%")
    print(f"  Coverage (Recall)        : {coverage:.1f}%")
    print(f"  Precision                : {precision:.1f}%")
    print(f"  F1                       : {f1_score:.1f}%")

    return {
        "chair_s"  : chair_s,
        "chair_i"  : chair_i,
        "coverage" : coverage,
        "precision": precision,
        "f1"       : f1_score,
        "total"    : total,
    }

# =====================================================================
# 3. PIPELINE: GENERATION AND LABELING
# =====================================================================
def build_chair_annotations(
    coco: COCO,
    img_ids: list,
    evaluator: CHAIR, # <-- Add this parameter
    save_path: str = "chair_annotations.json",
) -> dict:

    annotations = {}
    for img_id in tqdm(img_ids, desc="Building CHAIR annotations"):
        img_info = coco.loadImgs(img_id)[0]

        # USE THE EVALUATOR'S GROUND TRUTH INSTEAD OF RAW COCO
        objects = list(evaluator.imid_to_objects.get(img_id, set()))

        key = str(img_id).zfill(12)
        annotations[key] = {
            "image_id"         : img_id,
            "file_name"        : img_info["file_name"],
            "objects_in_image" : objects,
            "hallucinated_words": [],
            "grounded_words"    : [],
            "generated_caption" : "",
        }

    with open(save_path, "w") as f:
        json.dump(annotations, f, indent=2)

    return annotations


def generate_and_label_hallucinations(
    model,
    processor,
    annotations: dict,
    evaluator: CHAIR,
    image_dir: str = "coco/images/val2014",
    save_path: str = "chair_annotations.json",
    max_new_tokens: int = MAX_TOKENS,
) -> dict:

    prompt_template = "USER: <image>\nDescribe this image in detail.\nASSISTANT:"

    for key, ann in tqdm(annotations.items(), desc="Generating captions"):
        image_path = os.path.join(image_dir, ann["file_name"])
        if not os.path.exists(image_path):
            continue

        try:
            image    = Image.open(image_path).convert("RGB")
            inputs   = processor(images=image, text=prompt_template, return_tensors="pt")
            inputs   = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs, max_new_tokens=max_new_tokens, do_sample=False
                )
            caption = processor.decode(
                output_ids[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            ).lower()

            ann["generated_caption"] = caption

            chair_result = evaluator.compute_hallucinations(ann["image_id"], caption)

            ann["hallucinated_words"] = list(set(chair_result['mscoco_hallucinated_words']))
            ann["grounded_words"]     = list(set(chair_result['recall_words']))

        except Exception as e:
            print(f"Error on {key}: {e}")
            continue

    with open(save_path, "w") as f:
        json.dump(annotations, f, indent=2)

    return annotations
