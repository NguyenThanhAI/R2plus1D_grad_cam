import sys
import math
import json
import argparse
from pathlib import Path

import cv2
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from torchvision.transforms import Compose
from torchvision.models.video.resnet import VideoResNet, BasicBlock, R2Plus1dStem, Conv2Plus1D

from einops.layers.torch import Rearrange

from torch.nn import functional as F


def r2plus1d_34(num_classes):
    model = VideoResNet(block=BasicBlock,
                        conv_makers=[Conv2Plus1D] * 4,
                        layers=[3, 4, 6, 3],
                        stem=R2Plus1dStem)

    model.fc = nn.Linear(model.fc.in_features, out_features=num_classes)

    # Fix difference in PyTorch vs Caffe2 architecture
    # https://github.com/facebookresearch/VMZ/issues/89
    model.layer2[0].conv2[0] = Conv2Plus1D(128, 128, 288)
    model.layer3[0].conv2[0] = Conv2Plus1D(256, 256, 576)
    model.layer4[0].conv2[0] = Conv2Plus1D(512, 512, 1152)

    # We need exact Caffe2 momentum for BatchNorm scaling
    for m in model.modules():
        if isinstance(m, nn.BatchNorm3d):
            m.eps = 1e-3
            m.momentum = 0.9

    return model

def get_input_frames(args):
    cap = cv2.VideoCapture(args.video)

    frame_list = []

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.resize(frame, (112, 112))
        frame_list.append(frame.copy()[:, :, ::-1] / 255.)

    #print(frame_list)

    choice_list = range(len(frame_list) - 32 + 1)
    index = 30
    assert index < len(frame_list) - 31
    print("frame index:", index)
    frame_list = frame_list[index:index + 32]
    frames_to_show = frame_list.copy()
    #show_frames_on_figure(frames_to_show)
    preprocess = lambda x: (x - np.array([0.43216, 0.394666, 0.37645])[np.newaxis, np.newaxis, :]) / np.array([0.22803, 0.22145, 0.216989])[np.newaxis, np.newaxis, :]
    frame_list = list(map(preprocess, frame_list))
    frame_list = np.stack(frame_list, axis=0)
    frame_list = np.transpose(frame_list, axes=(3, 0, 1, 2))

    frame_list = torch.from_numpy(frame_list)
    frame_list = frame_list.unsqueeze(0).float().requires_grad_(True)
    return frame_list


class FrameRange:
    def __init__(self, video, first, last):
        assert first <= last

        for i in range(first):
            ret, _ = video.read()

            if not ret:
                raise RuntimeError("seeking to frame at index {} failed".format(i))

        self.video = video
        self.it = first
        self.last = last

    def __next__(self):
        if self.it >= self.last or not self.video.isOpened():
            raise StopIteration

        ok, frame = self.video.read()

        if not ok:
            raise RuntimeError("decoding frame at index {} failed".format(self.it))

        self.it += 1

        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


class BatchedRange:
    def __init__(self, rng, n):
        self.rng = rng
        self.n = n

    def __next__(self):
        ret = []

        for i in range(self.n):
            ret.append(next(self.rng))

        return ret


class TransformedRange:
    def __init__(self, rng, fn):
        self.rng = rng
        self.fn = fn

    def __next__(self):
        return self.fn(next(self.rng))


class VideoDataset(IterableDataset):
    def __init__(self, path, clip, transform=None):
        super().__init__()

        self.path = path
        self.clip = clip
        self.transform = transform

        video = cv2.VideoCapture(str(path))
        frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        video.release()

        self.first = 0
        self.last = frames

    def __iter__(self):
        info = get_worker_info()

        video = cv2.VideoCapture(str(self.path))

        if info is None:
            rng = FrameRange(video, self.first, self.last)
        else:
            per = int(math.ceil((self.last - self.first) / float(info.num_workers)))
            wid = info.id

            first = self.first + wid * per
            last = min(first + per, self.last)

            rng = FrameRange(video, first, last)

        if self.transform is not None:
            fn = self.transform
        else:
            fn = lambda v: v

        return TransformedRange(BatchedRange(rng, self.clip), fn)


class WebcamDataset(IterableDataset):
    def __init__(self, clip, transform=None):
        super().__init__()

        self.clip = clip
        self.transform = transform
        self.video = cv2.VideoCapture(0)

    def __iter__(self):
        info = get_worker_info()

        if info is not None:
            raise RuntimeError("multiple workers not supported in WebcamDataset")

        # treat webcam as fixed frame range for now: 10 minutes
        rng = FrameRange(self.video, 0, 30 * 60 * 10)

        if self.transform is not None:
            fn = self.transform
        else:
            fn = lambda v: v

        return TransformedRange(BatchedRange(rng, self.clip), fn)


class ToTensor:
    def __call__(self, x):
        print(torch.from_numpy(np.array(x)).float() / 255.)
        return torch.from_numpy(np.array(x)).float() / 255.


class Resize:
    def __init__(self, size, mode="bilinear"):
        self.size = size
        self.mode = mode

    def __call__(self, video):
        return torch.nn.functional.interpolate(video, size=self.size,
            mode=self.mode, align_corners=False)


class CenterCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, video):
        h, w = video.shape[-2:]
        th, tw = self.size

        i = int(round((h - th) / 2.))
        j = int(round((w - tw) / 2.))

        return video[..., i:(i + th), j:(j + tw)]


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, video):
        shape = (-1,) + (1,) * (video.dim() - 1)

        mean = torch.as_tensor(self.mean).reshape(shape)
        std = torch.as_tensor(self.std).reshape(shape)

        return (video - mean) / std


def main(args):
    labels = {0: 'abseiling', 1: 'air_drumming', 2: 'answering_questions', 3: 'applauding', 4: 'applying_cream',
                      5: 'archery', 6: 'arm_wrestling', 7: 'arranging_flowers', 8: 'assembling_computer',
                      9: 'auctioning', 10: 'baby_waking_up', 11: 'baking_cookies', 12: 'balloon_blowing',
                      13: 'bandaging', 14: 'barbequing', 15: 'bartending', 16: 'beatboxing', 17: 'bee_keeping',
                      18: 'belly_dancing', 19: 'bench_pressing', 20: 'bending_back', 21: 'bending_metal',
                      22: 'biking_through_snow', 23: 'blasting_sand', 24: 'blowing_glass', 25: 'blowing_leaves',
                      26: 'blowing_nose', 27: 'blowing_out_candles', 28: 'bobsledding', 29: 'bookbinding',
                      30: 'bouncing_on_trampoline', 31: 'bowling', 32: 'braiding_hair', 33: 'breading_or_breadcrumbing',
                      34: 'breakdancing', 35: 'brush_painting', 36: 'brushing_hair', 37: 'brushing_teeth',
                      38: 'building_cabinet', 39: 'building_shed', 40: 'bungee_jumping', 41: 'busking',
                      42: 'canoeing_or_kayaking', 43: 'capoeira', 44: 'carrying_baby', 45: 'cartwheeling',
                      46: 'carving_pumpkin', 47: 'catching_fish', 48: 'catching_or_throwing_baseball',
                      49: 'catching_or_throwing_frisbee', 50: 'catching_or_throwing_softball', 51: 'celebrating',
                      52: 'changing_oil', 53: 'changing_wheel', 54: 'checking_tires', 55: 'cheerleading',
                      56: 'chopping_wood', 57: 'clapping', 58: 'clay_pottery_making', 59: 'clean_and_jerk',
                      60: 'cleaning_floor', 61: 'cleaning_gutters', 62: 'cleaning_pool', 63: 'cleaning_shoes',
                      64: 'cleaning_toilet', 65: 'cleaning_windows', 66: 'climbing_a_rope', 67: 'climbing_ladder',
                      68: 'climbing_tree', 69: 'contact_juggling', 70: 'cooking_chicken', 71: 'cooking_egg',
                      72: 'cooking_on_campfire', 73: 'cooking_sausages', 74: 'counting_money',
                      75: 'country_line_dancing', 76: 'cracking_neck', 77: 'crawling_baby', 78: 'crossing_river',
                      79: 'crying', 80: 'curling_hair', 81: 'cutting_nails', 82: 'cutting_pineapple',
                      83: 'cutting_watermelon', 84: 'dancing_ballet', 85: 'dancing_charleston',
                      86: 'dancing_gangnam_style', 87: 'dancing_macarena', 88: 'deadlifting',
                      89: 'decorating_the_christmas_tree', 90: 'digging', 91: 'dining', 92: 'disc_golfing',
                      93: 'diving_cliff', 94: 'dodgeball', 95: 'doing_aerobics', 96: 'doing_laundry', 97: 'doing_nails',
                      98: 'drawing', 99: 'dribbling_basketball', 100: 'drinking', 101: 'drinking_beer',
                      102: 'drinking_shots', 103: 'driving_car', 104: 'driving_tractor', 105: 'drop_kicking',
                      106: 'drumming_fingers', 107: 'dunking_basketball', 108: 'dying_hair', 109: 'eating_burger',
                      110: 'eating_cake', 111: 'eating_carrots', 112: 'eating_chips', 113: 'eating_doughnuts',
                      114: 'eating_hotdog', 115: 'eating_ice_cream', 116: 'eating_spaghetti', 117: 'eating_watermelon',
                      118: 'egg_hunting', 119: 'exercising_arm', 120: 'exercising_with_an_exercise_ball',
                      121: 'extinguishing_fire', 122: 'faceplanting', 123: 'feeding_birds', 124: 'feeding_fish',
                      125: 'feeding_goats', 126: 'filling_eyebrows', 127: 'finger_snapping', 128: 'fixing_hair',
                      129: 'flipping_pancake', 130: 'flying_kite', 131: 'folding_clothes', 132: 'folding_napkins',
                      133: 'folding_paper', 134: 'front_raises', 135: 'frying_vegetables', 136: 'garbage_collecting',
                      137: 'gargling', 138: 'getting_a_haircut', 139: 'getting_a_tattoo',
                      140: 'giving_or_receiving_award', 141: 'golf_chipping', 142: 'golf_driving', 143: 'golf_putting',
                      144: 'grinding_meat', 145: 'grooming_dog', 146: 'grooming_horse', 147: 'gymnastics_tumbling',
                      148: 'hammer_throw', 149: 'headbanging', 150: 'headbutting', 151: 'high_jump', 152: 'high_kick',
                      153: 'hitting_baseball', 154: 'hockey_stop', 155: 'holding_snake', 156: 'hopscotch',
                      157: 'hoverboarding', 158: 'hugging', 159: 'hula_hooping', 160: 'hurdling',
                      161: 'hurling_-sport-', 162: 'ice_climbing', 163: 'ice_fishing', 164: 'ice_skating',
                      165: 'ironing', 166: 'javelin_throw', 167: 'jetskiing', 168: 'jogging', 169: 'juggling_balls',
                      170: 'juggling_fire', 171: 'juggling_soccer_ball', 172: 'jumping_into_pool',
                      173: 'jumpstyle_dancing', 174: 'kicking_field_goal', 175: 'kicking_soccer_ball', 176: 'kissing',
                      177: 'kitesurfing', 178: 'knitting', 179: 'krumping', 180: 'laughing', 181: 'laying_bricks',
                      182: 'long_jump', 183: 'lunge', 184: 'making_a_cake', 185: 'making_a_sandwich', 186: 'making_bed',
                      187: 'making_jewelry', 188: 'making_pizza', 189: 'making_snowman', 190: 'making_sushi',
                      191: 'making_tea', 192: 'marching', 193: 'massaging_back', 194: 'massaging_feet',
                      195: 'massaging_legs', 196: "massaging_person's_head", 197: 'milking_cow', 198: 'mopping_floor',
                      199: 'motorcycling', 200: 'moving_furniture', 201: 'mowing_lawn', 202: 'news_anchoring',
                      203: 'opening_bottle', 204: 'opening_present', 205: 'paragliding', 206: 'parasailing',
                      207: 'parkour', 208: 'passing_American_football_-in_game-',
                      209: 'passing_American_football_-not_in_game-', 210: 'peeling_apples', 211: 'peeling_potatoes',
                      212: 'petting_animal_-not_cat-', 213: 'petting_cat', 214: 'picking_fruit', 215: 'planting_trees',
                      216: 'plastering', 217: 'playing_accordion', 218: 'playing_badminton', 219: 'playing_bagpipes',
                      220: 'playing_basketball', 221: 'playing_bass_guitar', 222: 'playing_cards', 223: 'playing_cello',
                      224: 'playing_chess', 225: 'playing_clarinet', 226: 'playing_controller', 227: 'playing_cricket',
                      228: 'playing_cymbals', 229: 'playing_didgeridoo', 230: 'playing_drums', 231: 'playing_flute',
                      232: 'playing_guitar', 233: 'playing_harmonica', 234: 'playing_harp', 235: 'playing_ice_hockey',
                      236: 'playing_keyboard', 237: 'playing_kickball', 238: 'playing_monopoly', 239: 'playing_organ',
                      240: 'playing_paintball', 241: 'playing_piano', 242: 'playing_poker', 243: 'playing_recorder',
                      244: 'playing_saxophone', 245: 'playing_squash_or_racquetball', 246: 'playing_tennis',
                      247: 'playing_trombone', 248: 'playing_trumpet', 249: 'playing_ukulele', 250: 'playing_violin',
                      251: 'playing_volleyball', 252: 'playing_xylophone', 253: 'pole_vault',
                      254: 'presenting_weather_forecast', 255: 'pull_ups', 256: 'pumping_fist', 257: 'pumping_gas',
                      258: 'punching_bag', 259: 'punching_person_-boxing-', 260: 'push_up', 261: 'pushing_car',
                      262: 'pushing_cart', 263: 'pushing_wheelchair', 264: 'reading_book', 265: 'reading_newspaper',
                      266: 'recording_music', 267: 'riding_a_bike', 268: 'riding_camel', 269: 'riding_elephant',
                      270: 'riding_mechanical_bull', 271: 'riding_mountain_bike', 272: 'riding_mule',
                      273: 'riding_or_walking_with_horse', 274: 'riding_scooter', 275: 'riding_unicycle',
                      276: 'ripping_paper', 277: 'robot_dancing', 278: 'rock_climbing', 279: 'rock_scissors_paper',
                      280: 'roller_skating', 281: 'running_on_treadmill', 282: 'sailing', 283: 'salsa_dancing',
                      284: 'sanding_floor', 285: 'scrambling_eggs', 286: 'scuba_diving', 287: 'setting_table',
                      288: 'shaking_hands', 289: 'shaking_head', 290: 'sharpening_knives', 291: 'sharpening_pencil',
                      292: 'shaving_head', 293: 'shaving_legs', 294: 'shearing_sheep', 295: 'shining_shoes',
                      296: 'shooting_basketball', 297: 'shooting_goal_-soccer-', 298: 'shot_put', 299: 'shoveling_snow',
                      300: 'shredding_paper', 301: 'shuffling_cards', 302: 'side_kick',
                      303: 'sign_language_interpreting', 304: 'singing', 305: 'situp', 306: 'skateboarding',
                      307: 'ski_jumping', 308: 'skiing_-not_slalom_or_crosscountry-', 309: 'skiing_crosscountry',
                      310: 'skiing_slalom', 311: 'skipping_rope', 312: 'skydiving', 313: 'slacklining', 314: 'slapping',
                      315: 'sled_dog_racing', 316: 'smoking', 317: 'smoking_hookah', 318: 'snatch_weight_lifting',
                      319: 'sneezing', 320: 'sniffing', 321: 'snorkeling', 322: 'snowboarding', 323: 'snowkiting',
                      324: 'snowmobiling', 325: 'somersaulting', 326: 'spinning_poi', 327: 'spray_painting',
                      328: 'spraying', 329: 'springboard_diving', 330: 'squat', 331: 'sticking_tongue_out',
                      332: 'stomping_grapes', 333: 'stretching_arm', 334: 'stretching_leg', 335: 'strumming_guitar',
                      336: 'surfing_crowd', 337: 'surfing_water', 338: 'sweeping_floor', 339: 'swimming_backstroke',
                      340: 'swimming_breast_stroke', 341: 'swimming_butterfly_stroke', 342: 'swing_dancing',
                      343: 'swinging_legs', 344: 'swinging_on_something', 345: 'sword_fighting', 346: 'tai_chi',
                      347: 'taking_a_shower', 348: 'tango_dancing', 349: 'tap_dancing', 350: 'tapping_guitar',
                      351: 'tapping_pen', 352: 'tasting_beer', 353: 'tasting_food', 354: 'testifying', 355: 'texting',
                      356: 'throwing_axe', 357: 'throwing_ball', 358: 'throwing_discus', 359: 'tickling',
                      360: 'tobogganing', 361: 'tossing_coin', 362: 'tossing_salad', 363: 'training_dog',
                      364: 'trapezing', 365: 'trimming_or_shaving_beard', 366: 'trimming_trees', 367: 'triple_jump',
                      368: 'tying_bow_tie', 369: 'tying_knot_-not_on_a_tie-', 370: 'tying_tie', 371: 'unboxing',
                      372: 'unloading_truck', 373: 'using_computer', 374: 'using_remote_controller_-not_gaming-',
                      375: 'using_segway', 376: 'vault', 377: 'waiting_in_line', 378: 'walking_the_dog',
                      379: 'washing_dishes', 380: 'washing_feet', 381: 'washing_hair', 382: 'washing_hands',
                      383: 'water_skiing', 384: 'water_sliding', 385: 'watering_plants', 386: 'waxing_back',
                      387: 'waxing_chest', 388: 'waxing_eyebrows', 389: 'waxing_legs', 390: 'weaving_basket',
                      391: 'welding', 392: 'whistling', 393: 'windsurfing', 394: 'wrapping_present', 395: 'wrestling',
                      396: 'writing', 397: 'yawning', 398: 'yoga', 399: 'zumba'}

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #if torch.cuda.is_available():
    #    torch.backends.cudnn.benchmark = True

    model = r2plus1d_34(num_classes=args.classes)
    #model = model.to(device)

    #weights = torch.load(args.model, map_location=device)
    weights = torch.load(args.model)
    model.load_state_dict(weights)

    #model = nn.DataParallel(model)
    model.eval()

    inputs = get_input_frames(args)

    softmax = F.softmax(model(inputs), dim=1).cpu().data.numpy()[0]
    index = np.argmax(softmax)
    print(labels[index], softmax[index])

    transform = Compose([
        ToTensor(),
        Rearrange("t h w c -> c t h w"),
        Resize((128, 171)),
        Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]),
        CenterCrop((112, 112)),
    ])

    #dataset = WebcamDataset(args.frames, transform=transform)

    dataset = VideoDataset(args.video, args.frames, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    for inputs in loader:
        # NxCxTxHxW
        assert inputs.size() == (args.batch_size, 3, args.frames, 112, 112)

        #inputs = inputs.to(device)

        outputs = model(inputs)

        _, preds = torch.max(outputs, dim=1)
        preds = preds.data.cpu().numpy()

        scores = nn.functional.softmax(outputs, dim=1)
        scores = scores.data.cpu().numpy()

        for pred, score in zip(preds, scores):
            index = pred.item()
            label = labels[index]
            score = score.max().item()

            print("label='{}' score={}".format(label, score), file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    arg = parser.add_argument

    arg("--model", type=str, default="r2.5d_d34_l32.pth", help=".pth file to load model weights from")
    arg("--video", type=str, default=r"D:\kinetics400\videos_train\abseiling\_IkgLzKQVzk_000020_000030.mp4", help="video file to run feature extraction on")
    arg("--frames", type=int, choices=(8, 32), default=32, help="clip frames for video model")
    arg("--classes", type=int, choices=(400, 487), default=400, help="classes in last layer")
    arg("--batch-size", type=int, default=1, help="number of sequences per batch for inference")
    arg("--num-workers", type=int, default=0, help="number of workers for data loading")

    main(parser.parse_args())