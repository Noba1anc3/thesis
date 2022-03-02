from PIL import Image
from matplotlib import projections
import numpy as np
import os
import json
from torchvision.transforms import ToTensor
import torch
import torchvision

def resnet101(test):
    if test:
        image_folder = "/home/dreamaker/thesis/thesis/SG_Dataset/" + "test" + "_sen/image"
    else:
        image_folder = "/home/dreamaker/thesis/thesis/SG_Dataset/" + "train" + "_sen/image"

    output_folder = image_folder[:-5] + "resnet101"
    if not os.path.exists(output_folder): os.mkdir(output_folder)
    for i, image_name in enumerate(os.listdir(image_folder)):
        print(i)
        if os.path.exists(os.path.join(output_folder, image_name[:-4])): continue
        image_file = os.path.join(image_folder, image_name)
        json_file = os.path.join(image_folder[:-5] + "json", image_name[:-3] + "json")
        json_content = json.load(open(json_file))

        bboxes = [item[list(item.keys())[0]]["locations"] for item in json_content["items"]]

        image = Image.open(image_file)
        image = image.convert("RGB")
        # resize image
        target_size = 224
        resized_image = image.copy().resize((target_size, target_size))

        # resize corresponding bounding boxes (annotations)
        # Thanks, Stackoverflow: https://stackoverflow.com/questions/49466033/resizing-image-and-its-bounding-box
        def resize_bounding_box(bbox, original_image, target_size):
            x_, y_ = original_image.size

            x_scale = target_size / x_ 
            y_scale = target_size / y_
            
            origLeft, origTop, origRight, origBottom = bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1]
            
            x = int(np.round(origLeft * x_scale))
            y = int(np.round(origTop * y_scale))
            xmax = int(np.round(origRight * x_scale))
            ymax = int(np.round(origBottom * y_scale))
            
            return [x, y, xmax, ymax]

        resized_bounding_boxes = [resize_bounding_box(bbox, image, target_size) for bbox in bboxes]

        image = ToTensor()(resized_image).unsqueeze(0)
        model = torchvision.models.resnet101(pretrained=True)
        model = torch.nn.Sequential(*(list(model.children())[:-3]))

        with torch.no_grad():
            feature_map = model(image)
        
        output_size = (3, 3)
        spatial_scale = feature_map.shape[2] / target_size
        sampling_ratio = 2

        roi_align = torchvision.ops.RoIAlign(output_size,spatial_scale, sampling_ratio)

        def align_bounding_boxes(bboxes):
            aligned_bounding_boxes = []
            for bbox in bboxes:
                aligned_bbox = [bbox[0] - 0.5, bbox[1] - 0.5, bbox[2] + 0.5, bbox[3] + 0.5]
                aligned_bounding_boxes.append(aligned_bbox)

            return aligned_bounding_boxes
        
        feature_maps_bboxes = roi_align(input=feature_map, 
                                    rois=torch.tensor(
                                        [[0] + bbox 
                                        for bbox in align_bounding_boxes(resized_bounding_boxes)])
                                    .float()
                        )
        visual_embedding = torch.flatten(feature_maps_bboxes, 1)
        projection = torch.nn.Linear(in_features = visual_embedding.shape[-1], out_features = 768)
        output = projection(visual_embedding)
        
        torch.save(output, os.path.join(output_folder, image_name.split(".png")[0]))

def alexnet(test):
    if test:
        image_folder = "/home/dreamaker/thesis/thesis/SG_Dataset/" + "test" + "_sen/image"
    else:
        image_folder = "/home/dreamaker/thesis/thesis/SG_Dataset/" + "train" + "_sen/image"

    output_folder = image_folder[:-5] + "alexnet"
    if not os.path.exists(output_folder): os.mkdir(output_folder)
    for i, image_name in enumerate(os.listdir(image_folder)):
        print(i)
        if os.path.exists(os.path.join(output_folder, image_name[:-4])): continue
        image_file = os.path.join(image_folder, image_name)
        json_file = os.path.join(image_folder[:-5] + "json", image_name[:-3] + "json")
        json_content = json.load(open(json_file))

        bboxes = [item[list(item.keys())[0]]["locations"] for item in json_content["items"]]

        image = Image.open(image_file)
        image = image.convert("RGB")
        # resize image
        target_size = 224
        resized_image = image.copy().resize((target_size, target_size))

        # resize corresponding bounding boxes (annotations)
        # Thanks, Stackoverflow: https://stackoverflow.com/questions/49466033/resizing-image-and-its-bounding-box
        def resize_bounding_box(bbox, original_image, target_size):
            x_, y_ = original_image.size

            x_scale = target_size / x_ 
            y_scale = target_size / y_
            
            origLeft, origTop, origRight, origBottom = bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1]
            
            x = int(np.round(origLeft * x_scale))
            y = int(np.round(origTop * y_scale))
            xmax = int(np.round(origRight * x_scale))
            ymax = int(np.round(origBottom * y_scale))
            
            return [x, y, xmax, ymax]

        resized_bounding_boxes = [resize_bounding_box(bbox, image, target_size) for bbox in bboxes]

        image = ToTensor()(resized_image).unsqueeze(0)
        model = torchvision.models.alexnet(pretrained=True)
        model = torch.nn.Sequential(*(list(model.children())[:-3]))

        with torch.no_grad():
            feature_map = model(image)
        
        output_size = (3, 3)
        spatial_scale = feature_map.shape[2] / target_size
        sampling_ratio = 2

        roi_align = torchvision.ops.RoIAlign(output_size,spatial_scale, sampling_ratio)

        def align_bounding_boxes(bboxes):
            aligned_bounding_boxes = []
            for bbox in bboxes:
                aligned_bbox = [bbox[0] - 0.5, bbox[1] - 0.5, bbox[2] + 0.5, bbox[3] + 0.5]
                aligned_bounding_boxes.append(aligned_bbox)

            return aligned_bounding_boxes
        
        feature_maps_bboxes = roi_align(input=feature_map, 
                                    rois=torch.tensor(
                                        [[0] + bbox 
                                        for bbox in align_bounding_boxes(resized_bounding_boxes)])
                                    .float()
                        )
        visual_embedding = torch.flatten(feature_maps_bboxes, 1)
        projection = torch.nn.Linear(in_features = visual_embedding.shape[-1], out_features = 768)
        output = projection(visual_embedding)
        
        torch.save(output, os.path.join(output_folder, image_name.split(".png")[0]))

def googlenet(test):
    if test:
        image_folder = "/home/dreamaker/thesis/thesis/SG_Dataset/" + "test" + "_sen/image"
    else:
        image_folder = "/home/dreamaker/thesis/thesis/SG_Dataset/" + "train" + "_sen/image"

    output_folder = image_folder[:-5] + "googlenet"
    if not os.path.exists(output_folder): os.mkdir(output_folder)
    for i, image_name in enumerate(os.listdir(image_folder)):
        print(i)
        if os.path.exists(os.path.join(output_folder, image_name[:-4])): continue
        image_file = os.path.join(image_folder, image_name)
        json_file = os.path.join(image_folder[:-5] + "json", image_name[:-3] + "json")
        json_content = json.load(open(json_file))

        bboxes = [item[list(item.keys())[0]]["locations"] for item in json_content["items"]]

        image = Image.open(image_file)
        image = image.convert("RGB")
        # resize image
        target_size = 224
        resized_image = image.copy().resize((target_size, target_size))

        # resize corresponding bounding boxes (annotations)
        # Thanks, Stackoverflow: https://stackoverflow.com/questions/49466033/resizing-image-and-its-bounding-box
        def resize_bounding_box(bbox, original_image, target_size):
            x_, y_ = original_image.size

            x_scale = target_size / x_ 
            y_scale = target_size / y_
            
            origLeft, origTop, origRight, origBottom = bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1]
            
            x = int(np.round(origLeft * x_scale))
            y = int(np.round(origTop * y_scale))
            xmax = int(np.round(origRight * x_scale))
            ymax = int(np.round(origBottom * y_scale))
            
            return [x, y, xmax, ymax]

        resized_bounding_boxes = [resize_bounding_box(bbox, image, target_size) for bbox in bboxes]

        image = ToTensor()(resized_image).unsqueeze(0)
        model = torchvision.models.googlenet(pretrained=True)
        model = torch.nn.Sequential(*(list(model.children())[:-3]))

        with torch.no_grad():
            feature_map = model(image)
        
        output_size = (3, 3)
        spatial_scale = feature_map.shape[2] / target_size
        sampling_ratio = 2

        roi_align = torchvision.ops.RoIAlign(output_size,spatial_scale, sampling_ratio)

        def align_bounding_boxes(bboxes):
            aligned_bounding_boxes = []
            for bbox in bboxes:
                aligned_bbox = [bbox[0] - 0.5, bbox[1] - 0.5, bbox[2] + 0.5, bbox[3] + 0.5]
                aligned_bounding_boxes.append(aligned_bbox)

            return aligned_bounding_boxes
        
        feature_maps_bboxes = roi_align(input=feature_map, 
                                    rois=torch.tensor(
                                        [[0] + bbox 
                                        for bbox in align_bounding_boxes(resized_bounding_boxes)])
                                    .float()
                        )
        visual_embedding = torch.flatten(feature_maps_bboxes, 1)
        projection = torch.nn.Linear(in_features = visual_embedding.shape[-1], out_features = 768)
        output = projection(visual_embedding)
        
        torch.save(output, os.path.join(output_folder, image_name.split(".png")[0]))

def resnet18(test):
    if test:
        image_folder = "/home/dreamaker/thesis/thesis/SG_Dataset/" + "test" + "_sen/image"
    else:
        image_folder = "/home/dreamaker/thesis/thesis/SG_Dataset/" + "train" + "_sen/image"

    output_folder = image_folder[:-5] + "resnet18"
    if not os.path.exists(output_folder): os.mkdir(output_folder)
    for i, image_name in enumerate(os.listdir(image_folder)):
        print(i)
        if os.path.exists(os.path.join(output_folder, image_name[:-4])): continue
        image_file = os.path.join(image_folder, image_name)
        json_file = os.path.join(image_folder[:-5] + "json", image_name[:-3] + "json")
        json_content = json.load(open(json_file))

        bboxes = [item[list(item.keys())[0]]["locations"] for item in json_content["items"]]

        image = Image.open(image_file)
        image = image.convert("RGB")
        # resize image
        target_size = 224
        resized_image = image.copy().resize((target_size, target_size))

        # resize corresponding bounding boxes (annotations)
        # Thanks, Stackoverflow: https://stackoverflow.com/questions/49466033/resizing-image-and-its-bounding-box
        def resize_bounding_box(bbox, original_image, target_size):
            x_, y_ = original_image.size

            x_scale = target_size / x_ 
            y_scale = target_size / y_
            
            origLeft, origTop, origRight, origBottom = bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1]
            
            x = int(np.round(origLeft * x_scale))
            y = int(np.round(origTop * y_scale))
            xmax = int(np.round(origRight * x_scale))
            ymax = int(np.round(origBottom * y_scale))
            
            return [x, y, xmax, ymax]

        resized_bounding_boxes = [resize_bounding_box(bbox, image, target_size) for bbox in bboxes]

        image = ToTensor()(resized_image).unsqueeze(0)
        model = torchvision.models.resnet18(pretrained=True)
        model = torch.nn.Sequential(*(list(model.children())[:-3]))

        with torch.no_grad():
            feature_map = model(image)
        
        output_size = (3, 3)
        spatial_scale = feature_map.shape[2] / target_size
        sampling_ratio = 2

        roi_align = torchvision.ops.RoIAlign(output_size,spatial_scale, sampling_ratio)

        def align_bounding_boxes(bboxes):
            aligned_bounding_boxes = []
            for bbox in bboxes:
                aligned_bbox = [bbox[0] - 0.5, bbox[1] - 0.5, bbox[2] + 0.5, bbox[3] + 0.5]
                aligned_bounding_boxes.append(aligned_bbox)

            return aligned_bounding_boxes
        
        feature_maps_bboxes = roi_align(input=feature_map, 
                                    rois=torch.tensor(
                                        [[0] + bbox 
                                        for bbox in align_bounding_boxes(resized_bounding_boxes)])
                                    .float()
                        )
        visual_embedding = torch.flatten(feature_maps_bboxes, 1)
        projection = torch.nn.Linear(in_features = visual_embedding.shape[-1], out_features = 768)
        output = projection(visual_embedding)
        
        torch.save(output, os.path.join(output_folder, image_name.split(".png")[0]))

def resnet50(test):
    if test:
        image_folder = "/home/dreamaker/thesis/thesis/SG_Dataset/" + "test" + "_sen/image"
    else:
        image_folder = "/home/dreamaker/thesis/thesis/SG_Dataset/" + "train" + "_sen/image"

    output_folder = image_folder[:-5] + "resnet50"
    if not os.path.exists(output_folder): os.mkdir(output_folder)
    for i, image_name in enumerate(os.listdir(image_folder)):
        print(i)
        if os.path.exists(os.path.join(output_folder, image_name[:-4])): continue
        image_file = os.path.join(image_folder, image_name)
        json_file = os.path.join(image_folder[:-5] + "json", image_name[:-3] + "json")
        json_content = json.load(open(json_file))

        bboxes = [item[list(item.keys())[0]]["locations"] for item in json_content["items"]]

        image = Image.open(image_file)
        image = image.convert("RGB")
        # resize image
        target_size = 224
        resized_image = image.copy().resize((target_size, target_size))

        # resize corresponding bounding boxes (annotations)
        # Thanks, Stackoverflow: https://stackoverflow.com/questions/49466033/resizing-image-and-its-bounding-box
        def resize_bounding_box(bbox, original_image, target_size):
            x_, y_ = original_image.size

            x_scale = target_size / x_ 
            y_scale = target_size / y_
            
            origLeft, origTop, origRight, origBottom = bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1]
            
            x = int(np.round(origLeft * x_scale))
            y = int(np.round(origTop * y_scale))
            xmax = int(np.round(origRight * x_scale))
            ymax = int(np.round(origBottom * y_scale))
            
            return [x, y, xmax, ymax]

        resized_bounding_boxes = [resize_bounding_box(bbox, image, target_size) for bbox in bboxes]

        image = ToTensor()(resized_image).unsqueeze(0)
        model = torchvision.models.resnet50(pretrained=True)
        model = torch.nn.Sequential(*(list(model.children())[:-3]))

        with torch.no_grad():
            feature_map = model(image)
        
        output_size = (3, 3)
        spatial_scale = feature_map.shape[2] / target_size
        sampling_ratio = 2

        roi_align = torchvision.ops.RoIAlign(output_size,spatial_scale, sampling_ratio)

        def align_bounding_boxes(bboxes):
            aligned_bounding_boxes = []
            for bbox in bboxes:
                aligned_bbox = [bbox[0] - 0.5, bbox[1] - 0.5, bbox[2] + 0.5, bbox[3] + 0.5]
                aligned_bounding_boxes.append(aligned_bbox)

            return aligned_bounding_boxes
        
        feature_maps_bboxes = roi_align(input=feature_map, 
                                    rois=torch.tensor(
                                        [[0] + bbox 
                                        for bbox in align_bounding_boxes(resized_bounding_boxes)])
                                    .float()
                        )
        visual_embedding = torch.flatten(feature_maps_bboxes, 1)
        projection = torch.nn.Linear(in_features = visual_embedding.shape[-1], out_features = 768)
        output = projection(visual_embedding)
        
        torch.save(output, os.path.join(output_folder, image_name.split(".png")[0]))

def vgg16(test):
    if test:
        image_folder = "/home/dreamaker/thesis/thesis/SG_Dataset/" + "test" + "_sen/image"
    else:
        image_folder = "/home/dreamaker/thesis/thesis/SG_Dataset/" + "train" + "_sen/image"

    output_folder = image_folder[:-5] + "vgg16"
    if not os.path.exists(output_folder): os.mkdir(output_folder)
    for i, image_name in enumerate(os.listdir(image_folder)):
        print(i)
        if os.path.exists(os.path.join(output_folder, image_name[:-4])): continue
        image_file = os.path.join(image_folder, image_name)
        json_file = os.path.join(image_folder[:-5] + "json", image_name[:-3] + "json")
        json_content = json.load(open(json_file))

        bboxes = [item[list(item.keys())[0]]["locations"] for item in json_content["items"]]

        image = Image.open(image_file)
        image = image.convert("RGB")
        # resize image
        target_size = 224
        resized_image = image.copy().resize((target_size, target_size))

        # resize corresponding bounding boxes (annotations)
        # Thanks, Stackoverflow: https://stackoverflow.com/questions/49466033/resizing-image-and-its-bounding-box
        def resize_bounding_box(bbox, original_image, target_size):
            x_, y_ = original_image.size

            x_scale = target_size / x_ 
            y_scale = target_size / y_
            
            origLeft, origTop, origRight, origBottom = bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1]
            
            x = int(np.round(origLeft * x_scale))
            y = int(np.round(origTop * y_scale))
            xmax = int(np.round(origRight * x_scale))
            ymax = int(np.round(origBottom * y_scale))
            
            return [x, y, xmax, ymax]

        resized_bounding_boxes = [resize_bounding_box(bbox, image, target_size) for bbox in bboxes]

        image = ToTensor()(resized_image).unsqueeze(0)
        model = torchvision.models.vgg16(pretrained=True)
        model = torch.nn.Sequential(*(list(model.children())[:-3]))

        with torch.no_grad():
            feature_map = model(image)
        
        output_size = (3, 3)
        spatial_scale = feature_map.shape[2] / target_size
        sampling_ratio = 2

        roi_align = torchvision.ops.RoIAlign(output_size,spatial_scale, sampling_ratio)

        def align_bounding_boxes(bboxes):
            aligned_bounding_boxes = []
            for bbox in bboxes:
                aligned_bbox = [bbox[0] - 0.5, bbox[1] - 0.5, bbox[2] + 0.5, bbox[3] + 0.5]
                aligned_bounding_boxes.append(aligned_bbox)

            return aligned_bounding_boxes
        
        feature_maps_bboxes = roi_align(input=feature_map, 
                                    rois=torch.tensor(
                                        [[0] + bbox 
                                        for bbox in align_bounding_boxes(resized_bounding_boxes)])
                                    .float()
                        )
        visual_embedding = torch.flatten(feature_maps_bboxes, 1)
        projection = torch.nn.Linear(in_features = visual_embedding.shape[-1], out_features = 768)
        output = projection(visual_embedding)
        
        torch.save(output, os.path.join(output_folder, image_name.split(".png")[0]))

def mobilenet_v2(test):
    if test:
        image_folder = "/home/dreamaker/thesis/thesis/SG_Dataset/" + "test" + "_sen/image"
    else:
        image_folder = "/home/dreamaker/thesis/thesis/SG_Dataset/" + "train" + "_sen/image"

    output_folder = image_folder[:-5] + "mobilenet_v2"
    if not os.path.exists(output_folder): os.mkdir(output_folder)
    for i, image_name in enumerate(os.listdir(image_folder)):
        print(i)
        if os.path.exists(os.path.join(output_folder, image_name[:-4])): continue
        image_file = os.path.join(image_folder, image_name)
        json_file = os.path.join(image_folder[:-5] + "json", image_name[:-3] + "json")
        json_content = json.load(open(json_file))

        bboxes = [item[list(item.keys())[0]]["locations"] for item in json_content["items"]]

        image = Image.open(image_file)
        image = image.convert("RGB")
        # resize image
        target_size = 224
        resized_image = image.copy().resize((target_size, target_size))

        # resize corresponding bounding boxes (annotations)
        # Thanks, Stackoverflow: https://stackoverflow.com/questions/49466033/resizing-image-and-its-bounding-box
        def resize_bounding_box(bbox, original_image, target_size):
            x_, y_ = original_image.size

            x_scale = target_size / x_ 
            y_scale = target_size / y_
            
            origLeft, origTop, origRight, origBottom = bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1]
            
            x = int(np.round(origLeft * x_scale))
            y = int(np.round(origTop * y_scale))
            xmax = int(np.round(origRight * x_scale))
            ymax = int(np.round(origBottom * y_scale))
            
            return [x, y, xmax, ymax]

        resized_bounding_boxes = [resize_bounding_box(bbox, image, target_size) for bbox in bboxes]

        image = ToTensor()(resized_image).unsqueeze(0)
        model = torchvision.models.mobilenet_v2(pretrained=True)
        model = torch.nn.Sequential(*(list(model.children())[:-3]))

        with torch.no_grad():
            feature_map = model(image)
        
        output_size = (3, 3)
        spatial_scale = feature_map.shape[2] / target_size
        sampling_ratio = 2

        roi_align = torchvision.ops.RoIAlign(output_size,spatial_scale, sampling_ratio)

        def align_bounding_boxes(bboxes):
            aligned_bounding_boxes = []
            for bbox in bboxes:
                aligned_bbox = [bbox[0] - 0.5, bbox[1] - 0.5, bbox[2] + 0.5, bbox[3] + 0.5]
                aligned_bounding_boxes.append(aligned_bbox)

            return aligned_bounding_boxes
        
        feature_maps_bboxes = roi_align(input=feature_map, 
                                    rois=torch.tensor(
                                        [[0] + bbox 
                                        for bbox in align_bounding_boxes(resized_bounding_boxes)])
                                    .float()
                        )
        visual_embedding = torch.flatten(feature_maps_bboxes, 1)
        projection = torch.nn.Linear(in_features = visual_embedding.shape[-1], out_features = 768)
        output = projection(visual_embedding)
        
        torch.save(output, os.path.join(output_folder, image_name.split(".png")[0]))

# resnet101(1)
# resnet101(0)

# alexnet(1)
# alexnet(0)

# googlenet(1)
# googlenet(0)

# resnet18(1)
# resnet18(0)

# resnet50(1)
# resnet50(0)

# vgg16(1)
# vgg16(0)

# mobilenet_v2(1)
# mobilenet_v2(0)

a = torch.load("/home/dreamaker/thesis/thesis/SG_Dataset/test_sen/alexnet/ACTMAX_1")
print(1)