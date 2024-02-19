# @https://github.com/microsoft/table-transformer
from transformers import AutoModelForObjectDetection
from transformers import TableTransformerForObjectDetection
import easyocr
import torch
from tqdm.auto import tqdm
import numpy as np

# Image operations
from PIL import Image
from torchvision import transforms

import csv
import json

import os

device = "cuda" if torch.cuda.is_available() else "cpu"


class MaxResize(object):
    def __init__(self, max_size=800):
        self.max_size = max_size

    def __call__(self, image):
        width, height = image.size
        current_max_size = max(width, height)
        scale = self.max_size / current_max_size
        resized_image = image.resize((int(round(scale*width)), int(round(scale*height))))

        return resized_image


class TableExtraction:

    def __init__(self, image_dir: str):
        self.image_dir = image_dir

        self.table_detection_model = AutoModelForObjectDetection.from_pretrained(
            "microsoft/table-transformer-detection",
            revision="no_timm")

        self.table_structure_model = TableTransformerForObjectDetection.from_pretrained(
            "microsoft/table-structure-recognition-v1.1-all")

        self.table_reader = easyocr.Reader(['en'])

    def __load_image__(self,file_path):
        image = Image.open(file_path).convert("RGB")
        return image

    def __preprocess_image__(self, image):


        # transform image for detection
        detection_transform = transforms.Compose([
            MaxResize(800),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        detection_transform_pixel_values = detection_transform(image).unsqueeze(0)

        detection_transform_pixel_values = detection_transform_pixel_values.to(device)

        return detection_transform_pixel_values

    def __detect_table_rescale_bboxes__(self, out_bbox, size):
        img_w, img_h = size
        x_c, y_c, w, h = out_bbox.unbind(-1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
        b = torch.stack(b, dim=1)
        b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
        return b


    def __get_image_object__(self, outputs, img_size, id2label):
        m = outputs.logits.softmax(-1).max(-1)
        pred_labels = list(m.indices.detach().cpu().numpy())[0]
        pred_scores = list(m.values.detach().cpu().numpy())[0]
        pred_bboxes = outputs['pred_boxes'].detach().cpu()[0]
        pred_bboxes = [elem.tolist() for elem in self.__detect_table_rescale_bboxes__(pred_bboxes, img_size)]

        objects = []
        for label, score, bbox in zip(pred_labels, pred_scores, pred_bboxes):
            class_label = id2label[int(label)]
            if not class_label == 'no object':
                objects.append({'label': class_label, 'score': float(score),
                                'bbox': [float(elem) for elem in bbox]})

        return objects

    def __detect_table__(self, image, image_pixel_values):

        with torch.no_grad():
            outputs = self.table_detection_model(image_pixel_values)

        id2label = self.table_detection_model.config.id2label
        id2label[len(self.table_detection_model.config.id2label)] = "no object"

        image_object = self.__get_image_object__(outputs, image.size, id2label)

        return image_object

    def __crop_table__(self, img, objects):
        tokens = []
        class_thresholds = {
            "table": 0.5,
            "table rotated": 0.5,
            "no object": 10
        }
        padding = 25

        table_crops = []
        for obj in objects:
            if obj['score'] < class_thresholds[obj['label']]:
                continue

            cropped_table = {}

            bbox = obj['bbox']
            bbox = [bbox[0] - padding, bbox[1] - padding, bbox[2] + padding, bbox[3] + padding]

            cropped_img = img.crop(bbox)

            table_tokens = [token for token in tokens if iob(token['bbox'], bbox) >= 0.5]
            for token in table_tokens:
                token['bbox'] = [token['bbox'][0] - bbox[0],
                                 token['bbox'][1] - bbox[1],
                                 token['bbox'][2] - bbox[0],
                                 token['bbox'][3] - bbox[1]]

            # If table is predicted to be rotated, rotate cropped image and tokens/words:
            if obj['label'] == 'table rotated':
                cropped_img = cropped_img.rotate(270, expand=True)
                for token in table_tokens:
                    bbox = token['bbox']
                    bbox = [cropped_img.size[0] - bbox[3] - 1,
                            bbox[0],
                            cropped_img.size[0] - bbox[1] - 1,
                            bbox[2]]
                    token['bbox'] = bbox

            cropped_table['image'] = cropped_img
            cropped_table['tokens'] = table_tokens

            table_crops.append(cropped_table)

        return table_crops

    def __get_table_structure__(self,tab_image):
        self.table_structure_model.to(device)

        structure_transform = transforms.Compose([
            MaxResize(800),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        structure_transform_pixel_values = structure_transform(tab_image).unsqueeze(0)

        structure_transform_pixel_values = structure_transform_pixel_values.to(device)

        structure_id2label = self.table_structure_model.config.id2label
        structure_id2label[len(structure_id2label)] = "no object"

        with torch.no_grad():
            outputs = self.table_structure_model(structure_transform_pixel_values)

        cells = self.__get_image_object__(outputs, tab_image.size, structure_id2label)

        return cells

    def __get_cell_by_row__(self, table_data):
        # Extract rows and columns
        rows = [entry for entry in table_data if entry['label'] == 'table row']
        columns = [entry for entry in table_data if entry['label'] == 'table column']

        # Sort rows and columns by their Y and X coordinates, respectively
        rows.sort(key=lambda x: x['bbox'][1])
        columns.sort(key=lambda x: x['bbox'][0])

        # Function to find cell coordinates
        def find_cell_coordinates(row, column):
            cell_bbox = [column['bbox'][0], row['bbox'][1], column['bbox'][2], row['bbox'][3]]
            return cell_bbox

        # Generate cell coordinates and count cells in each row
        cell_coordinates = []

        for row in rows:
            row_cells = []
            for column in columns:
                cell_bbox = find_cell_coordinates(row, column)
                row_cells.append({'column': column['bbox'], 'cell': cell_bbox})

            # Sort cells in the row by X coordinate
            row_cells.sort(key=lambda x: x['column'][0])

            # Append row information to cell_coordinates
            cell_coordinates.append({'row': row['bbox'], 'cells': row_cells, 'cell_count': len(row_cells)})

        # Sort rows from top to bottom
        cell_coordinates.sort(key=lambda x: x['row'][1])

        return cell_coordinates

    def __ocr_table_data__(self, table_cells, table_img):
        cell_coordinates = self.__get_cell_by_row__(table_cells)
        # let's OCR row by row
        data = dict()
        max_num_columns = 0
        for idx, row in enumerate(tqdm(cell_coordinates)):
            row_text = []
            for cell in row["cells"]:
                # crop cell out of image
                cell_image = np.array(table_img.crop(cell["cell"]))
                # apply OCR
                result = self.table_reader.readtext(np.array(cell_image))
                if len(result) > 0:
                    # print([x[1] for x in list(result)])
                    text = " ".join([x[1] for x in result])
                    row_text.append(text)

            if len(row_text) > max_num_columns:
                max_num_columns = len(row_text)

            data[idx] = row_text

        print("Max number of columns:", max_num_columns)

        # pad rows which don't have max_num_columns elements
        # to make sure all rows have the same number of columns
        for row, row_data in data.copy().items():
            if len(row_data) != max_num_columns:
                row_data = row_data + ["" for _ in range(max_num_columns - len(row_data))]
            data[row] = row_data

        return data

    def __save_table_data__(self, data, file_name: str):
        with open(f'{file_name}.csv','w') as result_file:
            wr = csv.writer(result_file, dialect='excel')

            for row, row_text in data.items():
                wr.writerow(row_text)

        # function to dump data
        with open(f"{file_name}.json", 'w', encoding='utf-8') as jsonf:
            jsonf.write(json.dumps(data, indent=4))


    def get_table_data(self):
        for imgfile in os.listdir(self.image_dir):

            if not imgfile.lower().endswith("png"):
                continue
            else:
                image_path = os.path.join(self.image_dir,imgfile)

            # load image
            image = self.__load_image__(image_path)

            # preprocessing image for detection -> gives image object (scaled)
            preproceesed_image_pxl_val = self.__preprocess_image__(image)

            # detect table in image -> gives table bounding box
            detected_table_obj = self.__detect_table__(image, preproceesed_image_pxl_val)

            # crop table from image -> table object (convert to image)
            table_crops = self.__crop_table__(image, detected_table_obj)

            # get table structure -> gives cells boundaries
            cropped_tables = [self.__get_table_structure__(table['image'].convert("RGB")) for table in table_crops]

            cropped_tables_data = [
                                    self.__ocr_table_data__(cropped_table_str,cropped_table_img['image'].convert("RGB"))
                                        for cropped_table_str, cropped_table_img in zip(cropped_tables,table_crops)
                                  ]

            # # save data (json/csv)
            for ind,tab_data in enumerate(cropped_tables_data):
                self.__save_table_data__(tab_data,f"image_{ind}")



TblObj = TableExtraction(r"C:\Users\mehul\Documents\Projects - GIT\Agents\Data Governance\Code\sample_image")

TblObj.get_table_data()