import xml.etree.ElementTree as ET
from pathlib import Path
import os

SOURSE_PATH = Path(__file__).resolve().parent / 'sourse'
SOURSE_IMAGES_TRAIN_PATH = ''
SOURSE_IMAGES_VALIDATION_PATH = ''
SOURSE_LABELS_TRAIN_PATH = ''
SOURSE_LABELS_VALIDATION_PATH = ''

class_name_to_id_mapping = {"good": 0}


def extract_info_from_xml(xml_file, path):
    #print("extract_info_from_xml >", path + "" + xml_file)
    root = ET.parse(path + "/" + xml_file).getroot()

    # Initialise the info dict
    info_dict = {}
    info_dict['bboxes'] = []

    # Parse the XML Tree
    for elem in root:
        # Get the file name
        if elem.tag == "filename":
            info_dict['filename'] = elem.text

        # Get the image size
        elif elem.tag == "size":
            image_size = []
            for subelem in elem:
                image_size.append(int(subelem.text))

            info_dict['image_size'] = tuple(image_size)

        # Get details of the bounding box
        elif elem.tag == "object":
            bbox = {}
            for subelem in elem:
                if subelem.tag == "name":
                    bbox["class"] = subelem.text

                elif subelem.tag == "bndbox":
                    for subsubelem in subelem:
                        bbox[subsubelem.tag] = int(subsubelem.text)
            info_dict['bboxes'].append(bbox)

    return info_dict

def convert_to_yolov5(info_dict):
    print_buffer = []
    
    # For each bounding box
    for b in info_dict["bboxes"]:
        try:
            class_id = class_name_to_id_mapping[b["class"]]
        except KeyError:
            print("Invalid Class. Must be one from ", class_name_to_id_mapping.keys())
        
        # Transform the bbox co-ordinates as per the format required by YOLO v5
        b_center_x = (b["xmin"] + b["xmax"]) / 2 
        b_center_y = (b["ymin"] + b["ymax"]) / 2
        b_width    = (b["xmax"] - b["xmin"])
        b_height   = (b["ymax"] - b["ymin"])
        
        # Normalise the co-ordinates by the dimensions of the image
        image_w, image_h, _ = info_dict["image_size"]  
        b_center_x /= image_w 
        b_center_y /= image_h 
        b_width    /= image_w 
        b_height   /= image_h 
        
        #Write the bbox details to the file 
        print_buffer.append("{} {:.5f} {:.5f} {:.5f} {:.5f}".format(class_id, b_center_x, b_center_y, b_width, b_height))
        
    # Name of the file which we have to save 
    save_file_name = os.path.join("annotations", info_dict["filename"].replace("png", "txt"))
    
    # Save the annotation to disk
    #print("\n".join(print_buffer), file= open(save_file_name, "w"))

    return "\n".join(print_buffer)


def getDirectoryStructure(path):
    listDirPath = []
    for (dirpath, dirnames, filenames) in os.walk(path):
        listDirPath.append(dirpath)
    global SOURSE_IMAGES_TRAIN_PATH
    SOURSE_IMAGES_TRAIN_PATH = listDirPath[2]
    global SOURSE_IMAGES_VALIDATION_PATH
    SOURSE_IMAGES_VALIDATION_PATH = listDirPath[3]
    global SOURSE_LABELS_TRAIN_PATH
    SOURSE_LABELS_TRAIN_PATH = listDirPath[5]
    global SOURSE_LABELS_VALIDATION_PATH
    SOURSE_LABELS_VALIDATION_PATH = listDirPath[6]


def extractAllXml(path):
    for (dirpath, dirnames, filenames) in os.walk(path):
        for file in filenames:
            #extract_info_from_xml(file, path)
            if file == "frame13.xml":
                print(convert_to_yolov5(extract_info_from_xml(file, path)))


if __name__ == "__main__":
    getDirectoryStructure(SOURSE_PATH)

    extractAllXml(SOURSE_LABELS_TRAIN_PATH)
