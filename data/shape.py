import hashlib
import json
import os.path
import sys
import shutil
import uuid
import skimage.draw
import skimage.io

sys.path.append('../')
from config import cfg

num_images = 256

def md5sum(pathname, blocksize=65536):
    checksum = hashlib.md5()

    with open(pathname, "rb") as stream:
        for block in iter(lambda: stream.read(blocksize), b""):
            checksum.update(block)

    return checksum.hexdigest()


def __main__():
    directory = os.path.join(cfg.data_dir, "shape", "images")

    #如果存在该文件夹则移除掉
    if os.path.exists(directory): shutil.rmtree(directory)

    os.mkdir(directory)

    groups = ("train", "test")

    #row and column
    r, c = 224, 224

    for group in groups:
        dictionaries = []

        #生成256张图片
        for _ in range(num_images):
            #生成唯一的标识符
            identifier = uuid.uuid4()

            #画出随机形状
            image, objects = skimage.draw.random_shapes((r, c), 32, 2, 32,)

            filename = "{}.png".format(identifier)

            pathname = os.path.join(directory, filename)

            #保存图片
            skimage.io.imsave(pathname, image)

            if os.path.exists(pathname):
                dictionary = {
                    "image": {
                        "checksum": md5sum(pathname),
                        "pathname": pathname,
                        "shape": {
                            "r": r,
                            "c": c,
                            "channels": 3
                        }
                    },
                    "objects": []
                }

                for category, (bounding_box_r, bounding_box_c) in objects:
                    #左上角和右下角的坐标
                    minimum_r, maximum_r = bounding_box_r
                    minimum_c, maximum_c = bounding_box_c

                    object_dictionary = {
                        "bounding_box": {
                            "minimum": {
                                "r": minimum_r - 1,
                                "c": minimum_c - 1
                            },
                            "maximum": {
                                "r": maximum_r - 1,
                                "c": maximum_c - 1
                            }
                        },
                        "category": category
                    }

                    dictionary["objects"].append(object_dictionary)

                dictionaries.append(dictionary)

        filename = "{}.json".format(group)
        filename = os.path.join(cfg.data_dir, "shape", filename)
        with open(filename, "w") as stream:
            json.dump(dictionaries, stream)


if __name__ == "__main__":
    __main__()
