from PIL import Image
import os

# 获取照片路径
imagepaths = []
for dirname, _, filenames in os.walk('./Final_Test'):
    for filename in filenames:
        path = os.path.join(dirname, filename)
        if path.endswith("ppm"):
            imagepaths.append(path)
# imagepaths = imagepaths[:1]
print(len(imagepaths))
for path in imagepaths:
    category = path.split("/")[3]
    # print(category)
    label = category[0:5]
    # print(label)
    img = Image.open(path)
    img.save("./Final_Test_PNG/" + label + ".png")
    # img.show()
