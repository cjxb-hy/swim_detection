import cv2
import os

data_dir = './data'
infer_file = './infer.res'
out_dir = './visual_res'

font = cv2.FONT_HERSHEY_SIMPLEX
path_to_im = dict()

for line in open(infer_file):
    img_path, _, _, _ = line.strip().split('\t')
    if img_path not in path_to_im:
        im = cv2.imread(os.path.join(data_dir, img_path))
        path_to_im[img_path] = im

for line in open(infer_file):
    img_path, label, conf, bbox = line.strip().split('\t')
    xmin, ymin, xmax, ymax = map(float, bbox.split(' '))
    xmin = int(round(xmin))
    ymin = int(round(ymin))
    xmax = int(round(xmax))
    ymax = int(round(ymax))

    img = path_to_im[img_path]
    if int(label) == 0:
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax),(0, 0, 255), 1)
	cv2.rectangle(img, (xmin, ymin), (xmin+124, ymin+16),(0, 0, 255), -1)
	cv2.putText(img,'swimmers',(xmin+1,ymin+13),font,0.8,(255, 255, 255),2)
    elif int(label) == 1:
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax),(0, 255, 0), 1)
	cv2.rectangle(img, (xmin, ymin), (xmin+186, ymin+16),(0, 255, 0), -1)
	cv2.putText(img,'sus_swimmers',(xmin+1,ymin+13),font,0.8,(255, 255, 255),2)
    else:
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax),(255, 0, 0), 1)
	cv2.rectangle(img, (xmin, ymin), (xmin+114, ymin+16),(255, 0, 0), -1)
	cv2.putText(img,'passerby',(xmin+1,ymin+13),font,0.8,(255, 255, 255),2)

for img_path in path_to_im:
    im = path_to_im[img_path]
    out_path = os.path.join(out_dir, os.path.basename(img_path))
    cv2.imwrite(out_path, im)

print 'Done.'
