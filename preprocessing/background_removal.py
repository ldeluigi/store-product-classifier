# code taken from https://www.kaggle.com/vadbeg/opencv-background-removal and modified

def remove_background(img, threshold, use_mask=False):
  gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
  _, threshed = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)

  kernel_size = round(max(img.shape[0], img.shape[1]) * 0.02)
  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
  morphed = cv2.morphologyEx(threshed, cv2.MORPH_CLOSE, kernel)

  cnts = cv2.findContours(morphed, 
                          cv2.RETR_EXTERNAL,
                          cv2.CHAIN_APPROX_SIMPLE)[0] # should be [1] for cv2 version <= 4

  cnts = sorted(cnts, key=cv2.contourArea)

  mask = cv2.drawContours(threshed, [cnts[-1]], 0, [255], cv2.FILLED)

  x, y, w, h = cv2.boundingRect(cnts[-1])

  if use_mask:
    masked_data = cv2.bitwise_and(img, img, mask=mask)
    dst = masked_data[y: y + h, x: x + w]
    r, g, b = cv2.split(dst)
    alpha = mask[y: y + h, x: x + w]

    rgba = [r, g, b, alpha]
    dst = cv2.merge(rgba, 4)
  else:
    dst = img[y: y + h, x: x + w]

  return dst

n = 778
print(f'Index: {n}')
print(f'Class: {class_name(products_classes[n])}')
plot_grid([products[n], remove_background(products[n], 250)], 2, show_axis=True)
