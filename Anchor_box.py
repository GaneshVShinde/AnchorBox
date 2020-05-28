# coding: utf-8
#%%
import numpy as np
# import keras.backend as K
import cv2
from bounding_box_utils import convert_coordinates
from draw_bbox import *
#%%
def generate_anchor_boxes_for_layer(   feature_map_size,
                                        aspect_ratios,
                                        this_scale,
                                        next_scale,
                                        this_steps=None,
                                        this_offsets=None,
                                        diagnostics=False,
                                        two_boxes_for_ar1=False,
                                        clip_boxes=False,
                                        normalize_coords=False,
                                        coords='centroids'):
        img_height=300
        img_width=300
        size = min(img_height, img_width)
        wh_list = []
        for ar in aspect_ratios:
            if (ar == 1):
                box_height = box_width = this_scale * size
                wh_list.append((box_width, box_height))
                if two_boxes_for_ar1:
                    box_height = box_width = np.sqrt(this_scale * next_scale) * size
                    wh_list.append((box_width, box_height))
            else:
                box_width = this_scale * size * np.sqrt(ar)
                box_height = this_scale * size / np.sqrt(ar)
                wh_list.append((box_width, box_height))
        wh_list = np.array(wh_list)
        n_boxes = len(wh_list)

        if (this_steps is None):
            step_height = img_height / feature_map_size[0]
            step_width = img_width / feature_map_size[1]
        else:
            if isinstance(this_steps, (list, tuple)) and (len(this_steps) == 2):
                step_height = this_steps[0]
                step_width = this_steps[1]
            elif isinstance(this_steps, (int, float)):
                step_height = this_steps
                step_width = this_steps
        if (this_offsets is None):
            offset_height = 0.5
            offset_width = 0.5
        else:
            if isinstance(this_offsets, (list, tuple)) and (len(this_offsets) == 2):
                offset_height = this_offsets[0]
                offset_width = this_offsets[1]
            elif isinstance(this_offsets, (int, float)):
                offset_height = this_offsets
                offset_width = this_offsets
        cy = np.linspace(offset_height * step_height, (offset_height + feature_map_size[0] - 1) * step_height, feature_map_size[0])
        cx = np.linspace(offset_width * step_width, (offset_width + feature_map_size[1] - 1) * step_width, feature_map_size[1])
        cx_grid, cy_grid = np.meshgrid(cx, cy)
        cx_grid = np.expand_dims(cx_grid, -1) # This is necessary for np.tile() to do what we want further down
        cy_grid = np.expand_dims(cy_grid, -1) # This is necessary for np.tile() to do what we want further down

        boxes_tensor = np.zeros((feature_map_size[0], feature_map_size[1], n_boxes, 4))

        boxes_tensor[:, :, :, 0] = np.tile(cx_grid, (1, 1, n_boxes)) # Set cx
        boxes_tensor[:, :, :, 1] = np.tile(cy_grid, (1, 1, n_boxes)) # Set cy
        boxes_tensor[:, :, :, 2] = wh_list[:, 0] # Set w
        boxes_tensor[:, :, :, 3] = wh_list[:, 1] # Set h

        boxes_tensor = convert_coordinates(boxes_tensor, start_index=0, conversion='centroids2corners')

        if clip_boxes:
            x_coords = boxes_tensor[:,:,:,[0, 2]]
            x_coords[x_coords >= img_width] = img_width - 1
            x_coords[x_coords < 0] = 0
            boxes_tensor[:,:,:,[0, 2]] = x_coords
            y_coords = boxes_tensor[:,:,:,[1, 3]]
            y_coords[y_coords >= img_height] = img_height - 1
            y_coords[y_coords < 0] = 0
            boxes_tensor[:,:,:,[1, 3]] = y_coords

        if normalize_coords:
            boxes_tensor[:, :, :, [0, 2]] /= img_width
            boxes_tensor[:, :, :, [1, 3]] /= img_height

        if coords == 'centroids':
            # Convert `(xmin, ymin, xmax, ymax)` back to `(cx, cy, w, h)`.
            boxes_tensor = convert_coordinates(boxes_tensor, start_index=0, conversion='corners2centroids', border_pixels='half')
        elif coords == 'minmax':
            # Convert `(xmin, ymin, xmax, ymax)` to `(xmin, xmax, ymin, ymax).
            boxes_tensor = convert_coordinates(boxes_tensor, start_index=0, conversion='corners2minmax', border_pixels='half')

        if diagnostics:
            return boxes_tensor, (cy, cx), wh_list, (step_height, step_width), (offset_height, offset_width)
        else:
            return boxes_tensor


# #%%
# img=cv2.imread('./Assignments/Assignments/Test_Original_image.jpg')
# imgs=np.array([img])
# imgs.shape
# t_imgs=K.constant(imgs)
#%%
anhor_boxes=generate_anchor_boxes_for_layer(feature_map_size=(18,18),
                                        aspect_ratios=[1],
                                        this_scale=0.13,
                                        next_scale=0.1,
                                        this_steps=16,
                                        this_offsets=0.5,
                                        diagnostics=False,
                                        two_boxes_for_ar1=False,
                                        clip_boxes=True,
                                        normalize_coords=False,
                                        coords='centroids')
#%%
boxes_tensor = convert_coordinates(anhor_boxes, start_index=0, conversion='centroids2corners')
img=cv2.imread('./Test_Original_image.jpg')
imgs=np.array([img])
nth_bx=4
for j in range(1,18):
   if j%nth_bx==0:
      for i in range(1,18):
         if i%nth_bx==0 :
            x1,y1,x2,y2=boxes_tensor[j,i,0,:]
            draw_bbox(img,int(x1),int(y1),int(x2),int(y2))

cv2.imwrite('Output_4A_bx.jpg',img)



#%%
img=cv2.imread('./Test_Original_image.jpg')
imgs=np.array([img])
nth_bx=2
for j in range(1,18):
   if j%nth_bx==0:
      for i in range(1,18):
         if i%nth_bx==0 :
            x1,y1,x2,y2=boxes_tensor[j,i,0,:]
            draw_bbox(img,int(x1),int(y1),int(x2),int(y2))

cv2.imwrite('Output_2A_bx.jpg',img)


# %%
