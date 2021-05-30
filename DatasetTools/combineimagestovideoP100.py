import numpy as np
import glob
import cv2
import os


def saveonecameratovideo(PATH, outputfile):
    frame_width = 1920
    frame_height = 1280
    #out = cv2.VideoWriter('project.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
    out = cv2.VideoWriter(outputfile, cv2.VideoWriter_fourcc(
            'm', 'p', '4', 'v'), 5, (frame_width, frame_height))
    framenum=5152
    allcameras=["FRONT_IMAGE", "FRONT_LEFT_IMAGE", "FRONT_RIGHT_IMAGE", "SIDE_LEFT_IMAGE", "SIDE_RIGHT_IMAGE"]

    imagefiles_pattern=glob.glob(os.path.join(PATH, '*.jpg'))
    print(f'Total image files: {len(imagefiles_pattern)}')
    for fileidx in range(framenum+1):
        imagefile=str(fileidx)+'_'+allcameras[0]+".jpg"
        imagefile_path=os.path.join(PATH, imagefile)
        print(imagefile_path)
        if os.path.exists(imagefile_path):
            img = cv2.imread(imagefile_path)
            height, width, layers = img.shape
            size = (width,height)
            out.write(img)
        else:
            print(f'File not available, fileidx: {fileidx}')
    out.release()

# def concat_tile_resize(im_list_2d, interpolation=cv2.INTER_CUBIC):
#     im_list_v = [hconcat_resize_min(im_list_h, interpolation=cv2.INTER_CUBIC) for im_list_h in im_list_2d]
#     return vconcat_resize_min(im_list_v, interpolation=cv2.INTER_CUBIC)

def vconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    w_min = min(im.shape[1] for im in im_list)
    im_list_resize = [cv2.resize(im, (w_min, int(im.shape[0] * w_min / im.shape[1])), interpolation=interpolation)
                      for im in im_list]
    return cv2.vconcat(im_list_resize)

def saveallcameratovideo(PATH, outputfile):
    frame_width = 1920
    frame_height = 870 #1280
    #out = cv2.VideoWriter('project.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
    out = cv2.VideoWriter(outputfile, cv2.VideoWriter_fourcc(
            'm', 'p', '4', 'v'), 5, (frame_width, frame_height))
    framenum=5152
    allcameras=["FRONT_IMAGE", "FRONT_LEFT_IMAGE", "FRONT_RIGHT_IMAGE", "SIDE_LEFT_IMAGE", "SIDE_RIGHT_IMAGE"]

    imagefiles_pattern=glob.glob(os.path.join(PATH, '*.jpg'))
    totalimagefiles=int(len(imagefiles_pattern)/5)
    print(f'Total image files: {totalimagefiles}')
    for fileidx in range(totalimagefiles):
        front_left_imagefile_path=os.path.join(PATH, str(fileidx)+'_'+allcameras[1]+".jpg")
        if os.path.exists(front_left_imagefile_path)==False:
            break
        img_front_left = cv2.imread(front_left_imagefile_path)

        front_imagefile_path=os.path.join(PATH, str(fileidx)+'_'+allcameras[0]+".jpg")
        img_front = cv2.imread(front_imagefile_path)

        front_right_imagefile_path=os.path.join(PATH, str(fileidx)+'_'+allcameras[2]+".jpg")
        img_front_right = cv2.imread(front_right_imagefile_path)

        im_front = cv2.hconcat([img_front_left, img_front, img_front_right])

        side_left_imagefile_path=os.path.join(PATH, str(fileidx)+'_'+allcameras[3]+".jpg")
        img_side_left = cv2.imread(side_left_imagefile_path)

        side_right_imagefile_path=os.path.join(PATH, str(fileidx)+'_'+allcameras[4]+".jpg")
        img_side_right = cv2.imread(side_right_imagefile_path)

        im_side = cv2.hconcat([img_side_left, img_side_right])

        combinedimage=vconcat_resize_min([im_front, im_side])
        combinedimagesmall = cv2.resize(combinedimage, (0,0), fx=0.5, fy=0.5) #cut by half
        height, width, layers = combinedimagesmall.shape
        size = (width,height)
        print(f"Combined image size:",size )#1920, 870
        #cv2.imwrite('/home/010796032/MyRepo/myoutputs/outputallcamera0526_detectron899k_val6_combine/'+str(fileidx)+'.jpg', combinedimagesmall) #write to image folder
        out.write(combinedimagesmall)
    out.release()

if __name__ == "__main__":
    outputfile='/Developer/MyRepo/output/529tf500kval0_combine.mp4'
    PATH='/Developer/MyRepo/output/529tf500kval0/'
    #saveonecameratovideo(PATH, outputfile)
    saveallcameratovideo(PATH, outputfile)
# for filename in glob.glob(os.path.join(PATH, '*.jpg')):
#     img = cv2.imread(filename)
#     height, width, layers = img.shape
#     size = (width,height)
#     out.write(img)

# out.release()