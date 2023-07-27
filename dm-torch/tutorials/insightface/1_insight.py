import glob
import os.path as osp

import cv2
import numpy as np
from diffusers.utils import load_image
from insightface import model_zoo
from insightface.app.common import Face

from inner.tools import image_tools
from PIL import Image


class FaceHandler:
    def __init__(self,
                 dir: str,
                 swap_model_path: str,
                 allowed_modules=None,
                 **kwargs

                 ):
        self.dir = dir
        self.models = {}
        onnx_files = glob.glob(osp.join(self.dir, '*.onnx'))
        for onnx_file in onnx_files:
            model = model_zoo.get_model(onnx_file, **kwargs)
            if model is None:
                print('model not recognized:', onnx_file)
            elif allowed_modules is not None and model.taskname not in allowed_modules:
                print('model ignore:', onnx_file, model.taskname)
                del model
            elif model.taskname not in self.models and (allowed_modules is None or model.taskname in allowed_modules):
                print('find model:', onnx_file, model.taskname, model.input_shape, model.input_mean, model.input_std)
                self.models[model.taskname] = model
            else:
                print('duplicated model task type, ignore:', onnx_file, model.taskname)
                del model
        assert 'detection' in self.models
        self.det_model = self.models['detection']

        self.swap_model = model_zoo.get_model(swap_model_path, **kwargs)
        print("init success")

    def prepare(self, ctx_id, det_thresh=0.5, det_size=(640, 640)):
        self.det_thresh = det_thresh
        assert det_size is not None
        print('set det-size:', det_size)
        self.det_size = det_size
        for taskname, model in self.models.items():
            if taskname == 'detection':
                model.prepare(ctx_id, input_size=det_size, det_thresh=det_thresh)
            else:
                model.prepare(ctx_id)

    def get(self, img, max_num=0):
        bboxes, kpss = self.det_model.detect(img,
                                             max_num=max_num,
                                             metric='default')
        if bboxes.shape[0] == 0:
            return []
        ret = []
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i, 0:4]
            det_score = bboxes[i, 4]
            kps = None
            if kpss is not None:
                kps = kpss[i]
            face = Face(bbox=bbox, kps=kps, det_score=det_score)
            for taskname, model in self.models.items():
                if taskname == 'detection':
                    continue
                model.get(img, face)
            ret.append(face)
        return ret

    def draw_on(self, img, faces):
        dimg = img.copy()
        for i in range(len(faces)):
            face = faces[i]
            box = face.bbox.astype(np.int)
            color = (0, 0, 255)
            cv2.rectangle(dimg, (box[0], box[1]), (box[2], box[3]), color, 2)
            if face.kps is not None:
                kps = face.kps.astype(np.int)
                # print(landmark.shape)
                for l in range(kps.shape[0]):
                    color = (0, 0, 255)
                    if l == 0 or l == 3:
                        color = (0, 255, 0)
                    cv2.circle(dimg, (kps[l][0], kps[l][1]), 1, color,
                               2)
            if face.gender is not None and face.age is not None:
                cv2.putText(dimg, '%s,%d' % (face.sex, face.age), (box[0] - 1, box[1] - 4), cv2.FONT_HERSHEY_COMPLEX,
                            0.7, (0, 255, 0), 1)

            # for key, value in face.items():
            #    if key.startswith('landmark_3d'):
            #        print(key, value.shape)
            #        print(value[0:10,:])
            #        lmk = np.round(value).astype(np.int)
            #        for l in range(lmk.shape[0]):
            #            color = (255, 0, 0)
            #            cv2.circle(dimg, (lmk[l][0], lmk[l][1]), 1, color,
            #                       2)
        return dimg


app = FaceHandler(dir='/root/autodl-tmp/models/insightface/antelopev2/models',
                  swap_model_path="/root/autodl-tmp/models/insightface/antelopev2/swap.onnx")
app.prepare(
    ctx_id=0,
    det_thresh=0.5,
    det_size=(640, 640)
)


def image_detect(l):
    image_tools.show_image(l)
    faces = app.get(cv2.cvtColor(np.array(l), cv2.COLOR_RGB2BGR))
    rimg = app.draw_on(cv2.cvtColor(np.array(l), cv2.COLOR_RGB2BGR), faces)
    image_tools.show_image(Image.fromarray(cv2.cvtColor(rimg, cv2.COLOR_BGR2RGB)))


def get_face_single(img_data: np.ndarray, face_index=0, det_size=(640, 640)):
    face = app.get(img_data)

    if len(face) == 0 and det_size[0] > 320 and det_size[1] > 320:
        det_size_half = (det_size[0] // 2, det_size[1] // 2)
        return get_face_single(img_data, face_index=face_index, det_size=det_size_half)

    try:
        return sorted(face, key=lambda x: x.bbox[0])[face_index]
    except IndexError:
        return None


def swap_face(
        source_img: Image,
        target_img: Image,
):
    if isinstance(source_img, str):  # source_img is a base64 string
        import base64, io
        if 'base64,' in source_img:  # check if the base64 string has a data URL scheme
            base64_data = source_img.split('base64,')[-1]
            img_bytes = base64.b64decode(base64_data)
        else:
            # if no data URL scheme, just decode
            img_bytes = base64.b64decode(source_img)
        source_img = Image.open(io.BytesIO(img_bytes))
    source_img = cv2.cvtColor(np.array(source_img), cv2.COLOR_RGB2BGR)
    target_img = cv2.cvtColor(np.array(target_img), cv2.COLOR_RGB2BGR)
    target_face = get_face_single(target_img, face_index=0)
    source_face = get_face_single(source_img, face_index=0)
    result = target_img
    result = app.swap_model.get(
        result,
        target_face,
        source_face
    )
    result_image = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    return result_image


if __name__ == '__main__':
    # image_detect()
    src_image = load_image(
        "https://img0.baidu.com/it/u=2251306,2910716108&fm=253&fmt=auto&app=138&f=JPEG?w=500&h=500"
    )
    dst_image = load_image(
        "https://i1.hdslb.com/bfs/archive/e8b007745804efdbd6e6e7cfc26beb4302583e07.jpg"
    )
    image_detect(src_image)
    image_detect(dst_image)
    image_tools.show_image(swap_face(src_image, dst_image))
