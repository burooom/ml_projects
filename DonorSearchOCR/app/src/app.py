import uvicorn
import argparse
import logging
from model import prepare_davit, evaluate_image
from fastapi import FastAPI, UploadFile, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

app = FastAPI()

app.mount("/tmp", StaticFiles(directory="tmp"), name='images')
templates = Jinja2Templates(directory="templates")

app_logger = logging.getLogger(__name__)
app_logger.setLevel(logging.INFO)
app_handler = logging.StreamHandler()
app_formatter = logging.Formatter("%(name)s %(asctime)s %(levelname)s %(message)s")
app_handler.setFormatter(app_formatter)
app_logger.addHandler(app_handler)

INP_SIZE = 512
DEVICE = "cpu"
MODEL_NAME = 'davit_huge_fl'


@app.get("/health")
def orient():
    return {"status": 'OK'}


@app.get("/")
def main(request: Request):
    return templates.TemplateResponse("start_form.html",
                                      {"request": request})


@app.post("/predict-recognize")
def process_request(file: UploadFile, request: Request):
    model = prepare_davit(MODEL_NAME, '/app/weights/davit_weights.pt', 4, DEVICE)

    """save file to the local folder and send the image to the process function"""
    class_name = {0:'0', 1:'90', 2:'180', 3:'270'}
    img_tmp_path = "tmp/" + file.filename
    app_logger.info(f'loading image to {img_tmp_path}')
    with open(img_tmp_path, "wb") as fid:
        fid.write(file.file.read())

    rotated_img, ind, status = evaluate_image(img_tmp_path, model)
    if status == 'OK':
        app_logger.info(f'recognition: input image orientation was {class_name[ind]}')
        message = f"Input picture was recognized to be oriented at {class_name[ind]} degrees"
        rot_img_tmp_path = "tmp/" + "rotated_" + file.filename
        rotated_img.save(rot_img_tmp_path)
        return templates.TemplateResponse("recognition_form.html",
                                          {"request": request,
                                           "class_name": class_name[ind],
                                           "path": img_tmp_path,
                                           "path_rotated": rot_img_tmp_path,
                                           "message": message})
    else:
        app_logger.warning(f'some problems {status}')
        return templates.TemplateResponse("error_form.html",
                                          {"request": request,
                                           "result": status,
                                           "name": file.filename})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=8000, type=int, dest="port")
    parser.add_argument("--host", default="0.0.0.0", type=str, dest="host")
    args = vars(parser.parse_args())

    uvicorn.run(app, **args)
