import azure.functions as func
import logging
from onnx.onnx_cpp2py_export import ONNX_ML
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import torch

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

@app.route(route="ocr_doctr")
def ocr_doctr(req: func.HttpRequest) -> func.HttpResponse:
    # get the path and filename of the file for processing
    doc_filename = req.params.get('doc_filename')
    try:
        # determine whether to use GPU or CPU
        use_gpu = False
        if torch.cuda.is_available() and use_gpu==True:
            # device = torch.device("cuda:0")
            device = torch.device("cuda")
            print("Using GPU:", torch.cuda.get_device_name(0))
        else:
            device = torch.device("cpu")
            print("Using CPU")
        # load model
        print("Load model...")
        model = ocr_predictor('fast_base','crnn_mobilenet_v3_small', pretrained=True).to(device)
        # load doc
        print("Load doc ...")
        doc = DocumentFile.from_images(doc_filename)
        # run OCR
        print("Run OCR...")
        result = model(doc)
        # get string
        string_result = result.render()
        logging.info('Python HTTP trigger function processed a request.')
        return func.HttpResponse(string_result, status_code=200)
    except Exception as e:
        err_str = "There was an issue with processing your request.  Here is the error message: "+str(e)
        return func.HttpResponse(err_str, status_code=200)