import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from identification.process_file import ProcessFile
from identification.plot_graph import PlotGraph
from PIL import Image
import io
import torch
from transformers import DetrForObjectDetection, DetrImageProcessor
import base64
import matplotlib
matplotlib.use('Agg')
import boto3
import json
import matplotlib.colors as colors



s3_client = boto3.client('s3', 
    aws_access_key_id=os.environ.get('ACCESS_KEY'),
    aws_secret_access_key=os.environ.get('SECRET_KEY'))

def get_file_from_s3(bucket, key, path):
    if not os.path.exists(path):
        try:
            s3_client.head_object(Bucket=bucket, Key=key)
            s3_client.download_file(bucket, key, path)
            print("Baixado com sucesso")
        except Exception as e:
            print(f"Erro: {e}")
    else:
        print("Arquivo ja esta baixado")

app = Flask(__name__)
CORS(app, origins=["http://localhost:3008",
     "https://identification-uniform-interface.vercel.app"], supports_credentials=True)

# Definindo dispositivo de hardware
if torch.cuda.is_available():
    device = torch.device('cuda')  # nvidia
elif torch.backends.mps.is_available():
    device = torch.device('mps')  # apple
else:
    device = torch.device('cpu')  # cpu

model_path = os.path.join(os.getcwd(), 'trained-model', 'custom-model')

# Captura arquivos do modelo no S3
get_file_from_s3('detections-bucket', 'trained-model/custom-model/pytorch_model.bin', os.path.join(model_path, 'pytorch_model.bin'))
get_file_from_s3('detections-bucket', 'trained-model/custom-model/config.json', os.path.join(model_path, 'config.json'))

if os.path.exists(os.path.join(model_path, 'pytorch_model.bin')) and os.path.exists(os.path.join(model_path, 'config.json')):
    model = DetrForObjectDetection.from_pretrained(model_path).to(device)

processor = DetrImageProcessor.from_pretrained('facebook/detr-resnet-50')
colors_dict = dict(colors.TABLEAU_COLORS, **colors.XKCD_COLORS, **colors.CSS4_COLORS)


@app.route('/process', methods=['POST'])
def process():
    if 'image' not in request.files:
        return {'error': 'Imagem não encontrada.'}, 400

    image = request.files['image']

    if not 'model' in locals():
        model = DetrForObjectDetection.from_pretrained(model_path).to(device)

    # Processar o frame usando o modelo de rede neural
    process_file_class = ProcessFile(model, processor, device, colors_dict)
    processed_frame, response_detec = process_file_class.run(image)

    # Converter o frame processado para o formato PIL Image
    processed_image = Image.fromarray(processed_frame)

    # Salvar a imagem processada em um buffer e retorna a resposta
    buf = io.BytesIO()
    processed_image.save(buf, format='JPEG')
    img_str = base64.b64encode(buf.getvalue()).decode('utf-8')

    return jsonify({
        'image': img_str,
        'detections': response_detec
    })


@app.route('/plot', methods=['POST'])
def plot():
    data = json.loads(request.form.to_dict()['data'])
    plot_graph = PlotGraph()

    if len(data) == 0:
        # Envia o grafico gerado
        response = plot_graph.empty_graph()
    else:
        # Envia um grafico em branco
        response = plot_graph.run(data)

    return jsonify(response)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5003)
