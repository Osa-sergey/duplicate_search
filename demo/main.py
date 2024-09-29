import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, APIRouter, Request
from pymilvus import MilvusClient
from pydantic import BaseModel
import requests
import shutil
import uvicorn
import pyffmpeg

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from PIL import Image
from torchvision import transforms
from glob import glob

class VideoLinkRequest(BaseModel):
    videoLink: str


# Define response model
class VideoLinkResponse(BaseModel):
    is_duplicate: bool
    duplicate_for: str | None

router = APIRouter()

@router.post("/check-video-duplicate", response_model=VideoLinkResponse, tags=["API для проверки дубликатов видео"])
async def check_video_duplicate(url_link: VideoLinkRequest, request: Request):
    video_url = url_link.videoLink

    try:
        video_file_name = download_video(video_url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download video: {str(e)}")

    key_images, uuid = get_keyframes(video_file_name)

    width = 256  # Может изменяться

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((width, width)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    images = [transform(image) for image in key_images]
    images = torch.stack(images)

    app = request.app
    embeds = app.model(images.to('cpu'))
    vector = torch.mean(embeds, dim=0).tolist()
    trashold = 0.5  # Трэешхолд, по которому отбирается дубликат

    neighbours = get_k_neighbours(app, vector, 1)

    candidate = neighbours[0][0]
    is_duplicate = False
    duplicate_for = None
    if candidate['distance'] > trashold:
        is_duplicate = True
        duplicate_for = candidate['id']

    # удаление обработанного файла
    if os.path.exists(video_file_name):
        os.remove(video_file_name)


    return VideoLinkResponse(
        is_duplicate=is_duplicate,
        duplicate_for=duplicate_for
    )


def get_k_neighbours(app, vector, k=5):
    # Параметры поиска с указанием nprobe
    search_params = {
        "metric_type": "COSINE",
        "nprobe": 20
    }

    results = app.client.search(
        collection_name="image_embeddings",
        data=[vector],
        output_fields=["created"],
        search_params=search_params,
        limit=k
    )

    return results

def download_video(video_url: str) -> str:
    format = video_url[video_url.rfind('.'):]
    local_filename = os.path.join('downloads', f'{video_url.split("/")[-1]}')

    with requests.get(video_url, stream=True) as r:
        if r.status_code != 200:
            raise Exception(f"Error downloading video: {r.status_code}")

        with open(local_filename, 'wb') as f:
            shutil.copyfileobj(r.raw, f)

    return local_filename

def get_keyframes(video_file_name):
    out_dir = 'key_frames'
    ff = pyffmpeg.FFmpeg()
    output_file_name = os.path.join(out_dir, os.path.basename(video_file_name.split('.')[0]))
    try:
        os.makedirs(output_file_name, exist_ok=True)
        ff.options(
            f"-i {os.path.abspath(video_file_name)} -vf select='eq(pict_type\\,I)' -vsync vfr {os.path.abspath(output_file_name)}/%03d.jpg")
        keyframe_files = glob(os.path.join(output_file_name, '*.jpg'))

        if len(keyframe_files) == 0:
            print(output_file_name)

        images = [Image.open(keyframe_path) for keyframe_path in keyframe_files]
    except Exception as e:
        print(e)
        images = []
    return images, output_file_name

class MyEfficientNetEmbedding(nn.Module):
    def __init__(self, embedding_dim=128):
        super(MyEfficientNetEmbedding, self).__init__()
        self.efficientnet = models.efficientnet_b0()
        self.efficientnet.classifier = nn.Identity()
        self.fc = nn.Linear(1280, embedding_dim)

    def forward(self, x):
        x = self.efficientnet(x)
        x = self.fc(x)
        return F.normalize(x, p=2, dim=1)  # Нормализация эмбеддингов

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    os.makedirs('downloads', exist_ok=True)
    os.makedirs('key_frames', exist_ok=True)
    await load_model(app)
    await load_milvus(app)
    yield


async def load_model(app: FastAPI):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model = MyEfficientNetEmbedding(embedding_dim=128).to(device)
    print('start load weights')
    states = torch.load('model.pth', map_location=torch.device('cpu'))
    model.load_state_dict(states)
    print('End model init')
    app.model = model


async def load_milvus(app: FastAPI):
    client = MilvusClient(
        uri="http://localhost:19530"
    )
    app.client = client



def main():
    app = FastAPI(
        title="Video Duplicate Checker API",
        version="1.0.0",
        description="API for checking video duplicates",
        lifespan=lifespan
    )
    app.include_router(router)
    uvicorn.run(app, port=8001, host='0.0.0.0')


if __name__ == '__main__':
   main()
