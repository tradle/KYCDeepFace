import io
import json
import boto3
import requests

from imageio import imread
from base64 import urlsafe_b64decode, b64decode, b64encode
from config import embedding_setup
from vision.utils.lang import format_timings, optional_bool_value, timed, lazy
from vision.kycdeepface import decode_embeddings, face_embeddings, face_match, format_embeddings

def api_gateway_handler (event, context={}):
    # Using the event body
    input = event['body']
    if 'isBase64Encoded' in event and event['isBase64Encoded']:
        input = b64decode(input).decode('utf-8')
    # Assuming content is JSON
    input = json.loads(input)
    # Executing the wrapped lambda function
    output = handler(input, context)
    # Encode to base64 json string
    output = json.dumps(output).encode('utf-8')
    output = b64encode(output).decode()
    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json"
        },
        "isBase64Encoded": True,
        "body": output
    }

def handler (event, context={}):
    if 'face_embeddings' in event:
        return face_embeddings_handler(event['face_embeddings'], context)
    if 'face_match' in event:
        return face_match_handler(event['face_match'], context)
    raise Exception('"face_match" or "face_embeddings" property required')

s3 = lazy(lambda: boto3.client('s3'))

def image_from_urlsafe_base64 (image):
    return io.BytesIO(urlsafe_b64decode(image))

def image_from_s3 (image_s3):
    res = s3.get_object(
        Bucket=image_s3['bucket'],
        Key=image_s3['key'],
        VersionId=image_s3['version']
    )
    return res['Body'].read()

def image_from_url (image_url):
    res = requests.get(image_url)
    return io.BytesIO(res.content)

def image_from_event (event):
    if 'image_urlsafe_b64' in event:
        return image_from_urlsafe_base64(event['image_urlsafe_b64'])

    if 'image_s3' in event:
        return image_from_s3(event['image_s3'])
    
    if 'image_url' in event:
        return image_from_url(event['image_url'])

    raise Exception('Missing input, use either image_url, image_urlsafe_b64, image_s3')

def face_embeddings_handler (event, _context):
    timings = {}
    image = imread(timed(timings, 'load', lambda: image_from_event(event)))
    add_image = optional_bool_value(event, 'return_image')
    embed_timings, faces = face_embeddings(embedding_setup, image)
    timings.update(embed_timings)
    timings.update(embedding_setup.timings)
    return {
        'faces': [format_embeddings(landmarks, image, embeddings, add_image) for landmarks, image, embeddings in faces],
        'timings': format_timings(timings)
    }

def face_match_handler (event, _context):
    timings = {}
    a = timed(timings, 'decode_a', lambda: decode_embeddings(event['embedding_a']))
    b = timed(timings, 'decode_b', lambda: decode_embeddings(event['embedding_b']))

    return {
        'similarity': timed(timings, 'match', lambda: round(face_match(a, b), 3)),
        'timings': format_timings(timings)
    }
