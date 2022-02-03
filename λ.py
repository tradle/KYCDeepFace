import json

from base64 import b64decode, b64encode
from config import embedding_setup
from vision.utils.lang import format_timings, optional_bool_value, timed, lazy
from vision.kycdeepface import decode_embeddings, face_embeddings, face_match, format_embeddings
from vision.utils.image_loader import image_from_event

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

def face_embeddings_handler (event, _context):
    timings = {}
    image = timed(timings, 'load', lambda: image_from_event(event, allow_fs=False))
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
