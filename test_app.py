# -*- coding: utf-8 -*-
# @Author: yirui
# @Date:   2021-11-17 21:05:26
# @Last Modified by:   yirui
# @Last Modified time: 2021-11-17 22:17:24
from base64 import b64decode, urlsafe_b64encode
import json
import λ as app
from vision.kycdeepface import encode_embeddings

def run_api(input):
    res = app.api_gateway_handler({ 'body': json.dumps(input) }, {})
    res = b64decode(res['body'])
    return json.loads(res)

def load_image(path):
    in_file = open(path, 'rb')
    data = in_file.read()
    return urlsafe_b64encode(data).decode()

res = run_api({
    'face_embeddings': {
        'return_image': True,
        'image_urlsafe_b64': load_image('./test_images/test_app_1.png')
    }
})

print(res)

embeddings = app.decode_embeddings(res['faces'][0]['embedding'])
print(embeddings)
print(encode_embeddings(embeddings))

res_two_face = run_api({
    'face_embeddings': {
        'return_image': True,
        'image_urlsafe_b64': load_image('./test_images/test_app_2.png')
    }
})

print(res_two_face)

res_target3 = run_api({
    'face_embeddings': {
        'return_image': True,
        'image_urlsafe_b64': load_image('./test_images/target3.png')
    }
})

print(res_target3)

res_test6 = run_api({
    'face_embeddings': {
        'return_image': True,
        'image_urlsafe_b64': load_image('./test_images/test6.png')
    }
})

print(res_test6)

print({
    'app_1 x app_1': run_api({ 'face_match': {
        "embedding_a": res['faces'][0]['embedding'],
        "embedding_b": res['faces'][0]['embedding']
    }})
})

print({
    'app_1 x app_2[0]': run_api({ 'face_match': {
        "embedding_a": res['faces'][0]['embedding'],
        "embedding_b": res_two_face['faces'][0]['embedding']
    }})
})

print({
    'app_1 x app_2[1]': run_api({ 'face_match': {
        "embedding_a": res['faces'][0]['embedding'],
        "embedding_b": res_two_face['faces'][1]['embedding']
    }})
})

print({
    'app_2[0] x app_2[1]': run_api({ 'face_match': {
        "embedding_a": res_two_face['faces'][0]['embedding'],
        "embedding_b": res_two_face['faces'][1]['embedding']
    }})
})

print({
    'target3 x test6': run_api({ 'face_match': {
        "embedding_a": res_target3['faces'][0]['embedding'],
        "embedding_b": res_test6['faces'][0]['embedding']
    }})
})

print({
    'app_1 x target3': run_api({ 'face_match': {
        "embedding_a": res['faces'][0]['embedding'],
        "embedding_b": res_target3['faces'][0]['embedding']
    }})
})

print({
    'app_2[0] x target3': run_api({ 'face_match': {
        "embedding_a": res_two_face['faces'][0]['embedding'],
        "embedding_b": res_target3['faces'][0]['embedding']
    }})
})

print({
    'app_2[1] x target3': run_api({ 'face_match': {
        "embedding_a": res_two_face['faces'][1]['embedding'],
        "embedding_b": res_target3['faces'][0]['embedding']
    }})
})

print({
    'app_1 x test6': run_api({ 'face_match': {
        "embedding_a": res['faces'][0]['embedding'],
        "embedding_b": res_test6['faces'][0]['embedding']
    }})
})

print({
    'app_2[0] x test6': run_api({ 'face_match': {
        "embedding_a": res_two_face['faces'][0]['embedding'],
        "embedding_b": res_test6['faces'][0]['embedding']
    }})
})

print({
    'app_2[1] x test6': run_api({ 'face_match': {
        "embedding_a": res_two_face['faces'][1]['embedding'],
        "embedding_b": res_test6['faces'][0]['embedding']
    }})
})
