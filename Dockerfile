FROM public.ecr.aws/lambda/python:3.7

COPY requirements.txt .

RUN pip3 install -r requirements.txt --target ${LAMBDA_TASK_ROOT}

COPY λ.py \
    config.py \
    test_app.py \
    ${LAMBDA_TASK_ROOT}/

COPY core ${LAMBDA_TASK_ROOT}/core
COPY dataloader ${LAMBDA_TASK_ROOT}/dataloader
COPY masks ${LAMBDA_TASK_ROOT}/masks
COPY models ${LAMBDA_TASK_ROOT}/models
COPY vision ${LAMBDA_TASK_ROOT}/vision

CMD [ "λ.handler" ]
