# First stage - train model
FROM python:3.10.6-slim

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install -e .

EXPOSE 8000

CMD ["bash", "start.sh"]

#RUN ./scripts/train_save_model.py

# # Second stage - create service and get model from previous stage
# FROM python:3.10.6-slim

# WORKDIR /usr/src/app

# COPY requirements.txt ./
# RUN pip install --no-cache-dir -r requirements.txt

# COPY . .
# RUN pip install -e .

# COPY --from=trainer /usr/src/app/models/*.joblib ./models/

# EXPOSE 8000

# CMD ["bash", "start.sh"]
