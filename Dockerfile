FROM python:3.11.11-slim

# RUN pip install 

WORKDIR /app

COPY ["requirements.txt", "app.py","./model","./" ]


# install dependencies on the system vot
RUN pip install -r requirements.txt

EXPOSE 8501

ENTRYPOINT ["streamlit", "run" ]

CMD [ "app.py" ]