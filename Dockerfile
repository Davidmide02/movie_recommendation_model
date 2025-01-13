FROM python:3.12.4-slim

# RUN pip install 

WORKDIR /app

COPY ["requirements.txt", "app.py","model/similarity", "./" ]
COPY ["requirements.txt", "app.py","model/movie_list", "./" ]

# install dependencies on the system vot
RUN pip install -r requirements.txt

EXPOSE 8501

ENTRYPOINT ["streamlit", "run"]

CMD [ "app.py" ]