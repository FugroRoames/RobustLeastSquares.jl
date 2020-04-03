FROM julia:1.3

WORKDIR /home

COPY Project.toml /home

RUN julia --project=/home -e "using Pkg; Pkg.instantiate();"

COPY . /home

CMD ["bash", "-l"]
